
#include "general_inference.h"

#include <iostream>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <iterator>
#include <fstream>
#include <string>

#include "hiaiengine/log.h"
#include "ascenddk/ascend_ezdvpp/dvpp_process.h"
#include "opencv2/opencv.hpp"
#include "tool_api.h"

using hiai::Engine;
using hiai::ImageData;
using namespace std;
using namespace ascend::utils;

namespace
{
// model_path parameter key in graph.config
const string kModelPathParamKey = "model_path";

// output port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// model process timeout
const uint32_t kAiModelProcessTimeout = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// length of image info array
//const uint32_t kImageInfoLength = 3;
const uint32_t kImageInfoLength = 3;


// vpc input image offset
const uint32_t kImagePixelOffsetEven = 1;
const uint32_t kImagePixelOffsetOdd = 2;
}


GeneralInference::GeneralInference() 
{
  ai_model_manager_ = nullptr;
}

HIAI_StatusT GeneralInference::Init( const hiai::AIConfig& config, const vector<hiai::AIModelDescription>& model_desc)
{
  HIAI_ENGINE_LOG("Start initialize!");

  // initialize aiModelManager
  if (ai_model_manager_ == nullptr)
  {
    MAKE_SHARED_NO_THROW(ai_model_manager_, hiai::AIModelManager);
    if (ai_model_manager_ == nullptr) 
    {
      ERROR_LOG("Failed to initialize AIModelManager.");
      return HIAI_ERROR;
    }
  }

  // get parameters from graph.config
  // set model path to AI model description
  hiai::AIModelDescription fd_model_desc;
  for (int index = 0; index < config.items_size(); index++) 
  {
    const ::hiai::AIConfigItem& item = config.items(index);
    // get model path
    if (item.name() == kModelPathParamKey)
    {
      const char* model_path = item.value().data();
      fd_model_desc.set_path(model_path);
    }
    // else: noting need to do
  }

  // initialize model manager
  vector<hiai::AIModelDescription> model_desc_vec;
  model_desc_vec.push_back(fd_model_desc);
  hiai::AIStatus ret = ai_model_manager_->Init(config, model_desc_vec);
  // initialize AI model manager failed
  if (ret != hiai::SUCCESS) 
  {
    HIAI_ENGINE_LOG(HIAI_GRAPH_INVALID_VALUE, "initialize AI model failed");
    ERROR_LOG("Failed to initialize AI model.");
    return HIAI_ERROR;
  }

  HIAI_ENGINE_LOG("End initialize!");
  return HIAI_OK;
}


bool GeneralInference::Inference(const shared_ptr<CusBatchImagePara>& images_batch_handle, const vector<ImageData<u_int8_t>>& batch_resized_images, vector<shared_ptr<hiai::IAITensor>>& output_data_vec, const shared_ptr<EngineTransT>& transdata, const uint32_t all_size)
{
    // copy original data

    transdata->b_info = images_batch_handle->b_info;
    transdata->imgs = images_batch_handle->v_img;
    transdata->console_params=images_batch_handle->console_params;

    // copy image data
    std::shared_ptr<uint8_t> temp = std::shared_ptr<uint8_t>(new uint8_t[all_size], std::default_delete<uint8_t[]>());

    uint32_t last_size = 0;

    for (uint32_t i = 0; i < batch_resized_images.size(); i++)
    {

        // copy memory according to each size
        uint32_t each_size = batch_resized_images[i].size * sizeof(uint8_t);
        HIAI_ENGINE_LOG("each input image size: %u", each_size);
        errno_t mem_ret = memcpy_s(temp.get() + last_size, all_size - last_size, batch_resized_images[i].data.get(), each_size);
        // memory copy failed, no need to inference, send original image
        if (mem_ret != EOK)
        {
            HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "prepare image data: memcpy_s() error=%d", mem_ret);
            transdata->status = false;
            transdata->msg = "HiAIInference Engine memcpy_s image data failed";
            // send data to engine output port 0
            SendData(kSendDataPort, "EngineTransT", std::static_pointer_cast<void>(transdata));
            return HIAI_ERROR;
        }
        last_size += each_size;

    }

    // neural buffer
    std::shared_ptr<hiai::AINeuralNetworkBuffer> neural_buf = std::shared_ptr< hiai::AINeuralNetworkBuffer>(new hiai::AINeuralNetworkBuffer(), std::default_delete<hiai::AINeuralNetworkBuffer>());
    neural_buf->SetBuffer((void*)temp.get(), all_size);

    // input data
    std::shared_ptr<hiai::IAITensor> input_data = std::static_pointer_cast<hiai::IAITensor>(neural_buf);
    std::vector<std::shared_ptr<hiai::IAITensor>> input_data_vec;
    input_data_vec.push_back(input_data);

   shared_ptr<hiai::AINeuralNetworkBuffer> info_buf = nullptr;
    MAKE_SHARED_NO_THROW(info_buf, hiai::AINeuralNetworkBuffer);
    if (neural_buf == nullptr)
    {
        HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "new AINeuralNetworkBuffer failed");
        return false;
    }

      float info_tmp[ batch_resized_images.size()][kImageInfoLength] = {
        {(float)batch_resized_images[0].height, (float)batch_resized_images[0].width, (float)batch_resized_images[0].depth},
         {(float)batch_resized_images[1].height, (float)batch_resized_images[1].width, (float)batch_resized_images[1].depth},
          {(float)batch_resized_images[2].height, (float)batch_resized_images[2].width, (float)batch_resized_images[2].depth},
           {(float)batch_resized_images[3].height, (float)batch_resized_images[3].width, (float)batch_resized_images[3].depth},

        };
    uint32_t bin_info_len = batch_resized_images.size() *kImageInfoLength* sizeof(float);
    shared_ptr<uint8_t> data_ptr = shared_ptr<uint8_t>(new uint8_t[bin_info_len], default_delete<uint8_t>());
    int mem_ret = memcpy_s(data_ptr.get(), bin_info_len, info_tmp, bin_info_len);
    if(mem_ret != 0)
    {
        cout<<"[LOG] image info input tensor memory copy failed"<<endl;
        HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "image info input tensor memory copy failed");
        return false;
    }
    info_buf->SetBuffer((void *)data_ptr.get(), bin_info_len);

    //image info tensor
    shared_ptr<hiai::IAITensor> img_info = static_pointer_cast<hiai::IAITensor>(info_buf);
    input_data_vec.push_back(img_info);


    // Call Process
    // 1. create output tensor
    hiai::AIContext ai_context;

    hiai::AIStatus ret = ai_model_manager_->CreateOutputTensor(input_data_vec, output_data_vec);
    // create failed, also need to send data to post process
    if (ret != hiai::SUCCESS)
    {
        HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "failed to create output tensor");
        transdata->status = false;
        transdata->msg = "HiAIInference Engine CreateOutputTensor failed";
        // send data to engine output port 0
        SendData(kSendDataPort, "EngineTransT", std::static_pointer_cast<void>(transdata));
        return HIAI_ERROR;
    }

    // 2. process
    HIAI_ENGINE_LOG("aiModelManager->Process start!");


    try
    {
    cout<<"[LOG] ai_model_manage_->Process >>>>>>>>>>>>>>>>>>>>>"<<endl;
    ret = ai_model_manager_->Process(ai_context, input_data_vec, output_data_vec, kAiModelProcessTimeout);
    }
    catch(std::bad_alloc)
    {
    cout<<"ai_model_manage_->Process can not work!!!!!!!!!!"<<endl;
    std::terminate();
    }

    // process failed, also need to send data to post process
    if (ret != hiai::SUCCESS)
    {
        HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "failed to process ai_model");
        transdata->status = false;
        transdata->msg = "HiAIInference Engine Process failed";
        // send data to engine output port 0
        SendData(kSendDataPort, "EngineTransT", std::static_pointer_cast<void>(transdata));
        return HIAI_ERROR;
    }
    cout<<"[LOG] ai_model_manage_->Process end >>>>>>>>>>>>>>>>>>>>>"<<endl;
    HIAI_ENGINE_LOG("aiModelManager->Process end!");

    return true;
}

bool GeneralInference::SendToEngine( const shared_ptr<CusBatchImagePara> &images_batch_handle)
{
  // can not discard when queue full
  HIAI_StatusT hiai_ret;
  do {
    hiai_ret = SendData(kSendDataPort, "CusBatchImagePara", static_pointer_cast<void>(images_batch_handle));
    // when queue full, sleep
    if (hiai_ret == HIAI_QUEUE_FULL)
     {
      HIAI_ENGINE_LOG("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL);

  // send failed
  if (hiai_ret != HIAI_OK)
  {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "call SendData failed, err_code=%d", hiai_ret);
    return false;
  }
  return true;
}



void GeneralInference::SendError(const std::string &err_msg, std::shared_ptr<CusBatchImagePara> & images_batch_handle)
{
    images_batch_handle->err_msg.error = true;
    images_batch_handle->err_msg.err_msg = err_msg;

  if (!SendToEngine(images_batch_handle))
  {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "SendData err_msg failed");

  }
}


HIAI_StatusT GeneralInference::ImagePreProcess(const shared_ptr<CusBatchImagePara>& images_batch_handle, const shared_ptr<hiai::ImageData<u_int8_t>>& src_img, ImageData<u_int8_t>& resized_img)
{

  // assemble resize param struct
    DvppBasicVpcPara dvpp_resize_param;
    dvpp_resize_param.input_image_type = INPUT_BGR;

    // get original image size and set to resize parameter
    int32_t width = src_img->width;
    int32_t height = src_img->height;

    dvpp_resize_param.src_resolution.height = height;
    dvpp_resize_param.src_resolution.width = width;

    // the value of crop_right and crop_left must be odd.
    //dvpp_resize_param.crop_right = src_img.width % 2 == 0 ? src_img.width - kImagePixelOffsetEven : src_img.width - kImagePixelOffsetOdd;
    //dvpp_resize_param.crop_down = src_img.height % 2 == 0 ? src_img.height - kImagePixelOffsetEven : src_img.height - kImagePixelOffsetOdd;

    // crop parameters, only resize, no need crop, so set original image size
   // set crop left-top point (need even number)
    dvpp_resize_param.crop_left = 0;
    dvpp_resize_param.crop_up = 0;
    // set crop right-bottom point (need odd number)
    uint32_t crop_right = ((width >> 1) << 1) - 1;
    uint32_t crop_down = ((height >> 1) << 1) - 1;
    dvpp_resize_param.crop_right = crop_right;
    dvpp_resize_param.crop_down = crop_down;

    // set destination resolution ratio (need even number)

    uint32_t dst_width = ((images_batch_handle->console_params.model_width) >> 1) << 1;
    uint32_t dst_height = ((images_batch_handle->console_params.model_height) >> 1) << 1;
    dvpp_resize_param.dest_resolution.width = dst_width;
    dvpp_resize_param.dest_resolution.height = dst_height;


    // set input image align or not
    dvpp_resize_param.is_input_align = false;

    //call
    DvppProcess dvpp_resize_img(dvpp_resize_param);

    DvppVpcOutput dvpp_out;
    int ret = dvpp_resize_img.DvppBasicVpcProc(src_img->data.get(), src_img->size, &dvpp_out);

    if (ret != kDvppOperationOk)
    {
        HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "call dvpp resize failed with code % d!", ret);
        return false;
    }

    // dvpp_out->pbuf
    // call success, set data and size
    resized_img.data.reset(dvpp_out.buffer, default_delete<u_int8_t[]>());
 
    resized_img.size = dvpp_out.size;

    resized_img.width = dst_width;
    resized_img.height = dst_height;

    src_img->width = dst_width;
    src_img->height = dst_height;
  
    return HIAI_OK;
}



HIAI_IMPL_ENGINE_PROCESS("general_inference", GeneralInference, INPUT_SIZE) 
{
  HIAI_StatusT ret = HIAI_OK;

  // arg0 is empty
  if (arg0 == nullptr) 
  {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "arg0 is empty.");
    return HIAI_ERROR;
  }

  // just send data when finished
  shared_ptr<CusBatchImagePara> images_batch_handle = static_pointer_cast<CusBatchImagePara>(arg0);


  if (images_batch_handle->is_finished)
  {
    if (SendToEngine(images_batch_handle)) 
    {
      return HIAI_OK;
    }
    SendError("Failed to send finish data. Reason: Inference SendData failed.", images_batch_handle);
    return HIAI_ERROR;
  }

  vector<ImageData<u_int8_t>> batch_resized_images;

  vector<CusImageData>::iterator it;

  uint32_t all_input_size = 0;

  uint32_t i=0;
  for (it = images_batch_handle->v_img.begin(); it != images_batch_handle->v_img.end(); it++)
  {

      // resize image
      ImageData<u_int8_t> resized_image;

      shared_ptr <hiai::ImageData<u_int8_t>> src_image = make_shared<hiai::ImageData<u_int8_t>>((*it).image_info.image_data);

      HIAI_StatusT vpc_ret = ImagePreProcess(images_batch_handle,src_image, resized_image);

      (*it).image_info.image_data.height=src_image->height;
      (*it).image_info.image_data.width=src_image->width;

      if (vpc_ret != HIAI_OK) 
      {
          cout<<"image pre process error"<<endl;
          HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "image pre process error");
          continue;
      }

      all_input_size += resized_image.size * sizeof(uint8_t);
      batch_resized_images.push_back(resized_image);

  }


  // input size is less than zero, do not need to inference
  if (all_input_size <= 0)
  {
      HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "all input image size=%u is less than zero", all_input_size);
      return HIAI_ERROR;
  }

  std::vector<std::shared_ptr<hiai::IAITensor>> output_data_vector;
  std::shared_ptr<EngineTransT> trans_data = std::make_shared<EngineTransT>();

  if (!Inference(images_batch_handle,batch_resized_images, output_data_vector, trans_data,all_input_size))
  {
      string err_msg = "Failed to deal file  Reason: inference failed.";
      SendError(err_msg, images_batch_handle);
      return HIAI_ERROR;
  }


  // generate output data
  trans_data->status = true;
  for (uint32_t i = 0; i < output_data_vector.size(); i++)
  {
      //cout<<"out "<<i<<endl;
      std::shared_ptr<hiai::AISimpleTensor> result_tensor = std::static_pointer_cast<hiai::AISimpleTensor>(output_data_vector[i]);
      OutputT out;
      out.size = result_tensor->GetSize();
      //cout<<"size :"<<out.size<<endl;
      out.data = std::shared_ptr<uint8_t>(new uint8_t[out.size], std::default_delete<uint8_t[]>());
      errno_t mem_ret = memcpy_s(out.data.get(), out.size, result_tensor->GetBuffer(), result_tensor->GetSize());
      //cout<<"output_data_vector:"<<output_data_vector[i]<<endl;

      // memory copy failed, skip this result
      if (mem_ret != EOK)
      {
          HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "dealing results: memcpy_s() error=%d", mem_ret);
          continue;
      }
      trans_data->output_datas.push_back(out);
  }
//cout<<"Generate output data."<<endl;

  // send results and original image data to post process (port 0)
  HIAI_StatusT hiai_ret = SendData(kSendDataPort, "EngineTransT", std::static_pointer_cast<void>(trans_data));

  //cout<<"Send results and original image data to post engine."<<endl;
  HIAI_ENGINE_LOG("End process!");
  return hiai_ret;
}
