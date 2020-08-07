/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "general_post.h"

#include <iostream>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "hiaiengine/log.h"
#include "opencv2/opencv.hpp"
#include "tool_api.h"

using hiai::Engine;
using namespace std;

namespace 
{
// callback port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// size of output tensor vector should be 2.
const uint32_t kOutputTensorSize = 2;
const uint32_t kOutputNumIndex = 0;
const uint32_t kOutputTesnorIndex = 1;

const uint32_t kCategoryIndex = 2;
const uint32_t kScorePrecision = 3;

// bounding box line solid
const uint32_t kLineSolid = 2;

// output image prefix
const string kOutputFilePrefix = "out_";

// boundingbox tensor shape
const static std::vector<uint32_t> kDimDetectionOut = {4, 64, 304, 8};

// num tensor shape
const static std::vector<uint32_t> kDimBBoxCnt = {4,32};

// opencv draw label params.
const double kFountScale = 0.5;
const cv::Scalar kFontColor(0, 0, 255);
const uint32_t kLabelOffset = 11;
const string kFileSperator = "/";

// opencv color list for boundingbox
const vector<cv::Scalar> kColors {
  cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255), cv::Scalar(50, 205, 50),
  cv::Scalar(139, 85, 26)};
// output tensor index
enum BBoxIndex {kTopLeftX, kTopLeftY, kLowerRigltX, kLowerRightY, kScore};


// detection function return value
const int32_t kFdFunSuccess = 0;
const int32_t kFdFunFailed = -1;

// need to deal results when index is 2
const int32_t kDealResultIndex = 2;

// each results size
const int32_t kEachResultSize = 7;
}
 // namespace

HIAI_StatusT GeneralPost::Init( const hiai::AIConfig &config, const vector<hiai::AIModelDescription> &model_desc) 
{
  // do noting
  return HIAI_OK;
}

bool GeneralPost::SendSentinel() 
{
cout<<"[LOG] Start SendSentinel:"<<endl;
  // can not discard when queue full
  HIAI_StatusT hiai_ret = HIAI_OK;
  shared_ptr<string> sentinel_msg(new (nothrow) string);
  do {
    hiai_ret = SendData(kSendDataPort, "string", static_pointer_cast<void>(sentinel_msg));
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

/////////////////////////////////////////////////////////////////////////////////////////////

HIAI_StatusT GeneralPost::HandleResults( const std::shared_ptr<EngineTransT>& inference_res) 
{
    cout<<"[LOG] Start HandleResuts:"<<endl;
    HIAI_StatusT status = HIAI_OK;
    std::vector<CusImageData> img_vec = inference_res->imgs;   //warning
    std::vector<OutputT> output_data_vec = inference_res->output_datas;
    cout<<"[LOG] output_data_vec size: "<<output_data_vec.size()<<endl;
    cout<<"[LOG] batch size: "<<inference_res->b_info.batch_size<<endl;

    uint32_t *num_buffer = reinterpret_cast<uint32_t *>(output_data_vec[kOutputNumIndex].data.get());
    float *bbox_buffer = reinterpret_cast<float *>(output_data_vec[kOutputTesnorIndex].data.get());

    cout<<"[LOG] num_buffer size:"<<output_data_vec[kOutputNumIndex].size<<endl;
    cout<<"[LOG] bbox_buffer size:"<<output_data_vec[kOutputTesnorIndex].size<<endl;

    Tensor<uint32_t> tensor_num;
    Tensor<float> tensor_bbox;


    bool ret=true;

    ret=tensor_num.FromArray(num_buffer,kDimBBoxCnt);
     if (!ret)
      {
          ERROR_LOG("Failed to resolve tensor from array.");
          status=HIAI_ERROR;
      }
      ret = tensor_bbox.FromArray(bbox_buffer, kDimDetectionOut);
      if (!ret)
      {
          ERROR_LOG("Failed to resolve tensor from array.");
           status=HIAI_ERROR;
      }

      cout<<"[LOG] tensor_num.Size:"<<tensor_num.Size()<<endl;
      cout<<"[LOG] tensor_bbox.Size:"<<tensor_bbox.Size()<<endl;
/*
    for (uint32_t i=0;i<tensor_num.Size();i++)
      {
      cout<<"tensor num "<<i<<": "<<tensor_num[i]<<endl;
      }

      for (uint32_t i=0;i<tensor_bbox.Size();i++)
      {
      cout<<"tensor_bbox "<<i<<": "<<tensor_bbox[i]<<endl;
      }
*/
  for (uint32_t ind = 0; ind < inference_res->b_info.batch_size; ind++)
{
    vector<BoundingBox> bboxes;
    cout<<"[LOG] Detected: "<<ind<<" picture."<<endl;
    for (uint32_t attr = 0; attr < inference_res->console_params.output_nums; ++attr)
    {
       uint32_t box_ind=ind*kDimBBoxCnt[1]+attr;
       //cdcout<<" box_ind="<< box_ind<<endl;
      for (uint32_t bbox_idx = 0; bbox_idx < tensor_num[ box_ind]; ++bbox_idx)
      {
      /*
      cout<<"attr="<<attr<<endl;
      cout<<"box_ind="<<box_ind<<endl;
      cout<<"bbox_idx="<<bbox_idx<<endl;
      */

      uint32_t class_idx = attr * kCategoryIndex;
      /*
      cout<<"ind="<<ind<<endl;
      cout<<"class_idx="<<class_idx<<endl;
      cout<<"bbox_idx="<<bbox_idx<<endl;
      */

     uint32_t lt_x = tensor_bbox(ind,class_idx, bbox_idx, BBoxIndex::kTopLeftX);
     uint32_t lt_y = tensor_bbox(ind,class_idx, bbox_idx, BBoxIndex::kTopLeftY);
     uint32_t rb_x = tensor_bbox(ind,class_idx, bbox_idx, BBoxIndex::kLowerRigltX);
     uint32_t rb_y = tensor_bbox(ind,class_idx, bbox_idx, BBoxIndex::kLowerRightY);
     float score = tensor_bbox(ind,class_idx, bbox_idx, BBoxIndex::kScore);
     bboxes.push_back( {lt_x, lt_y, rb_x, rb_y, attr, score});

        /*
     cout<<"lt_x="<<lt_x<<endl;
     cout<<"lt_y="<<lt_y<<endl;
     cout<<"rb_x="<<rb_x<<endl;
     cout<<"rb_y="<<rb_y<<endl;
     */

      }
  }

  if (bboxes.empty())
  {
    INFO_LOG("There is none object detected in image %s", inference_res->imgs[ind].image_info.path.c_str());
    continue;
  }

  cv::Mat mat = cv::imread(inference_res->imgs[ind].image_info.path, CV_LOAD_IMAGE_UNCHANGED);
  if (mat.empty())
   {
    ERROR_LOG("Fialed to deal file=%s. Reason: read image failed.", inference_res->imgs[ind].image_info.path.c_str());
    continue;
  }
  float scale_width = (float)mat.cols / inference_res->imgs[ind].image_info.image_data.width;
  cout<<"image_info.image_data.width="<<inference_res->imgs[ind].image_info.image_data.width<<endl;
  float scale_height = (float)mat.rows / inference_res->imgs[ind].image_info.image_data.height;
  cout<<"image_info.image_data.height="<< inference_res->imgs[ind].image_info.image_data.height<<endl;


  stringstream sstream;
  for (int i = 0; i < bboxes.size(); ++i)
   {
    cv::Point p1, p2;
    p1.x = scale_width * bboxes[i].lt_x;
    p1.y = scale_height * bboxes[i].lt_y;
    p2.x = scale_width * bboxes[i].rb_x;
    p2.y = scale_height * bboxes[i].rb_y;
    cv::rectangle(mat, p1, p2, kColors[i % kColors.size()], kLineSolid);

    //cout<<"p1.x="<<p1.x<<endl;
    //cout<<"p1.y="<<p1.y<<endl;
    //cout<<"p2.x="<<p2.x<<endl;
    //cout<<"p2.y="<<p2.y<<endl;

    sstream.str("");
    sstream << bboxes[i].attribute << " ";
    sstream.precision(kScorePrecision);
    sstream << 100 * bboxes[i].score << "%";
    string obj_str = sstream.str();
    cv::putText(mat, obj_str, cv::Point(p1.x, p1.y + kLabelOffset), cv::FONT_HERSHEY_COMPLEX, kFountScale, kFontColor);
  }

   int pos = inference_res->imgs[ind].image_info.path.find_last_of(kFileSperator);
   string file_name(inference_res->imgs[ind].image_info.path.substr(pos + 1));
   bool save_ret(true);
   sstream.str("");
   sstream << inference_res->console_params.output_path << kFileSperator << kOutputFilePrefix << file_name;
    string output_path = sstream.str();
    save_ret = cv::imwrite(output_path, mat);
    if (!save_ret)
    {
    ERROR_LOG("Failed to deal file=%s. Reason: save image failed.", inference_res->imgs[ind].image_info.path.c_str());
    continue;

    }


}

return HIAI_OK;

}


HIAI_IMPL_ENGINE_PROCESS("general_post", GeneralPost, INPUT_SIZE)
{
    HIAI_StatusT ret = HIAI_OK;

    // check arg0
    if (arg0 == nullptr) 
    {
      ERROR_LOG("Failed to deal file=nothing. Reason: arg0 is empty.");
      return HIAI_ERROR;
    }

    // just send to callback function when finished
    shared_ptr<EngineTransT> result = static_pointer_cast<EngineTransT>(arg0);

    cout<<"[LOG] result->status:"<<result->status<<endl;

   // inference failed
  if (!result->status)
  {
    if (SendSentinel()) 
    {
      return HIAI_OK;
    }
    cout<<"Failed to send finish data. Reason: SendData failed."<<endl;
    cout<<"Please stop this process manually."<<endl;
    ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
    ERROR_LOG("Please stop this process manually.");
    return HIAI_ERROR;
  }

  if (result->imgs.empty())

  {
  cout<<"Failed to process invalid message, original image is null."<<endl;
      HIAI_ENGINE_LOG( HIAI_ENGINE_RUN_ARGS_NOT_RIGHT, "Failed to process invalid message, original image is null.");
      return HIAI_ERROR;
  }

    // inference success, dealing inference results
    return HandleResults(result);
}
