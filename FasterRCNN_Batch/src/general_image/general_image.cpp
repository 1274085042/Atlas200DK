
#include "general_image.h"

#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <string.h>
//#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "hiaiengine/log.h"
//#include "hiaiengine/data_type.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "tool_api.h"

using hiai::Engine;
//using hiai::BatchImagePara;
using namespace std;

namespace {
// output port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// get stat success
const int kStatSuccess = 0;
// image file path split character
const string kImagePathSeparator = ",";
// path separator
const string kPathSeparator = "/";

}

HIAI_StatusT GeneralImage::Init( const hiai::AIConfig& config, const vector<hiai::AIModelDescription>& model_desc)
{
  // do noting
  return HIAI_OK;
}

void GeneralImage::GetAllFiles(const string &path, vector<string> &file_vec) 
{
  // split file path
  vector<string> path_vector;
  SplitPath(path, path_vector);

  for (string every_path : path_vector)
   {
    // check path exist or not
    if (!IsPathExist(path))
    {
      ERROR_LOG("Failed to deal path=%s. Reason: not exist or can not access.",
                every_path.c_str());
      continue;
    }

    // get files in path and sub-path
    GetPathFiles(every_path, file_vec);
  }

}

bool GeneralImage::IsDirectory(const string &path) 
{
  // get path stat
  struct stat buf;
  if (stat(path.c_str(), &buf) != kStatSuccess) {
    return false;
  }

  // check
  if (S_ISDIR(buf.st_mode)) {
    return true;
  } else {
    return false;
  }
}

bool GeneralImage::IsPathExist(const string &path) 
{
  //判断路径是否存在
  //file 文件句柄
  ifstream file(path);
  if (!file) {
    return false;
  }
  return true;
}

void GeneralImage::SplitPath(const string &path, vector<string> &path_vec) 
{
  char *char_path = const_cast<char*>(path.c_str());     //string.c_str()返回一个指针(char *)
  const char *char_split = kImagePathSeparator.c_str();
  //strtok()分割字符串
  char *tmp_path = strtok(char_path, char_split);
  while (tmp_path)
  {
    path_vec.emplace_back(tmp_path);
    tmp_path = strtok(nullptr, char_split);

  }
}

void GeneralImage::GetPathFiles(const string &path, vector<string> &file_vec)
{
  struct dirent *dirent_ptr = nullptr;
  DIR *dir = nullptr;
  if (IsDirectory(path))
  {
    dir = opendir(path.c_str());
    while ((dirent_ptr = readdir(dir)) != nullptr)
    {
      // skip . and ..
      if (dirent_ptr->d_name[0] == '.')
      {
        continue;
      }

      // file path
      string full_path = path + kPathSeparator + dirent_ptr->d_name;
      // directory need recursion
      if (IsDirectory(full_path))
      {
        GetPathFiles(full_path, file_vec);
      } else 
      {
        // put file
        file_vec.emplace_back(full_path);
      }
    }
  } 
  else
  {
    file_vec.emplace_back(path);
  }
}

bool GeneralImage::ArrangeImageInfo(CusImageData &image_handle, const string &image_path)
{
  // read image using OPENCV
  cv::Mat mat = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  if (mat.empty()) 
  {
    ERROR_LOG("Failed to deal file=%s. Reason: read image failed.", image_path.c_str());
    return false;
  }

  // set property
  image_handle.image_info.path = image_path;
  image_handle.image_info.image_data.width = mat.cols;
  image_handle.image_info.image_data.height = mat.rows;

  // set image data
  uint32_t size = mat.total() * mat.channels();
  u_int8_t *image_buf_ptr = new (nothrow) u_int8_t[size];
  if (image_buf_ptr == nullptr)
  {
    HIAI_ENGINE_LOG("new image buffer failed, size=%d!", size);
    ERROR_LOG("Failed to deal file=%s. Reason: new image buffer failed.", image_path.c_str());
    return false;
  }

  error_t mem_ret = memcpy_s(image_buf_ptr, size, mat.ptr<u_int8_t>(), mat.total() * mat.channels());

  if (mem_ret != EOK)
  {
    delete[] image_buf_ptr;
    ERROR_LOG("Failed to deal file=%s. Reason: memcpy_s failed.", image_path.c_str());
    image_buf_ptr = nullptr;
    return false;
  }
  image_handle.image_info.image_data.size = size;
  image_handle.image_info.image_data.data.reset(image_buf_ptr, default_delete<u_int8_t[]>());

  return true;
}

bool GeneralImage::SendToEngine(const shared_ptr<CusBatchImagePara> & images_batch_handle)
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

HIAI_IMPL_ENGINE_PROCESS("general_image", GeneralImage, INPUT_SIZE)
{
    HIAI_StatusT ret = HIAI_OK;

    // Step1: check arg0
    if (arg0 == nullptr)
    {
        ERROR_LOG("Failed to deal file=nothing. Reason: arg0 is empty.");
        return HIAI_ERROR;
    }

    // Step2: get all files
    shared_ptr<ConsoleParams> console_param = static_pointer_cast<ConsoleParams>(arg0);
    string input_path = string(console_param->input_path);
    vector<string> file_vec;
    GetAllFiles(input_path, file_vec);
    if (file_vec.empty())
    {
        ERROR_LOG("Failed to deal all empty path=%s.", input_path.c_str());
        return HIAI_ERROR;
    }

    //images_batch_handle存放一个batch图片
    shared_ptr<CusBatchImagePara> images_batch_handle = nullptr;
    MAKE_SHARED_NO_THROW(images_batch_handle, CusBatchImagePara);

    // handle one batch every time
    images_batch_handle->b_info.batch_size = 4;
    images_batch_handle->b_info.max_batch_size = 4;


    if (images_batch_handle == nullptr)
    {
        ERROR_LOG("Failed to deal file. Reason: new CusBatchImagePara failed.");
    }

    for (string path : file_vec)
    {
        CusImageData img_data;
        // arrange image information, if failed, skip this image
        if (!ArrangeImageInfo(img_data, path))
        {
            continue;
        }

        // send data to inference engine
        images_batch_handle->console_params.input_path = console_param->input_path;
        images_batch_handle->console_params.model_height = console_param->model_height;
        images_batch_handle->console_params.model_width = console_param->model_width;
        images_batch_handle->console_params.output_nums = console_param->output_nums;
        images_batch_handle->console_params.output_path = console_param->output_path;

        images_batch_handle->v_img.push_back(img_data);
    }

    if (!SendToEngine(images_batch_handle))
    {
        ERROR_LOG("Send data failed.");
        return HIAI_ERROR;
    }

    // sleep
    usleep(kSleepInterval);

   // Step4: send finished batch data
    shared_ptr <CusBatchImagePara> images_batch_handle_fin = nullptr;
    MAKE_SHARED_NO_THROW(images_batch_handle_fin, CusBatchImagePara);  //images_batch_handle = MakeSharedNoThrow<CusBatchImagePara>();
    if (images_batch_handle_fin == nullptr)
    {
        ERROR_LOG("Failed to send finish data. Reason: new CusBatchImagePara failed.");
        ERROR_LOG("Please stop this process manually.");
        return HIAI_ERROR;
    }

    images_batch_handle_fin->is_finished = true;

    if (SendToEngine(images_batch_handle_fin))
    {
        return HIAI_OK;
    }
    ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
    ERROR_LOG("Please stop this process manually.");
    return HIAI_ERROR;
}
