
#ifndef GENERAL_IMAGE_GENERAL_IMAGE_H_
#define GENERAL_IMAGE_GENERAL_IMAGE_H_

#include "hiaiengine/engine.h"
#include "hiaiengine/data_type.h"
#include "data_type.h"

#define INPUT_SIZE 1
#define OUTPUT_SIZE 1

/**
 * @brief: inference engine class
 */
class GeneralImage : public hiai::Engine {
public:
  /**
   * @brief: engine initialize
   * @param [in]: engine's parameters which configured in graph.config
   * @param [in]: model description
   * @return: HIAI_StatusT
   */
  HIAI_StatusT Init(const hiai::AIConfig& config,
                    const std::vector<hiai::AIModelDescription>& model_desc);

  /**
   * @brief: engine processor which override HIAI engine
   *         get every image, and then send data to inference engine
   * @param [in]: input size
   * @param [in]: output size
   */
  HIAI_DEFINE_PROCESS(INPUT_SIZE, OUTPUT_SIZE);

private:

  /**
   * @brief: arrange image information
   * @param [out]: image_handle: image handler
   * @param [in]: image file path
   * @return: true: success; false: failed
   */
  bool ArrangeImageInfo(CusImageData &image_handle, const std::string &image_path);

  /**
   * @brief: send result
   * @param [in]: image_handle: engine transform image
   * @return: true: success; false: failed
   */
  //bool SendToEngine(const std::shared_ptr<EngineTrans> &image_handle);
  bool SendToEngine(const std::shared_ptr<CusBatchImagePara> & images_batch_handle);

  /**
   * @brief: get files from path and it's subpath
   * @param [in]: path can be liked as "path1,file,path2"
   * @param [out]: all existed files
   * @return: true: success; false: failed
   */
  void GetAllFiles(const std::string &path, std::vector<std::string> &file_vec);

  /**
   * @brief: path is directory or not
   * @param [in]: path
   * @return: true: directory; false: not directory
   */
  bool IsDirectory(const std::string &path);

  /**
   * @brief: path is exist or not
   * @param [in]: path
   * @return: true: exist; false: not exist
   */
  bool IsPathExist(const std::string &path);

  /**
   * @brief: split file path
   * @param [in]: path can be liked as "path1,file,path2"
   * @param [out]: split file paths
   */
  void SplitPath(const std::string &path, std::vector<std::string> &path_vec);

  /**
   * @brief: get files from one file path
   * @param [in]: path can be liked as "path1" or "path2"
   * @param [out]: files in this path
   */
  void GetPathFiles(const std::string &path,
                    std::vector<std::string> &file_vec);
};

#endif /* GENERAL_IMAGE_GENERAL_IMAGE_H_ */
