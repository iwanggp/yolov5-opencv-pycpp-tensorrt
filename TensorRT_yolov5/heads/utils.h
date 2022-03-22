#ifndef UTILS_H
#define UTILS_H
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include<iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <opencv2\opencv.hpp>
#include "yolov5.h"
#include "direct.h"
using namespace std;
/**
  定义TensorRT的工具类
**/
static class Logger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char* msg) override{
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            cout << msg << endl;
        }
    }
} gLogger;
//static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
//    DIR *p_dir = opendir(p_dir_name);
//    if (p_dir == nullptr) {
//        return -1;
//    }

//    struct dirent* p_file = nullptr;
//    while ((p_file = readdir(p_dir)) != nullptr) {
//        if (strcmp(p_file->d_name, ".") != 0 &&
//            strcmp(p_file->d_name, "..") != 0) {
//            //std::string cur_file_name(p_dir_name);
//            //cur_file_name += "/";
//            //cur_file_name += p_file->d_name;
//            std::string cur_file_name(p_file->d_name);
//            file_names.push_back(cur_file_name);
//        }
//    }

//    closedir(p_dir);
//    return 0;
//}

#define CHECK(status) do {            \
    cudaError_t err = (status);       \
    if (err != cudaSuccess) {         \
    fprintf(stderr, "API error"   \
    "%s:%d Returned:%d\n",    \
    __FILE__, __LINE__, err); \
    exit(1);                      \
    }                                 \
    } while(0)
class Utils
{
private:
    Yolov5 yolov5;
    static const int netHeight=640;//网络的输入高度
    static const int netWidth=640;//网络的输出高度

    static const int classes_num=85;//网络的classes总数
    static const int output_size=25200;//网络的输出宽度
    static const int batch_size=1;//batch_size大小

public:
    Utils();
    ~Utils(){};
    nvinfer1::ICudaEngine* loadOnnxEngine(const string onnx_filename);//加载onnx模型并转换为trt模型
    nvinfer1::ICudaEngine* loadTRTEngine(const string enginePath,nvinfer1::IRuntime* runtime);//加载trt模型
    nvinfer1::IExecutionContext* context;// 执行上下文
    vector<string> getClassNames(const string & imagenet_classes);
    cv::Mat preprocess_img(cv::Mat& img);
    cv::Mat letterbox(cv::Mat& img);//实现yolov5的letterbox图片预处理
    void doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void **buffers,vector< nvinfer1::Dims> input_dims,vector< nvinfer1::Dims > output_dims,float* input, float* output, int batchSize);
    float* getTRTResult(string img_path, nvinfer1::ICudaEngine*  trtEngine, cv::Mat &img,nvinfer1::IExecutionContext* context,cudaStream_t stream,void* buffers[2]);//获得trt执行结果
    void mat2float(float *data,cv::Mat img);
    void mat2float2(float *data,cv::Mat img);
    cv::Rect get_rect(cv::Mat& img,float bbox[4]);
};

#endif // UTILS_H
