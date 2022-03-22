#include "mainwindow.h"
#include <QApplication>
#include "utils.h"
#include <assert.h>
#include <memory>
#include <opencv2\opencv.hpp>
#include "cuda_runtime_api.h"
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <numeric>
#include<chrono>
#include "yolov5.h"
#define CHECK(status) do {            \
    cudaError_t err = (status);       \
    if (err != cudaSuccess) {         \
    fprintf(stderr, "API error"   \
    "%s:%d Returned:%d\n",    \
    __FILE__, __LINE__, err); \
    exit(1);                      \
    }                                 \
    } while(0)
//计算tensor的size
size_t getSizeByDim(const nvinfer1::Dims& dims){
    size_t size=1;
    for(size_t i=0;i<dims.nbDims;++i){
        size*=dims.d[i];
    }
    return size;
}
struct Output{
    int id;//结果类别id
    float confidence;//结果置信度
    cv::Rect box;//矩形框
};

int main(int argc, char *argv[])
{

    Utils utils;
    Yolov5 yolov5;
    const char* onnx="D:/Projects/best_np4.onnx";
//    utils.loadOnnxEngine(onnx);//转tensorrt
    const string engine="D:/Projects/yolov5l.trt";
    const string path="D:/Projects/wine.jpg";
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);//创建runtime时
    nvinfer1::ICudaEngine*  trtEngine=utils.loadTRTEngine(engine,runtime);
    nvinfer1::IExecutionContext* context=trtEngine->createExecutionContext();//获取context运行上下文
    //定义cuda数据流
    cudaStream_t stream;
    void* buffers[2];//定义buffer输入缓存
    assert(context!=nullptr);
    for(size_t i=0;i<100;i++){
        cv::Mat img;
        cout<<"the number is...."<<i;
    //    img=utils.letterbox(cv::imread(path));
    //    cv::imwrite("D:/Projects/wine.png",img);
        auto start = std::chrono::system_clock::now();
        float* pdata;
        pdata=utils.getTRTResult(path,trtEngine,img,context,stream,buffers);//通过trt引擎获得trt的执行结果
        yolov5.postresult_trt(pdata,img);
        auto end = std::chrono::system_clock::now();
        std::cout <<"the total time is........."<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    //一定要记得释放cuda显存否则会撑爆显存
    cudaStreamDestroy(stream);//释放stream
    cudaFree(buffers[0]);//释放输入buffer
    cudaFree(buffers[1]);//释放输出buffer
    context->destroy();//释放上下文
    trtEngine->destroy();//释放trt引擎
    runtime->destroy();//运行时销毁
    return 0;
}
