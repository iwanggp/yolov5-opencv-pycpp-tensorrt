#ifndef YOLOV5_H
#define YOLOV5_H
#include<iostream>
#include <opencv2\opencv.hpp>
#include<math.h>
using namespace std;
class Yolov5
{
public:
    Yolov5();
    ~Yolov5(){};
    //设置Sigmoid函数
    float Sigmoid(float x){
        return static_cast<float>(1.f/(1.f+exp(-x)));
    }
    //anchors
    const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
    //stride
    const float netStride[3] = { 8.0, 16.0, 32.0 };
    const int netWidth = 640; //网络模型输入大小
    const int netHeight = 640;
    float nmsThreshold = 0.45;
    float boxThreshold = 0.25;
    float classThreshold = 0.15;
    float* pdata;//存储模型推理的结果
    vector<cv::Scalar> color;//颜色列表
//    std::vector<std::string> className = { "dent", "scratch"};

    std::vector<std::string> className = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                           "hair drier", "toothbrush" };

    const int net_width=className.size()+5;
    //模型的输出的结构
    struct Output{
        int id;//结果类别id
        float confidence;//结果置信度
        cv::Rect box;//矩形框
    };
    void postresult_trt(float* pdata,cv::Mat img);//处理TensorRT的结果
    cv::Rect get_rect(cv::Mat& img,float bbox[4]);
    //对结果画框并输出
    void drawPred(cv::Mat &img,std::vector<Output> result,std::vector<cv::Scalar> color);

};

#endif // YOLOV5_H
