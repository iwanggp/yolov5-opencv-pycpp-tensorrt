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
    //设置yolov5的网络参数
    //计算归一化参数
    float Sigmoid(float x){
        return static_cast<float>(1.f/(1.f+exp(-x)));
    }
    cv::dnn::Net yolov5Net;//yoloNet
    vector<cv::Scalar> color;//颜色列表
    //anchors
    const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
    //stride
    const float netStride[3] = { 8.0, 16.0, 32.0 };
    const int netWidth = 640; //网络模型输入大小
    const int netHeight = 640;
    float nmsThreshold = 0.45;
    float boxThreshold = 0.25;
    float classThreshold = 0.15;
    //类名
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

public:
    //读取模型函数
    bool readModel(cv::dnn::Net &net,std::string &netPath,bool isCuda);
    //模型的输出
    struct Output{
        int id;//结果类别id
        float confidence;//结果置信度
        cv::Rect box;//矩形框
    };

    //检测函数
    bool detect(cv::Mat &Srcimg,cv::dnn::Net &net,std::vector<Output> &output);
    bool detect2(cv::Mat &Srcimg,cv::dnn::Net &net,std::vector<Output> &output);
    //对结果画框并输出
    void drawPred(cv::Mat &img,std::vector<Output> result,std::vector<cv::Scalar> color);
    //测试模型单张图片
    cv::Mat test_one(string img_path,cv::dnn::Net yolov5Net);
    //测试一个文件夹所有的图片
    void multi_test(string img_dir,cv::dnn::Net yolov5Net);
    void sigmoid2(cv::Mat* out, int length);
};

#endif // YOLOV5_H
