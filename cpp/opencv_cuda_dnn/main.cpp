#include "mainwindow.h"
#include "yolov5.h"
#include <QApplication>
#include<iostream>
#include<QDebug>
#include<time.h>
#include<opencv2/opencv.hpp>
using namespace std;
int main(int argc, char *argv[])
{
    string img_path="D:/Projects/wine.jpg";
    string onnx="D:/Projects/yolov5l.onnx";
//    string img_dir="E:/dc/second_hole/";
//    cv::Mat img=cv::imread(img_path);
    Yolov5 yolov5;
    cv::dnn::Net yolov5Net;
    //加载模型
    if(yolov5.readModel(yolov5Net,onnx,true)){
        qDebug()<<"read onnxnet ok........";
    }else{
        return -1;
    }
    for(size_t i=0;i<100;i++){
        cv::Mat result=yolov5.test_one(img_path,yolov5Net);
    }
//    cv::Mat result=yolov5.test_one(img_path,yolov5Net);
//    cv::imwrite("D:/Projects/result_wine.png",result);
//    yolov5.multi_test(img_dir,yolov5Net);
//    cv::VideoCapture capture(0);
//    while (true) {
//        cv::Mat frame;
//        capture>>frame;
//        cv::Mat blob;
//        cv::dnn::blobFromImage(frame,blob,1/255.0,cv::Size(640,640),cv::Scalar(0,0,0),true,false);
//        if (yolov5.detect(frame, yolov5Net, result)) {
//            yolov5.drawPred(frame, result, color);
//        }
//        cv::imshow("reading.....",frame);
//        cv::waitKey(1);
//    }

//    system("pause");
    return 0;

}
