#include "yolov5.h"
#include <fstream>
#include <thread>
#include <math.h>
#include <QDebug>
#include <string>
#include<iostream>
#include<QtCore>
#include <iomanip>
#include <chrono>
using namespace std;
Yolov5::Yolov5()
{
    //初始化color
    for (int i = 0; i < className.size(); i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }
}
/**
 * @brief Yolov5::postresult_trt 后处理trt的结果
 * @param pdata trt的结果
 * @param img 原生的img
 */
void Yolov5::postresult_trt(float *pdata, cv::Mat img)
{
    //求缩放比
    float ratio_h=(float)img.rows/netHeight;
    float ratio_w=(float)img.cols/netWidth;
    vector<int> classIds;//结果id数组
    vector<float> confidences;//结果每个id对应置信度数组
    vector<cv::Rect> boxes;//每个id矩形框
    clock_t start_time=clock();
    for(int stride=0;stride<3;stride++){//stride遍历
        int grid_x=(int)(netWidth/netStride[stride]);
        int grid_y=(int)(netHeight/netStride[stride]);
        int area=grid_x*grid_y;
        //anchor遍历
        for(int anchor=0;anchor<3;anchor++){
            const float anchor_w=netAnchors[stride][anchor*2];//获得anchor的宽度
            const float anchor_h=netAnchors[stride][anchor*2+1];//获得anchor的高度
            for(int i=0;i<grid_y;i++){
                for(int j=0;j<grid_x;j++){
                    float _box_score=pdata[4];
                    float box_score=Sigmoid(_box_score);//一行是否有物体的概率
                    if(box_score>boxThreshold){
                        //为了使用minMaxLoc()，将85长度数组变成Mat对象
                        cv::Mat scores(1,className.size(),CV_32FC1,pdata+5);
                        cv::Point classIdPoint;
                        double max_class_score;
                        cv::minMaxLoc(scores,0,&max_class_score,0,&classIdPoint);
                        max_class_score=Sigmoid((float)max_class_score);
                        float conf=box_score*max_class_score;
                        if(conf>classThreshold){
                            //rect [x,y,w,h]获得检测框
                            float x=(Sigmoid(pdata[0])*2.f-0.5f+j)*netStride[stride];//中心点x坐标
                            float y=(Sigmoid(pdata[1])*2.f-0.5f+i)*netStride[stride];//中心点y坐标
                            float w=powf(Sigmoid(pdata[2])*2.f,2.f)*anchor_w;//w
                            float h=powf(Sigmoid(pdata[3])*2.f,2.f)*anchor_h;//h
                            float _boxs[4]={x,y,w,h};
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back(conf);
                            cv::Rect real_rect=get_rect(img,_boxs);//获取真实的坐标值
                            boxes.push_back(real_rect);
                        }
                    }
                    pdata+=net_width;//指针下移一行
                }
            }
        }
    }//处理结束
    clock_t end_time2=clock();
    cout<<"the detect img  time is...."<<static_cast<double>(end_time2-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes,confidences,classThreshold,nmsThreshold,nms_result);
    vector<Output> output;
    for(int i=0;i<nms_result.size();i++){
        int idx=nms_result[i];
        Output result;
        result.id=classIds[idx];
        result.confidence=confidences[idx];
        result.box=boxes[idx];
        output.push_back(result);
    }
    clock_t end_time3=clock();
    double total_time=static_cast<double>(end_time3-start_time)/CLOCKS_PER_SEC*1000;
    cout<<"the total time is........."<<total_time;
    drawPred(img,output,color);
}

cv::Rect Yolov5::get_rect(cv::Mat &img, float bbox[4])
{
    int l, r, t, b;
    float r_w = netWidth / (img.cols * 1.0);
    float r_h = netHeight / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (netHeight - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (netHeight - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (netWidth - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (netWidth - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}
/**
 * @brief Yolov5::drawPred 画图函数
 * @param img 待检测图片
 * @param result 检测结果
 * @param color 颜色
 */
void Yolov5::drawPred(cv::Mat &img, std::vector<Yolov5::Output> result, std::vector<cv::Scalar> color)
{
    for(int i=0;i<result.size();i++){
        int left,top;
        left=result[i].box.x;
        top=result[i].box.y;
        int color_num=i;
        cv::rectangle(img,result[i].box,color[result[i].id],1,8);
        QString str=QString::number(result[i].confidence,'f',4);//保留几位小数
        string confidence=str.toStdString();
        string label=className[result[i].id]+":"+confidence;//标签
        int baseLine;
        cv::Size labelSize=cv::getTextSize(label,cv::FONT_HERSHEY_SIMPLEX,0.3,1,&baseLine);
        top=max(top,labelSize.height);//防止超出去
        cv::putText(img,label,cv::Point(left,top),cv::FONT_HERSHEY_SIMPLEX,1,color[result[i].id],2);
    }
    cv::imwrite("D:/Projects/trt.png",img);
//    cv::imshow("res",img);
//    cv::waitKey(0);
}
