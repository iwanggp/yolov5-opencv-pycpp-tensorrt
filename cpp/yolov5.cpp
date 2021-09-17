#include "yolov5.h"
#include <fstream>
#include <thread>
#include <math.h>
#include <QDebug>
#include <string>
#include<iostream>
#include<QtCore>
#include <iomanip>
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
 * @brief Yolov5::readModel 加载模型函数
 * @param net dnn Net
 * @param netPath 模型路径
 * @param isCuda 是否使用cuda加速
 * @return
 */
bool Yolov5::readModel(cv::dnn::Net &net, std::string &netPath, bool isCuda)
{
    try{
        net=cv::dnn::readNetFromONNX(netPath);
    }catch(const std::exception&){
        return false;
    }
    if (isCuda){
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return true;
}
/**
 * @brief Yolov5::detect yolo的检测函数
 * @param Srcimg 读取的源文件
 * @param net yolo网络
 * @param output 输出
 * @return
 */
bool Yolov5::detect(cv::Mat &Srcimg, cv::dnn::Net &net, vector<Yolov5::Output> &output)
{
    clock_t fps1=clock();

    clock_t start_time=clock();
    clock_t end_time_blob=clock();
    //下面是固定写法
    cv::Mat blob;
    cv::dnn::blobFromImage(Srcimg,blob,1/255.0,cv::Size(netWidth,netHeight),cv::Scalar(0,0,0),true,false);
    net.setInput(blob);
    vector<cv::Mat> netOutputImg;
    clock_t end_time100=clock();
    net.forward(netOutputImg,net.getUnconnectedOutLayersNames());//获取网络的输出结果
    clock_t end_time1=clock();
//    cout<< "2:opencv dnn net time: "<<static_cast<double>(end_time1-end_time100)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时间
    cout<< "the cuda opencv cost is..........: "<<static_cast<double>(end_time1-end_time100)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时间
    //下一步就是遍历网络的输出一个二维数组[25200*85],需要遍历这个一维数组
    //这里主要是用vector来实现数据的封装
    vector<int> classIds;//结果id数组
    vector<float> confidences;//结果每个id对应置信度数组
    vector<cv::Rect> boxes;//每个id矩形框
    //求缩放比
    float ratio_h=(float)Srcimg.rows/netHeight;
    float ratio_w=(float)Srcimg.cols/netWidth;
    int net_width=className.size()+5;//输出网络宽度为类别+5
    float* pdata=(float*)netOutputImg[0].data;//获取一个输出数据
    cout<<"get the ration_h is............"<<ratio_h<<"the ratio_w is...."<<ratio_w;
//    cout<<pdata[4];
    //先从stride遍历
    for(int stride=0;stride<3;stride++){//stride遍历
        int grid_x=(int)(netWidth/netStride[stride]);
        int grid_y=(int)(netHeight/netStride[stride]);
        int area=grid_x*grid_y;
        //anchor遍历
        for(int anchor=0;anchor<3;anchor++){
            const float anchor_w=netAnchors[stride][anchor*2];//获得anchor的宽度
            const float anchor_h=netAnchors[stride][anchor*2+1];//获得anchor的宽度
            for(int i=0;i<grid_y;i++){
                for(int j=0;j<grid_y;j++){
                    float _box_score=pdata[4];
                    float box_score=Sigmoid(_box_score);//一行是否有物体的概率
                    if(box_score>boxThreshold){
                        //为了使用minMaxLoc()，将85长度数组变成Mat对象
                        cv::Mat scores(1,className.size(),CV_32FC1,pdata+5);
                        cv::Point classIdPoint;
                        double max_class_score;
                        cv::minMaxLoc(scores,0,&max_class_score,0,&classIdPoint);
//                        max_class_score=(max_class_score*box_score);
                        max_class_score=Sigmoid((float)max_class_score);
//                        max_class_score=max_class_score*box_score;
//                        qDebug()<<"ssssss   the max_class_score is......"<<max_class_score;
                        float conf=box_score*max_class_score;
                        if(conf>classThreshold){
                            //rect [x,y,w,h]获得检测框
                            float x=(Sigmoid(pdata[0])*2.f-0.5f+j)*netStride[stride];//中心点x坐标
                            float y=(Sigmoid(pdata[1])*2.f-0.5f+i)*netStride[stride];//中心点y坐标
                            float w=powf(Sigmoid(pdata[2])*2.f,2.f)*anchor_w;//w
                            float h=powf(Sigmoid(pdata[3])*2.f,2.f)*anchor_h;//w
                            //获取左上坐标
                            int left=(x-0.5*w)*ratio_w;
                            int top=(y-0.5*h)*ratio_h;
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back(conf);
                            boxes.push_back(cv::Rect(left,top,int(w*ratio_w),int(h*ratio_h)));
                        }
                    }
                    pdata+=net_width;//指针下移一行
                }
            }
        }
    }
    clock_t end_time2=clock();
    cout<<"the detect img  time is...."<<static_cast<double>(end_time2-end_time1)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes,confidences,classThreshold,nmsThreshold,nms_result);
    for(int i=0;i<nms_result.size();i++){
        int idx=nms_result[i];
        Output result;
        result.id=classIds[idx];
        result.confidence=confidences[idx];
        result.box=boxes[idx];
        output.push_back(result);
    }
    clock_t fps2=clock();
    int time_ms=static_cast<double>(fps2-fps1)/CLOCKS_PER_SEC*1000;
    int fps=1000/time_ms;
    string fps_str="FPS:"+std::to_string(fps);
//    cv::putText(Srcimg, fps_str, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);//测量FPS使用
    clock_t end_time3=clock();
//    cout<< "4:NMS time: "<<static_cast<double>(end_time3-end_time2)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出NMS运行时间

    double total_time=static_cast<double>(end_time3-start_time)/CLOCKS_PER_SEC*1000;
    qDebug()<<"the total time is........."<<total_time;
    QString str=QString::number(total_time,'f',2);
    string cost=str.toStdString()+" ms";
    cv::putText(Srcimg, "Cost:"+cost, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.75, cv::Scalar(0, 0, 255), 2);
    cout<< "total time: "<<static_cast<double>(end_time3-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时间
    qDebug()<<"the......sucesss................"<<output.size();
    if(output.size()) return true;
    else return false;

}
void Yolov5::sigmoid2(cv::Mat* out, int length)
{
    float* pdata = (float*)(out->data);
    int i = 0;
    for (i = 0; i < length; i++)
    {
        pdata[i] = 1.0 / (1 + expf(-pdata[i]));
    }
}
bool Yolov5::detect2(cv::Mat &Srcimg, cv::dnn::Net &net, std::vector<Yolov5::Output> &output)
{
    clock_t fps1=clock();

    clock_t start_time=clock();
    clock_t end_time_blob=clock();
    //下面是固定写法
    cv::Mat blob;
    cv::dnn::blobFromImage(Srcimg,blob,1/255.0,cv::Size(netWidth,netHeight),cv::Scalar(0,0,0),true,false);
    net.setInput(blob);
    vector<cv::Mat> outs;
    net.forward(outs,net.getUnconnectedOutLayersNames());//获取网络的输出结果
    clock_t end_time1=clock();
    cout<< "2:opencv dnn net time: "<<static_cast<double>(end_time1-end_time_blob)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时间
    //下一步就是遍历网络的输出一个二维数组[25200*85],需要遍历这个一维数组
    //这里主要是用vector来实现数据的封装
    vector<int> classIds;//结果id数组
    vector<float> confidences;//结果每个id对应置信度数组
    vector<cv::Rect> boxes;//每个id矩形框
    //求缩放比
    float ratio_h=(float)Srcimg.rows/netHeight;
    float ratio_w=(float)Srcimg.cols/netWidth;
    int n = 0, q = 0, i = 0, j = 0, nout = 85, c = 0;
    int net_width=className.size()+5;//输出网络宽度为类别+5
//    float* pdata=(float*)netOutputImg[0].data;//获取一个输出数据
    //先从stride遍历
    for(int stride=0;stride<3;stride++){//stride遍历
        int grid_x=(int)(netWidth/netStride[stride]);
        int grid_y=(int)(netHeight/netStride[stride]);
        int area=grid_x*grid_y;
        sigmoid2(&outs[n],3*nout*area);
        //anchor遍历
        for(int anchor=0;anchor<3;anchor++){
            const float anchor_w=netAnchors[stride][anchor*2];//获得anchor的宽度
            const float anchor_h=netAnchors[stride][anchor*2+1];//获得anchor的宽度
            float* pdata=(float*)outs[n].data+anchor*nout*area;
            for(int i=0;i<grid_y;i++){
                for(int j=0;j<grid_y;j++){
//                    float box_score=Sigmoid(pdata[4]);//一行是否有物体的概率
                    float box_score=pdata[4*area+i*grid_x+j];
                    if(box_score>boxThreshold){
                        //为了使用minMaxLoc()，将85长度数组变成Mat对象
//                        cv::Mat scores(1,className.size(),CV_32FC1,pdata+5);
//                        cv::Point classIdPoint;
                        float max_class_score=0.0;
                        float class_socre = 0.f;
                        int max_class_id = 0;
                        for(int c=0;c<80;c++){
                            class_socre=pdata[(c+5)*area+i*grid_x+j];
//                            class_socre=Sigmoid(class_socre);
                            if(class_socre>max_class_score){
                                max_class_score=class_socre;
                                max_class_id=c;
                            }
                        }
                        if(max_class_score>classThreshold){
                            cout<<"the score is........."<<(class_socre);
                            float cx = (pdata[i * grid_x + j] * 2.f - 0.5f + j) * netStride[stride];  ///cx
                            float cy = (pdata[area + i * grid_x + j] * 2.f - 0.5f + i) * netStride[stride];   ///cy
                            float w = powf(pdata[2 * area + i * grid_x + j] * 2.f, 2.f) * anchor_w;   ///w
                            float h = powf(pdata[3 * area + i * grid_x + j] * 2.f, 2.f) * anchor_h;  ///h

                            int left = (cx - 0.5*w)*ratio_w;
                            int top = (cy - 0.5*h)*ratio_h;   ///坐标还原到原图上

                            classIds.push_back(max_class_id);
                            confidences.push_back(max_class_score);
                            boxes.push_back(cv::Rect(left, top, (int)(w*ratio_w), (int)(h*ratio_h)));
                            }
                        }
//                        cv::minMaxLoc(scores,0,&max_class_score,0,&classIdPoint);
//                        qDebug()<<"ssssss33333   the max_class_score is......"<<max_class_score;
//                        max_class_score=Sigmoid((float)max_class_score);

//                    pdata+=net_width;//指针下移一行
                }
            }
        }
    }
    clock_t end_time2=clock();
    cout<<"the detect img  time is...."<<static_cast<double>(end_time2-end_time1)/CLOCKS_PER_SEC*1000<<"ms"<<endl;
    cout<<"the......sucesss.."<<boxes.size();
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes,confidences,classThreshold,nmsThreshold,nms_result);
    for(int i=0;i<nms_result.size();i++){
        int idx=nms_result[i];
        Output result;
        result.id=classIds[idx];
        result.confidence=confidences[idx];
        result.box=boxes[idx];
        output.push_back(result);
    }
    clock_t fps2=clock();
    int time_ms=static_cast<double>(fps2-fps1)/CLOCKS_PER_SEC*1000;
    int fps=1000/time_ms;
    string fps_str="FPS:"+std::to_string(fps);
//    cv::putText(Srcimg, fps_str, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);//测量FPS使用
    clock_t end_time3=clock();
//    cout<< "4:NMS time: "<<static_cast<double>(end_time3-end_time2)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出NMS运行时间

    double total_time=static_cast<double>(end_time3-start_time)/CLOCKS_PER_SEC*1000;
    qDebug()<<"the total time is........."<<total_time;
    QString str=QString::number(total_time,'f',2);
    string cost=str.toStdString()+" ms";
    cv::putText(Srcimg, "Cost:"+cost, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.75, cv::Scalar(0, 0, 255), 2);
    cout<< "total time: "<<static_cast<double>(end_time3-start_time)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时间
    qDebug()<<"the......sucesss................"<<output.size();
    if(output.size()) return true;
    else return false;

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
//    cv::imshow("res",img);
//    cv::waitKey(0);
}
/**
 * 单张图片测试
 * @brief Yolov5::test_one
 * @param img_path 图片路径
 */
cv::Mat Yolov5::test_one(string img_path,cv::dnn::Net yolov5Net)
{
    vector<Output> result;
    cv::Mat img=cv::imread(img_path);
    if(detect(img,yolov5Net,result)){
        drawPred(img,result,color);
        return img;
    }else{
        return cv::Mat(0,3,CV_32FC1);
    }
}
/**
 * 多张图片测试
 * @brief Yolov5::multi_test
 * @param img_dir 文件夹
 * @param yolov5Net cv::dnn::Net yolov5Net
 */
void Yolov5::multi_test(string img_dir, cv::dnn::Net yolov5Net)
{
    QString str=QString::fromStdString(img_dir);//string2 QString
    QDir dir(str);
    if(!dir.exists()) {
        qDebug()<<"the file is empty";
        return;
    };
    dir.setFilter(QDir::Files);
    QFileInfoList list=dir.entryInfoList();
    int file_count=list.count();
    if(file_count<=0) return;
    QStringList string_list;
    for(int i=0;i<file_count;i++){
        QFileInfo file_info=list.at(i);
        QString file_path=file_info.absoluteFilePath();
        QString file_name=file_info.baseName();
        string _file_name=file_name.toStdString();
        string result_name=img_dir+"/"+_file_name+"_reslut.png";
        cv::Mat result=test_one(file_path.toStdString(),yolov5Net);
        if(!result.data) continue;
        if(!result.empty()) cv::imwrite(result_name,result);
    }

}
