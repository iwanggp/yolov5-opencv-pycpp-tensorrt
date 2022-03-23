# yolov5-opencv-pycpp-tensorrt

本文目录如下：

- [ ] Python版Opencv部署Yolov5
- [ ] CPP版Opencv部署Yolov5
- [ ] TensorRT版部署Yolov5

### 1 Python版用Opencv部署Yolov5

这里默认使用了**CUDA**版本的OpenCV(有快速的没什么不用)，关于编译Opencv CUDA版本可以参考[编译CUDA版的OpenCV](https://www.jianshu.com/p/2ec17797a924)用DNN模块来加载ONNX模型。从此模型的部署就彻底地摆脱了深度学习框架的依赖。

python版本的主要主程序**python/python_deploy.py**

### 2 CPP版用Opencv部署Yolov5

同样这里也默认使用**CUDA**版本的Opencv来进行部署的，主函数位于**cpp/main.cpp**。同样不依赖深度学习框架。

### 3 TensorRT版本用Opencv部署Yolov5

前面已经有了Python和CPP版本的用Opencv部署Yolov5，但是这里的推理速度仍然有提升的空间。所以这里也增加了**TensorRT**版本的**yolov5**来进行部署，这里就不在介绍TensorRT相关知识，可以去官网学习相关知识。通过使用**TRT(TensorRT)**相比速度有接近**30%**的提升，提升效果还是比较明显的。在工程下**结果对比.xlsx**中我用100张图片进行测试，对TRT的提升效果进行统计。

#### 3.1 如何使用TensorRT部署Yolov5

通过前文可以看出TRT的加速效果还是比较明显的，所以如果项目对CT要求比较严格的话，首先考虑的就是通过TRT进行部署，经过我的验证转TRT的精度损失几乎可以忽略的。下面就着重的介绍如何用TRT部署ONNX模型。

#### 3.1.1 将ONNX模型转换为TRT模型

首先第一步要将ONNX模型进行转化，方法在**TensorRT_yolov5/utils.cpp**中的**nvinfer1::ICudaEngine *Utils::loadOnnxEngine(const string onnx_filename)**方法中。该方法部分代码如下：

```cpp
nvinfer1::ICudaEngine *Utils::loadOnnxEngine(const string onnx_filename)
{
    //1 load the onnx model
    //创建builder，builder接受的参数为gLogger类型
    nvinfer1::IBuilder* builder{nvinfer1::createInferBuilder(gLogger)};//创建builder
    //创建flag，固定写法
    const auto flag= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    //创建网络定义
    nvinfer1::INetworkDefinition* network=builder->createNetworkV2(flag);
    //创建onnx解释器，用于接受network和gLogger这两个参数
    nvonnxparser::IParser* parser=nvonnxparser::createParser(*network,gLogger);
    //验证是否支持一下FP16，以后可以转换为fp16格式的进行推理
    cout<<"support FP16--------"<<builder->platformHasFastFp16();//验证是否支持FP16
    //读取ONNX源文件，如果报错直接返回为空。否则读到parser解释器中
    if(!parser->parseFromFile(onnx_filename.c_str(),static_cast<int>(Logger::Severity::kWARNING))){
        cerr<<"ERROR:Could not parse onnx engine \n";
        return nullptr;
    }
    for(int i=0;i<parser->getNbErrors();++i){//打印onnx的信息
        std::cout<<parser->getError(i)->desc()<<std::endl;
    }
    cout<<"successfully load the onnx model"<<endl;
    //2 build the engine
    //因为TensorRT只是一个可以在GPU上独立运行的一个库，并不能够进行完整的训练流程
    //所以我们一般是通过神经网络框架训练后导出模型再通过TensorRT转化工具转化为TensorRT的格式再去运行
    unsigned int maxBatchSize=1;
    builder->setMaxBatchSize(maxBatchSize);//设置最大的BatchSize值
    nvinfer1::IBuilderConfig* config=builder->createBuilderConfig();//创建config
    config->setMaxWorkspaceSize(1<<20);//设置每层神经网络的最大空间值
    nvinfer1::ICudaEngine* engine=builder->buildEngineWithConfig(*network,*config);
    cout<<"successfully create the engine"<<endl;
    //3 serialize Model 序列化模型
    nvinfer1::IHostMemory *gieModelStream=engine->serialize();//模型序列化
    size_t lastindex=onnx_filename.find_last_of(".");//寻找最后一个点
    string trtfile=onnx_filename.substr(0,lastindex)+".trt";//命名trt文件
    ofstream engieFile(trtfile,ios::binary);//trtfile写入文件
    //写入文件格式为trt
    engieFile.write(static_cast<char*>(gieModelStream->data()),gieModelStream->size());
    engine->destroy();//及时释放资源
    builder->destroy();//及时释放资源
}
```

整个过程主要有如下的三个步骤：

1.  加载onnx模型
2.  创建TRT引擎
3.  序列化模型

上述的三个步骤在代码中都有详细的备注，具体可以参考代码。

#### 3.1.2 加载TRT模型

通过上步获得TRT模型后，接下来就是去加载TRT模型。这一部分主要就是线性操作，是比较固定的。最重要的一步就是对模型的反序列化，如下的关键代码：

```cpp
nvinfer1::ICudaEngine* trtengine=runtime->deserializeCudaEngine(modelData, modelSize, nullptr);//反序列化文件
```

#### 3.1.3 使用TRT进行推理

获得TRT模型后，下一步就是使用TRT模型进行推理，整个过程同样也是线性操作。这个过程也是比较统一的。首先第一步就是创建TRT执行的上下文。

```cpp
nvinfer1::IExecutionContext* context=trtEngine->createExecutionContext();//获取context运行上下文
```

接下来就是定义cuda数据流和定义缓存：

```cpp
//定义cuda数据流
cudaStream_t stream;
void* buffers[2];//定义buffer输入缓存
```

准备好这些必须的条件后，下一步就是将**TRT模型**，**运行上下文**，**CUDA数据流**和**缓存区**就可进行推理了，这部分代码主要如下：

```cpp
float* Utils::getTRTResult(string img_path, nvinfer1::ICudaEngine* trtEngine, cv::Mat &img,nvinfer1::IExecutionContext* context,cudaStream_t stream,void* buffers[2])
{
    img=cv::imread(img_path);
//    nvinfer1::ICudaEngine*  trtEngine=loadTRTEngine(engine_path,runtime);
//    nvinfer1::IExecutionContext* context=trtEngine->createExecutionContext();//获取context运行上下文
//    assert(context!=nullptr);
    cout << "layers= " << trtEngine->getNbLayers() << endl;
    // 指定输入和输出节点名来获取输入输出索引
    vector< nvinfer1::Dims> input_dims;
    vector< nvinfer1::Dims> output_dims;
    //创建cuda流，用于管理数据复制，存取和计算的并发操作
//    cudaStream_t stream;
//    CHECK(cudaStreamCreate(&stream));
    cout << "layers= " << trtEngine->getNbLayers() << endl;
    for(size_t i=0;i<trtEngine->getNbBindings();++i){
        if(trtEngine->bindingIsInput(i)){//判断如果是输入
            input_dims.emplace_back(trtEngine->getBindingDimensions(i));//输入
        }else{//输出大小
            output_dims.emplace_back(trtEngine->getBindingDimensions(i));
        }
    }
    if(input_dims.empty()||output_dims.empty()){//判断输入和输出是否为空
        cerr<<"Expect at least one input and one output for network\n";
        return nullptr;
    }
    std::cout << "input_dims[0] is: " << input_dims[0].d[0] << ", "
                                                            << input_dims[0].d[1] << ", "
                                                                                  << input_dims[0].d[2] << ", "
                                                                                                        << input_dims[0].d[3] << ", " << std::endl;

    cout<<"out_dim is "<<output_dims[0].d[0]<<", "
                                           <<output_dims[0].d[1]<<" ,"
                                                               <<output_dims[0].d[2]<<" ,"
                                                                                   <<output_dims[0].d[3]<<" ,"<<endl;
    static float data[batch_size * 3 * netHeight * netWidth];
    static float prob[batch_size * output_size*classes_num];
//    void* buffers[2];//定义buffer输入缓存
    //// // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[0], batch_size * 3 * netWidth * netHeight * sizeof(float)));//申请输入的缓存
    CHECK(cudaMalloc(&buffers[1], batch_size * output_size*classes_num * sizeof(float)));//申请输出的缓存
    //   clock_t start_time=clock();
//    cv::Mat pr_img=letterbox(img);//letterbox BGR to RGB
    cv::Mat pr_img=preprocess_img(img);//letterbox BGR to RGB
    mat2float2(data,pr_img);//将Mat转化为tensorRT的float类型
//    for(size_t i=0;i<1000;i++){
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, input_dims,output_dims,data, prob, batch_size);
    auto end = std::chrono::system_clock::now();
    std::cout <<"the doinference cost is......"<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return prob;
}
```

在这里也有很重要的一步就是执行推理的过程，该部分的主要代码是**doInference**方法，这是整个TRT引擎执行的核心代码：

```cpp
void Utils::doInference(nvinfer1::IExecutionContext &context, cudaStream_t &stream, void **buffers, vector<nvinfer1::Dims> input_dims, vector<nvinfer1::Dims> output_dims, float *input, float *output, int batchSize)
{
    int batch_size,channel,inputHeight,intputWidth;
    int output_size,out_classes;
    batch_size=input_dims[0].d[0];
    channel=input_dims[0].d[1];
    inputHeight=input_dims[0].d[2];
    intputWidth=input_dims[0].d[3];
    output_size=output_dims[0].d[0];
    out_classes=output_dims[0].d[1];
    cudaMemcpyAsync(buffers[0], input, batch_size * channel * inputHeight * intputWidth * sizeof(float), cudaMemcpyHostToDevice, stream);
    context.enqueue(1, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], batchSize * output_size*out_classes * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}
```

获得执行结果后，下一步就是进行yolo的结果后处理，这一步整体处理流程都是一样的。如下代码：

```cpp
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
```

自此，用TRT推理yolov5就算推理完成了。最后，在吹一下TRT加速真的很明显。
