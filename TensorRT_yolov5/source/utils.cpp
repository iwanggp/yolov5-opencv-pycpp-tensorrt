#include "utils.h"
#include<chrono>
using namespace std;
Utils::Utils()
{

}
/**
 * @brief Utils::loadOnnxEngine 加载onnx文件并转换为cuda engine
 * @param onnx_filename onnx模型路径
 * @return
 */

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
    //读取ONNX源文件，如果报错直接返回为空。否者读到parser解释器中
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
/**
 * @brief Utils::loadTRTEngine 加载trt模型
 * @param enginePath trt模型路径
 * @return
 */
nvinfer1::ICudaEngine *Utils::loadTRTEngine(const string enginePath,nvinfer1::IRuntime* runtime)
{
    ifstream gieModelStream(enginePath, ios::binary);
    if (!gieModelStream.good()){
        cerr << "ERROR: Could not read engine! \n";
        gieModelStream.close();
        return nullptr;
    }
    gieModelStream.seekg(0, ios::end);
    size_t modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, ios::beg);

    void* modelData = malloc(modelSize);
    if(!modelData)
    {
        cerr << "ERROR: Could not allocate memory for onnx engine! \n";
        gieModelStream.close();
        return nullptr;
    }
    gieModelStream.read((char*)modelData, modelSize);
    gieModelStream.close();

//    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);//创建runtime时
    if (runtime == nullptr) {
        cerr << "ERROR: Could not create InferRuntime! \n";
        return nullptr;
    }
    nvinfer1::ICudaEngine* trtengine=runtime->deserializeCudaEngine(modelData, modelSize, nullptr);//反序列化文件
    cout<<"sucessful get trt engine.................";
    return trtengine;
}
/**
 * @brief Utils::getClassNames 读取类型文件
 * @param imagenet_classes 类别名称
 * @return
 */
vector<string> Utils::getClassNames(const string &imagenet_classes)
{
    ifstream classes_file(imagenet_classes);
    vector<string> classes;
    if(!classes_file.good()){
        std::cerr<<"ERROR: can't read file with classes names .\n";
        return classes;
    }
    string class_name;
    while (getline(classes_file,class_name)) {
        classes.push_back(class_name);
    }
    return classes;
}
/**
 * @brief Utils::preprocess_img 图片预处理过程
 * @param img Mat对象
 * @return
 */
cv::Mat Utils::preprocess_img(cv::Mat& img)
{
    int w,h,x,y;
    float r_w=netWidth/(img.cols*1.0);
    float r_h=netHeight/(img.rows*1.0);
    if(r_h>r_w){
        w=netWidth;
        h=r_w*img.rows;
        x=0;
        y=(netHeight-h)/2;
    }else{
        w=r_h*img.cols;
        h=netHeight;
        x=(netWidth-w)/2;
        y=0;
    }
    cv::Mat re(h,w,CV_8UC3);
    cv::resize(img,re,re.size(),0,0,cv::INTER_LINEAR);
    cv::Mat out(netHeight,netWidth,CV_8UC3,cv::Scalar(128,128,128));
    re.copyTo(out(cv::Rect(x,y,re.cols,re.rows)));
    return out;
}
/**
 * @brief Utils::letterbox 实现yolov5图像的letterbox
 * @param img
 * @return
 */
cv::Mat Utils::letterbox(cv::Mat &img)
{
    //生成带边框的图像
    int in_w=img.cols;
    int in_h=img.rows;
    int tar_w=netWidth;
    int tar_h=netHeight;
    //那个缩放比例小选用那个
    float r=min(float(tar_h)/in_h,float(tar_w)/in_w);
    int inside_w=round(in_w*r);
    int inside_h=round(in_h*r);
    int padd_w=tar_w-inside_w;
    int padd_h=tar_h-inside_h;
    //内层图像resize
    cv::Mat resize_img;
    cv::resize(img,resize_img,cv::Size(inside_w,inside_h));
    padd_w=padd_w/2;
    padd_h=padd_h/2;
    //外层边框填充灰色
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cout << resize_img.size() << endl;
//    cv::imshow("pad", resize_img);
//    cv::waitKey(0);
    return resize_img;
}
/**
 * @brief Utils::doInference 整个流程的推理过程
 * @param context 执行的上下文
 * @param stream cuda流
 * @param buffers 数据缓存区
 * @param input 输入的Tensor
 * @param output 输出的Tensor
 * @param batchSize batchSize值
 */
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
/**
 * @brief Utils::getTRTResult 获取TRT引擎的执行结果
 * @param img_path 图片的路径
 * @param engine_path 引擎的地址
 * @param img 得到最终的结果
 * @return
 */
float* Utils::getTRTResult(string img_path, nvinfer1::ICudaEngine*  trtEngine, cv::Mat &img,nvinfer1::IExecutionContext* context,cudaStream_t stream,void* buffers[2])
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
/**
 * @brief Utils::mat2float 将Mat数据转换为float数据，转换为TensorRT可以接受的数据类型
 * @param data
 * @param img cvMat 数据
 * @return
 */
void Utils::mat2float(float *data, cv::Mat img)
{
    int i=0;
    for(int row=0;row<netHeight;++row){
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < netWidth; ++col) {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + netHeight * netWidth] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * netHeight * netWidth] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
//    return data;
}
/**
 * @brief Utils::mat2float2 第二种mat转float的方式
 * @param data
 * @param img
 * @return
 */
void Utils::mat2float2(float *data, cv::Mat img)
{
    unsigned int volChl = netHeight * netWidth;

    for(int c = 0; c < 3; ++c)
    {
        for (unsigned j = 0; j < volChl; ++j)
            data[c*volChl + j] = float(img.data[j * 3 + c]) / 255.0;
    }
}
/**
 * @brief Utils::get_rect 获得真实的坐标值
 * @param img
 * @param bbox
 * @return
 */
cv::Rect Utils::get_rect(cv::Mat &img, float bbox[4])
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

