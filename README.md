# yolov5-opencv-pycpp-tensorrt

本文目录如下：

- [ ] Python版Opencv部署Yolov5
- [ ] CPP版Opencv部署Yolov5
- [ ] TensorRT版部署Yolov5

### Python版用Opencv部署Yolov5

这里默认使用了**CUDA**版本的OpenCV(有快速的没什么不用)，关于编译Opencv CUDA版本可以参考[编译CUDA版的OpenCV](https://www.jianshu.com/p/2ec17797a924)用DNN模块来加载ONNX模型。从此模型的部署就彻底地摆脱了深度学习框架的依赖。

python版本的主要主程序**python/python_deploy.py**

