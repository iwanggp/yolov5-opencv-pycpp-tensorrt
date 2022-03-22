#-------------------------------------------------
#
# Project created by QtCreator 2021-09-06T14:43:24
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TensorRTDemo
TEMPLATE = app
DEFINES += QT_DEPRECATED_WARNINGS
INCLUDEPATH += E:\OpenCV\opencv_450_install\install\include
CONFIG(release,debug|release){
    LIBS += E:\OpenCV\opencv_450_install\install\x64\vc15\lib\opencv_world450.lib
}

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
INCLUDEPATH += E:/cuda/CUDA/v10.0/include/
INCLUDEPATH += E:/software/TensorRT-7.2.3.4/include/
LIBS += E:/software/TensorRT-7.2.3.4/lib/nvinfer.lib
LIBS += E:/software/TensorRT-7.2.3.4/lib/nvonnxparser.lib
LIBS += E:/cuda/CUDA/v10.0/lib/x64/cuda.lib
LIBS += E:/cuda/CUDA/v10.0/lib/x64/cublas.lib
LIBS += E:/cuda/CUDA/v10.0/lib/x64/cudart.lib
LIBS += E:/cuda/CUDA/v10.0/lib/x64/cudnn.lib
# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    utils.cpp \
    yolov5.cpp

HEADERS += \
        mainwindow.h \
    utils.h \
    yolov5.h \
    dirent.h

FORMS += \
        mainwindow.ui
