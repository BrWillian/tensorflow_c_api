#ifndef CLASSTENSORFLOW_H
#define CLASSTENSORFLOW_H
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <string>


class ClassTensorflow
{
public:
    ClassTensorflow();
    ~ClassTensorflow();
    void LoadGraph(const char* modelPath);
    void Run(cv::Mat image_record);
    static void Deallocator(void* data, size_t lenght, void* arg);
private:
    static TF_Buffer* ReadBufferFromFile(const char* file);
    static void DeallocateBuffer(void* data, size_t);

    const char* tags = "serve";

    TF_Session* session;
    TF_Graph* graph;
};

#endif // CLASSTENSORFLOW_H
