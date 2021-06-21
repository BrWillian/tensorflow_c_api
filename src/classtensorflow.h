#ifndef CLASSTENSORFLOW_H
#define CLASSTENSORFLOW_H
#include <tensorflow/c/c_api.h>
#include <string>


class ClassTensorflow
{
public:
    ClassTensorflow();
    void LoadModel(const char* modelPath);
    TF_Buffer* ReadBufferFromFile(const char* file);
    static void DeallocateBuffer(void* data, size_t) {
        std::free(data);
    }
};

#endif // CLASSTENSORFLOW_H
