#include "classtensorflow.h"
#include <tensorflow/c/c_api.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>

ClassTensorflow::ClassTensorflow()
{

}
void ClassTensorflow::LoadGraph(const char *modelPath)
{
    graph = TF_NewGraph();
    TF_Status* Status = TF_NewStatus();
    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = nullptr;

    const char* tags = "serve";

    session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, modelPath, &tags, 1, graph, nullptr, Status);

    if (TF_GetCode(Status) == TF_OK){
        std::cout<<"Tensorflow model loaded...."<<std::endl;
    }else {
        std::cout<<TF_Message(Status);
    }

    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

}
TF_Buffer* ClassTensorflow::ReadBufferFromFile(const char* file)
{
    std::ifstream f(file, std::ios::binary);

    if (f.fail() || !f.is_open()){
            return nullptr;
    }

    f.seekg(0, std::ios::end);
    const auto fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    if (fsize < 1) {
        f.close();
        return nullptr;
    }


    char* data = static_cast<char*>(std::malloc(fsize));
    f.read(data, fsize);
    f.close();

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;
    return buf;
}
void ClassTensorflow::DeallocateBuffer(void *data, size_t)
{
    std::free(data);
}
void ClassTensorflow::Deallocator(void *data, size_t lenght, void *arg){
    std::cout<<"Dellocation of the input tensor"<<std::endl;
    std::free(data);
    data = nullptr;
}
ClassTensorflow::~ClassTensorflow()
{
    TF_DeleteGraph(graph);
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout<<"Error close session"<<std::endl;
    }
    TF_DeleteSession(session, status);
    TF_DeleteStatus(status);
}
