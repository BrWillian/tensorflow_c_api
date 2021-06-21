#include "classtensorflow.h"
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <fstream>

ClassTensorflow::ClassTensorflow()
{

}
void ClassTensorflow::LoadModel(const char* modelPath)
{

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
