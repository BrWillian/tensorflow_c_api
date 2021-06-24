#include <classtensorflow.h>

int main()
{

    std::string modelPath = "./saved_model";

    ClassTensorflow network;
    network.LoadGraph(modelPath.c_str());

}
