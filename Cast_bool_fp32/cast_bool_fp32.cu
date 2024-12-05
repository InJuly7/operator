#include <iostream>
__global__ void Cast_Bool_FP32(bool* d_input0, float* d_output0, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        d_output0[i] = (d_input0[i])? 1.0:0.0;
    }
}

// 二维张量 Greater
void CastBoolFP32Kernel(bool* d_input0, float* d_output0, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    std::cout<< "Test Cast_Bool_FP32 "<<std::endl;
    Cast_Bool_FP32<<<dimGrid,dimBlock>>>(d_input0,d_output0,numElements);
}

int main()
{
    const int height = 1024, width = 1024;
    int numElements = height * width;
    // 输入是bool类型
    bool* h_input0 = new bool[numElements];
    float* h_output0 = new float[numElements]; 
    
    for (int i = 0; i < numElements; i++)
    {
        if(i%2 == 1) h_input0[i] = true;
        else h_input0[i] = false;
    }
    
    bool *d_input0;
    float *d_output0;
    cudaMalloc(&d_input0, numElements * sizeof(bool));
    cudaMalloc(&d_output0, numElements * sizeof(float));
    cudaMemcpy(d_input0, h_input0, numElements * sizeof(bool), cudaMemcpyHostToDevice);
    CastBoolFP32Kernel(d_input0, d_output0, numElements);
    cudaMemcpy(h_output0, d_output0, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Output (first 10 values): \n";
    
    for (int i = numElements; i > numElements-10; i--)
    {
        printf("%s", h_input0[i] ?"true":"false");
        std::cout <<" --> "<< h_output0[i] << std::endl;
    }
    
    cudaFree(d_input0);
    cudaFree(d_output0);

    delete[] h_input0;
    delete[] h_output0;

    return 0;
}