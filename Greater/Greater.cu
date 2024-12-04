#include <iostream>
__global__ void Greater222(float* d_input0, float* d_input1, bool* d_output0, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        d_output0[i] = (d_input0[i] > d_input1[i])? 1:0;
    }
}


// 二维张量 Greater
void GreaterKernel(float* d_input0, float* d_input1, bool* d_output0, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    std::cout<< "Test Greater 222 "<<std::endl;
    Greater222<<<dimGrid,dimBlock>>>(d_input0,d_input1,d_output0,numElements);
}

int main()
{
    // test Greater 222 
    const int height = 1024, width = 1024;
    int numElements = height * width;
    float* h_input0 = new float[numElements];
    float* h_input1 = new float[numElements];
    // 输出 bool 类型
    bool* h_output0 = new bool[numElements]; 
    
    for (int i = 0; i < numElements; i++)
    {
        h_input0[i] = static_cast<float>(i);
        if(i%2 == 1) h_input1[i] = h_input0[i]+1;
        else h_input1[i] = h_input0[i]-1;
    }
    
    float *d_input0, *d_input1;
    bool *d_output0;
    cudaMalloc(&d_input0, numElements * sizeof(float));
    cudaMalloc(&d_input1, numElements * sizeof(float));
    cudaMalloc(&d_output0, numElements * sizeof(bool));
    cudaMemcpy(d_input0, h_input0, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input1, h_input1, numElements * sizeof(float), cudaMemcpyHostToDevice);
    
    GreaterKernel(d_input0, d_input1, d_output0, numElements);
    cudaMemcpy(h_output0, d_output0, numElements * sizeof(bool), cudaMemcpyDeviceToHost);
    std::cout << "Output (first 10 values): \n";
    
    for (int i = 0; i < 10; i++)
    {
        std::cout << h_input0[i] <<" ";
        std::cout << h_input1[i] <<" --> "; 
        printf("%s", h_output0[i] ?"true":"false");
        std::cout << std::endl;
    }
    
    cudaFree(d_input0);
    cudaFree(d_input1);
    cudaFree(d_output0);

    delete[] h_input0;
    delete[] h_input1;
    delete[] h_output0;

    return 0;
}