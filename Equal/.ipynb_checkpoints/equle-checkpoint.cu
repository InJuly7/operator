#include <iostream>
#define TEST_EQUAL222 1
#define TEST_EQUAL303 !(TEST_EQUAL222)
__global__ void Equal222(const float *d_input0, const float *d_input1, float *d_output0, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        d_output0[i] = (d_input0[i] == d_input1[i]) ? 1 : 0; 
    }
}

__global__ void Equal303(const float *d_input0, float value,float *d_output0, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        d_output0[i] = (d_input0[i] == value) ? 1 : 0; 
    }
}

void EqualKernel(const float *d_input0, float *d_input1, float *d_output0, float value, int numElements)
{
    dim3 dimBlock = (256);
    dim3 dimGrid = ((numElements+256-1)/256);
    
    // Test Equal 222
    if(d_input1 != NULL)
    {   
        std::cout<< "Test Equal 222 (d_input1 != NULL)"<<std::endl;
        Equal222<<<dimGrid,dimBlock>>>(d_input0,d_input1,d_output0,numElements);
    }

    // Test Equal 303
    else
    {
        std::cout<< "Test Equal 303 (d_input1 == NULL) value: "<<value<<std::endl;
        Equal303<<<dimGrid,dimBlock>>>(d_input0,value,d_output0,numElements);
    }
    cudaDeviceSynchronize();
}

int main()
{
    // test Equal 222 
    // 不涉及 广播 两个输入shape相同
    if(TEST_EQUAL222)
    {
        const int height = 1024, width = 1024;
        int value = 0;
        int numElements = height * width;
        float* h_input0 = new float[numElements];
        float* h_input1 = new float[numElements];
        float* h_output0 = new float[numElements]; 
        
        for (int i = 0; i < numElements; i++)
        {
            h_input0[i] = static_cast<float>(i);
            if(i%2 == 1) h_input1[i] = (-1)*static_cast<float>(i);
            else h_input1[i] = static_cast<float>(i);
        }
        float *d_input0, *d_input1, *d_output0;
        cudaMalloc(&d_input0, numElements * sizeof(float));
        cudaMalloc(&d_input1, numElements * sizeof(float));
        cudaMalloc(&d_output0, numElements * sizeof(float));  
        cudaMemcpy(d_input0, h_input0, numElements * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input1, h_input1, numElements * sizeof(float), cudaMemcpyHostToDevice);
        EqualKernel(d_input0, d_input1, d_output0, value, numElements);
        cudaMemcpy(h_output0, d_output0, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Output (first 10 values): \n";
        for (int i = 0; i < 10; i++) {
            std::cout << h_input0[i] <<" ";
            std::cout << h_input1[i] <<" --> "; 
            std::cout << h_output0[i] << " "<<std::endl;
        }
        std::cout << std::endl;
        cudaFree(d_input0);
        cudaFree(d_input1);
        cudaFree(d_output0);

        delete[] h_input0;
        delete[] h_input1;
        delete[] h_output0;
        
    }
    
    if(TEST_EQUAL303)
    {
        const int channel = 3, height = 1024, width = 1024;
        int value = 8;
        int numElements = channel*height * width;
        float* h_input0 = new float[numElements];
        float* h_output0 = new float[numElements]; 
        
        for (int i = 0; i < numElements; i++)
        {
            h_input0[i] = static_cast<float>(i);
        }

        h_input0[0] = value;
        h_input0[1] = value;
        h_input0[2] = value;
        h_input0[3] = value;
        
        float *d_input0, *d_output0;
        // 用于判决 第二个输入是Tensor 还是 constant 
        float *d_input1 = NULL;
        cudaMalloc(&d_input0, numElements * sizeof(float));
        cudaMalloc(&d_output0, numElements * sizeof(float));  
        cudaMemcpy(d_input0, h_input0, numElements * sizeof(float), cudaMemcpyHostToDevice);
        EqualKernel(d_input0, d_input1, d_output0, value, numElements);
        cudaMemcpy(h_output0, d_output0, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Output (first 10 values): \n";
        for (int i = 0; i < 10; i++) {
            std::cout << h_input0[i] <<" value "<<value<<" --> ";;
            std::cout << h_output0[i] << " "<<std::endl;
        }
        std::cout << std::endl;
        cudaFree(d_input0);
        cudaFree(d_input1);
        cudaFree(d_output0);

        delete[] h_input0;
        delete[] h_output0;
    }
    return 0;
}