#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16
// BM*BK = BN*BK
template<const int BM, const int BK, const int BN>
__global__ void bmm_kernel(float* d_input0, float* d_input1, float* d_output0, 
                            const int B, const int M, const int K, const int N)
{
    // 构建三维坐标(row,col,batch_idx)
    int batch_idx = blockIdx.z;
    // 行索引
    // int row = blockIdx.y*blockDim.y+threadIdx.y;
    // 列索引    
    // int col = blockIdx.x*blockDim.x+threadIdx.x;
    
    __shared__ float tile_input0[BM*BK];
    __shared__ float tile_input1[BK*BN];

    float temp = 0;
    // (0, blockIdx.y*blockDim.y, batch_idx)
    float* temp_input0 = &d_input0[blockIdx.y*blockDim.y*K + batch_idx*M*K];
    // (blockIdx.x*blockDim.x, 0, batch_idx)
    float* temp_input1 = &d_input1[blockIdx.x*blockDim.x + batch_idx*K*N];
    // (blockIdx.x*blockDim.x, blockIdx.y*blockDim.y, batch_idx)
    float* temp_output0 = &d_output0[blockIdx.x*blockDim.x + blockIdx.y*blockDim.y*N + batch_idx*M*N];

    // step = K/BLOCK_SIZE
    for (int k = 0; k < K; k += BK)
    {
        // (threadIdx.x, threadIdx.y)               (threadIdx.x, threadIdx.y) + offset
        tile_input0[threadIdx.x + threadIdx.y*BK] = temp_input0[threadIdx.x + threadIdx.y*K + k];
        // (threadIdx.x, threadIdx.y)               (threadIdx.x, threadIdx.y) + offset
        tile_input1[threadIdx.x + threadIdx.y*blockDim.x] = temp_input1[threadIdx.x + threadIdx.y*N + k*N];

        __syncthreads();

        for(int i = 0; i < BK; i++)
        {   // (i, threadIdx.y)         (threadIdx.x, i)
            temp += tile_input0[i + threadIdx.y*BK]*tile_input1[threadIdx.x + i*BN];
        }
        __syncthreads();
    }
    // (threadIdx.x, threadIdx.y)
    temp_output0[threadIdx.x + threadIdx.y*N] = temp;
}


torch::Tensor bmm(torch::Tensor d_input0, torch::Tensor d_input1)
{   
    // 扩展功能未实现
    if(d_input0.dim() == 3)
    {
        int B = d_input0.size(0);
        int M = d_input0.size(1);
        int K = d_input0.size(2);
        int N = d_input1.size(2);
        std::cout << "Input Matmax Dim : 3" << std::endl;
        std::cout << "B M K N" << std::endl; 
        std::cout<< B << " " << M << " " <<  K << " " << N <<std::endl;
        // 创建输出矩阵 (B,M,N)
        torch::Tensor d_output0 = torch::zeros({B, M, N}, torch::device(d_input0.device()));

        // Block Grid 分配
        dim3 Block(TILE_SIZE,TILE_SIZE,1);
        dim3 Grid(N/TILE_SIZE,M/TILE_SIZE,B);
        bmm_kernel<TILE_SIZE,TILE_SIZE,TILE_SIZE><<<Grid, Block>>>(
            d_input0.data_ptr<float>(), d_input1.data_ptr<float>(), d_output0.data_ptr<float>(),
            B, M, K, N);
        cudaDeviceSynchronize();
        return d_output0;
    }
    
    else if(d_input0.dim() == 5)
    {
        int B0 = d_input0.size(0);
        int B1 = d_input0.size(1);
        int B2 = d_input0.size(2);
        int B = B0*B1*B2;
        int M = d_input0.size(3);
        int K = d_input0.size(4);
        int N = d_input1.size(4);
        std::cout << "Input Matmax Dim : 5" << std::endl;
        std::cout << "B0 B1 B2 M K N" << std::endl;
        std::cout<< B0 << " " << B1 << " " << B2 << " " << M << " " <<  K << " " << N <<std::endl;
        torch::Tensor d_output0 = torch::zeros({B0, B1, B2, M, N}, torch::device(d_input0.device()));

        // Block Grid 分配
        dim3 Block(TILE_SIZE,TILE_SIZE,1);
        dim3 Grid(N/TILE_SIZE,M/TILE_SIZE,B);
        bmm_kernel<TILE_SIZE,TILE_SIZE,TILE_SIZE><<<Grid, Block>>>(
            d_input0.data_ptr<float>(), d_input1.data_ptr<float>(), d_output0.data_ptr<float>(),
            B, M, K, N);
        cudaDeviceSynchronize();
        return d_output0;
    }
}
