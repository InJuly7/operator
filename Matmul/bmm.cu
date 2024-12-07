#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#define TILE_SIZE 16
// BM*BK = BN*BK
template<const int BK>
__global__ void bmm_kernel(float* d_input0, float* d_input1, float* d_output0, 
                            const int B, const int M, const int K, const int N)
{
    // 构建三维坐标(row,col,batch_idx)
    int batch_idx = blockIdx.z;
    // 行索引
    // int row = blockIdx.y*blockDim.y+threadIdx.y;
    // 列索引    
    // int col = blockIdx.x*blockDim.x+threadIdx.x;
    
    __shared__ float tile_input0[TILE_SIZE*BK];
    __shared__ float tile_input1[TILE_SIZE*BK];

    float temp = 0;
    float* temp_input0 = &d_input0[batch_idx*M*K+blockIdx.y*blockDim.y*K];
    float* temp_input1 = &d_input1[batch_idx*K*N+blockIdx.x*blockDim.x];
    float* temp_output0 = &d_output0[batch_idx*M*N+blockIdx.y*blockDim.y*N+blockIdx.x*blockDim.x];
    // step = K/BLOCK_SIZE
    for (int k = 0; k < K; k += BK)
    {
        tile_input0[threadIdx.y*BK+threadIdx.x] = temp_input0[threadIdx.y*K+threadIdx.x];
        tile_input1[threadIdx.y*blockDim.x+threadIdx.x] = temp_input1[threadIdx.y*N+threadIdx.x];

        __syncthreads();

        for(int i = 0; i < BK; i++)
        {
            temp += temp_input0[threadIdx.y*BK+i]*temp_input1[threadIdx.x+i*blockDim.x];
        }
        __syncthreads();
    }
    temp_output0[threadIdx.y*N+threadIdx.x] = temp;
}


torch::Tensor bmm(torch::Tensor d_input0, torch::Tensor d_input1)
{
    int B = d_input0.size(0);
    int M = d_input0.size(1);
    int K = d_input0.size(2);
    int N = d_input1.size(2);
    std::cout<< B << " " << M << " " <<  K << " " << N <<std::endl;

    // 创建输出矩阵 (B,M,N)
    torch::Tensor d_output0 = torch::zeros({B, M, N}, torch::device(d_input0.device()));

    // Block Grid 分配
    dim3 Block(TILE_SIZE,TILE_SIZE,1);
    dim3 Grid(N/TILE_SIZE,M/TILE_SIZE,B);
    bmm_kernel<TILE_SIZE><<<Grid, Block>>>(
        d_input0.data_ptr<float>(), d_input1.data_ptr<float>(), d_output0.data_ptr<float>(),
        B, M, K, N);
    cudaDeviceSynchronize();
    
    return d_output0;
}