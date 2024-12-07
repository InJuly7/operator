#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


template<const int BLOCK_SIZE>
__global__ void bmm_kernel(const float* d_input0, const float* d_input1, float* d_output0, 
                            const int B, const int M, const int K, const int N)
{
    // 构建三维坐标(row,col,batch_idx)
    int batch_idx = blockIdx.z;
    // 行索引
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    // 列索引    
    int col = blockIdx.x*blockDim.x+threadIdx.x
    
    __shared__ float tile_input0[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_input1[BLOCK_SIZE][BLOCK_SIZE];
    
    
    float temp = 0.0;
    
    
}


torch::Tensor bmm(torch::Tensor d_input0, torch::Tensor d_input1)
{
    int B = d_input0.size(0);
    int M = d_input0.size(1);
    int K = d_input0.size(2);
    int N = d_input1.size(2);
    std::cout<< B << " " << M << " " <<  K << " " << N <<std::endl;

    // 创建输出矩阵 (B,M,N)
    torch::Tensor d_output0 = torch::zeros({B, M, N}, torch::device("cuda"));

    // Block Grid 分配
    dim3 Block(TILE,TILE,1);
    dim3 Grid(
        (N+TILE_DIM-1)/TILE_DIM,
        (M+TILE_DIM-1)/TILE_DIM,
        B
    );
    bmm_kernel<<<Grid, Block>>>(
        d_input0.data_ptr<float>(), d_input1.data_ptr<float>(), d_output0.data_ptr<float>(),
        B, M, K, N);
    
    return d_output0;
}