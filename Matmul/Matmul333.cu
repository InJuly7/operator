// This program computes a simple version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

// TILE_SIZE >= BLOCK_SIZE
const int TILE_SIZE = 1 << 10;

// Matrix size of 1024 x 1024;
// M*K * K*N = M*N
const int M = 1 << 10;
const int N = 1 << 10;
const int K = 1 << 10;
// 定义第三维度数值
const int C = 3;


__global__ void matrixMul(const float *a, const float *b, float *c)
{
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Statically allocated shared memorya
    // 最好满足 block_size >= tile_size, 加载数据时候每个线程至少可以加载4B
    __shared__ float s_a[TILE_SIZE];
    __shared__ float s_b[TILE_SIZE];
    
    // Accumulate in temporary variable
    int tmp = 0;
    int step = blockDim.x;
    // Sweep tile across matrix
    for (int i = 0; i < K; i += step)
    {
    // Load in elements for this tile
        // 块内索引 与 共享内存映射
        // 线程索引 与 输入矩阵映射
    s_a[threadIdx.y * step + threadIdx.x] = a[row * K + i + threadIdx.x];
    s_b[threadIdx.y * step + threadIdx.x] = b[i * N + threadIdx.y * N + col];
    
    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();
    
    // Do matrix multiplication on the small matrix
    for (int j = 0; j < step; j++) 
        {
      tmp += s_a[threadIdx.y * step + j] * s_b[j * blockDim.x + threadIdx.x];
    }
    
    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
    }
    
    // Write back results
    c[row * N + col] = tmp;
}

// Check result on the CPU
// MxN = MxK * KxN
 void verify_result(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, int M, int N, int K) {
   // For every row...
   for (int row = 0; row < M; row++) {
    // For every column...
     for (int col = 0; col < N; col++) {
      // For every element in the row-column pair
      float tmp = 0;
      for (int i = 0; i < K; i++) {
        // Accumulate the partial results
        tmp += a[row * N + i] * b[i * N + col];
      }
      // Check against the CPU result
      assert(tmp == c[row * N + col]);
     }
  }
}

// 三维矩阵乘法
int main() {

    // Size (in bytes) of matrix
    size_t MatA_bytes = C * M * N * sizeof(float);
    size_t MatB_bytes = C * N * K * sizeof(float);
    size_t MatC_bytes = C * M * K * sizeof(float);
    // Host vectors
    std::vector<float> h_a(MatA_bytes);
    std::vector<float> h_b(MatB_bytes);
    std::vector<float> h_c(MatC_bytes);
    
    // Initialize matrices
    std::generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    std::generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, MatA_bytes);
    cudaMalloc(&d_b, MatB_bytes);
    cudaMalloc(&d_c, MatC_bytes);
    
    // Copy data to the device
    cudaMemcpy(d_a, h_a.data(), MatA_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), MatB_bytes, cudaMemcpyHostToDevice);
    
    // Threads per CTA dimension
    int THREADS = 32;
    
    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCK_X = M / THREADS;
    int BLOCK_Y = K / THREADS;
    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCK_X, BLOCK_Y);
    
    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
    
    // Copy back to the host
    cudaMemcpy(h_c.data(), d_c, MatC_bytes, cudaMemcpyDeviceToHost);
    
    // Check result
    // verify_result(h_a, h_b, h_c, M, N, K);
    std::cout << "COMPLETED SUCCESSFULLY\n";
    
    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}