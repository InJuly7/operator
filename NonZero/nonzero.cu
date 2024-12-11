#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <torch/types.h>
// #include <cub/cub.cuh>
// 最大支持的张量维度数
#define TILE_SIZE 32

// 核函数：标记非零元素的位置
__global__ void mark_nonzero(const float* d_input0, int* d_flags, int* d_count, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && d_input0[idx] != 0)
    {
        int pos = atomicAdd(d_count, 1);
        d_flags[idx] = pos;
    }
    else d_flags[idx] = size;
}

// 核函数：从标记中提取非零元素的索引
__global__ void collect_indices(const int* d_flags, int* d_linear_indices, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size && (d_flags[idx] != size))
    {
        int pos = d_flags[idx];
        d_linear_indices[pos] = idx;
    }
}

template <const int tile_size>
__global__ void SortKernel(int* d_linear_indices, int* d_sort_linear_indices, int num_nonzero)
{
	int tid = threadIdx.x;
	int ttid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    int val = -1;
    __shared__ int cache[tile_size];
    if(ttid < num_nonzero)
    {
        val = d_linear_indices[ttid];
    }
        
    // 将数据加载到shared_mem
    for (int block_start = 0; block_start < num_nonzero; block_start += tile_size)
    {
        int idx = block_start+tid;
        if(idx < num_nonzero)
        {
            cache[tid] = d_linear_indices[idx];
        }
        __syncthreads();
        if(ttid < num_nonzero)
        {
            // 如果最后一个块 加载的不完整
            int vaild_count = min(num_nonzero-(((idx)/tile_size)*tile_size), tile_size);
            for (int j = 0; j < vaild_count; ++j)
            {
                if (val > cache[j])
                    count++;
            }
            __syncthreads();
        }
    }
    if(ttid < num_nonzero)
	    d_sort_linear_indices[count] = val;

}

// 核函数：从展平索引转换为多维索引
__global__ void compute_indices(const int* d_sort_linear_indices, int* d_output0, const int* d_shape, int num_nonzero, int dims) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_nonzero)
    {
        int flat_idx = d_sort_linear_indices[idx];
        for (int dim = dims - 1; dim >= 0; --dim)
        {
            d_output0[idx * dims + dim] = flat_idx % d_shape[dim];
            flat_idx /= d_shape[dim];
        }
    }
}

torch::Tensor nonzero(torch::Tensor d_input0)
{
    int size = d_input0.numel();
    int dims = d_input0.dim();
    torch::Tensor d_flags = torch::empty_like(d_input0, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor d_num_nonzero = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    // std::cout<<d_flags<<std::endl;
    int blockSize = 32;
    int Grid = (size + blockSize - 1) / blockSize;
    mark_nonzero<<<Grid, blockSize>>>(d_input0.data_ptr<float>(), d_flags.data_ptr<int>(), d_num_nonzero.data_ptr<int>(), size);
    cudaDeviceSynchronize();
    // std::cout<< d_flags << std::endl;
    
    int num_nonzero = d_num_nonzero.item<int>();
    // std::cout<< num_nonzero << std::endl;

    // Debug
    // std::vector<int32_t> data = {
    //     1, 5, 8, 9, 11, 13, 14, 15, 18, 19, 20, 22, 23, 25, 27, 28, 30,
    //     65, 66, 67, 68, 69, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    //     84, 88, 89, 33, 34, 35, 37, 39, 40, 42, 44, 45, 46, 47, 48, 49,
    //     50, 52, 54, 55, 56, 58, 59, 60, 62, 63
    // };
    // num_nonzero = 59;
    // torch::Tensor d_linear_indices_cpu = torch::from_blob(data.data(), {num_nonzero}, torch::TensorOptions().dtype(torch::kInt32));
    // torch::Tensor d_linear_indices = d_linear_indices_cpu.to(torch::kCUDA);


    torch::Tensor d_linear_indices = torch::empty({num_nonzero}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor d_sort_linear_indices = torch::full({num_nonzero}, -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    // std::cout<< d_sort_linear_indices << std::endl;
    collect_indices<<<Grid, blockSize>>>(d_flags.data_ptr<int>(), d_linear_indices.data_ptr<int>(), size);
    cudaDeviceSynchronize();
    // std::cout<< d_linear_indices << std::endl;
    SortKernel<TILE_SIZE><<<(num_nonzero+TILE_SIZE-1)/TILE_SIZE,TILE_SIZE>>>(d_linear_indices.data_ptr<int>(), 
                                                                            d_sort_linear_indices.data_ptr<int>(), num_nonzero);
    cudaDeviceSynchronize();
    // std::cout<< d_sort_linear_indices << std::endl;

    torch::Tensor d_output0 = torch::empty({num_nonzero, dims}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor d_shape = torch::empty({dims}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto shape = d_input0.sizes();
    for (int i = 0; i < dims; ++i)
    {
        d_shape[i] = static_cast<int>(shape[i]);
    }
    compute_indices<<<(num_nonzero+TILE_SIZE-1)/TILE_SIZE,TILE_SIZE>>>(d_sort_linear_indices.data_ptr<int>(), d_output0.data_ptr<int>(),
                                            d_shape.data_ptr<int>(), num_nonzero, dims);
    cudaDeviceSynchronize();
    return d_output0;
}




// 主函数：CUDA 实现 Nonzero
// 主函数：CUDA 实现 Nonzero
// std::vector<std::vector<int>> cuda_nonzero(const std::vector<int>& data, const std::vector<int>& dims) {
//     int size = data.size();
//     int ndim = dims.size();

//     // 分配 GPU 内存
//     int *d_data, *d_dims, *d_count, *d_flags;
//     cudaMalloc(&d_data, size * sizeof(int));
//     cudaMalloc(&d_dims, ndim * sizeof(int));
//     cudaMalloc(&d_flags, size * sizeof(int));
//     cudaMalloc(&d_count, sizeof(int));

//     // 初始化计数器
//     int zero = 0;
//     cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

//     // 拷贝数据到设备
//     cudaMemcpy(d_data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_dims, dims.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);

//     // 标记非零元素
//     int blockSize = 256;
//     int numBlocks = (size + blockSize - 1) / blockSize;
//     mark_nonzero<<<numBlocks, blockSize>>>(d_data, d_flags, d_count, size);
//     cudaDeviceSynchronize();

//     // // Debug
//     // int* h_flags = new int[size];
//     // cudaMemcpy(h_flags, d_flags, size*sizeof(int), cudaMemcpyDeviceToHost);
//     // for(int i = 0; i < size; i++) std::cout<< h_flags[i] << std::endl;
//     // exit(0);
    
//     // 拷贝非零元素计数到主机
//     int num_nonzero = 0;
//     cudaMemcpy(&num_nonzero, d_count, sizeof(int), cudaMemcpyDeviceToHost);;
//     int* d_linear_indices, *d_sort_linear_indices;
//     cudaMalloc(&d_linear_indices, num_nonzero * sizeof(int));
//     cudaMalloc(&d_sort_linear_indices, num_nonzero * sizeof(int));

//     // 提取非零元素的线性索引
//     cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
//     collect_indices<<<numBlocks, blockSize>>>(d_flags, d_linear_indices, size);
//     cudaDeviceSynchronize();

//     // Debug
//     // int* h_linear_indices = new int[num_nonzero];
//     // cudaMemcpy(h_linear_indices, d_linear_indices, num_nonzero*sizeof(int), cudaMemcpyDeviceToHost);
//     // for(int i = 0; i < num_nonzero; i++) std::cout<< h_linear_indices[i] << std::endl;
//     // exit(0);

//     SortKernel<TILE_SIZE><<<(num_nonzero+TILE_SIZE-1)/TILE_SIZE,TILE_SIZE>>>(d_linear_indices, d_sort_linear_indices, num_nonzero);
//     cudaDeviceSynchronize();

//     // Debug
//     // int* h_sort_linear_indices = new int[num_nonzero];
//     // cudaMemcpy(h_sort_linear_indices, d_sort_linear_indices, num_nonzero*sizeof(int), cudaMemcpyDeviceToHost);
//     // for(int i = 0; i < num_nonzero; i++) std::cout<< h_sort_linear_indices[i] << std::endl;
//     // exit(0);

//     // 排序线性索引以保证有序性
//     // cub 执行
//     // void* d_temp_storage = nullptr;
//     // int* d_sorted_indices;
//     // size_t temp_storage_bytes = 0;
//     // cudaMalloc(&d_sorted_indices, size * sizeof(int));
//     // cub::DeviceRadixSort::SortKeys(
//     //     d_temp_storage, temp_storage_bytes,
//     //     d_linear_indices, d_sort_linear_indices, num_nonzero
//     // );
//     // cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     // cub::DeviceRadixSort::SortKeys(
//     //     d_temp_storage, temp_storage_bytes,
//     //     d_linear_indices, d_sort_linear_indices, num_nonzero
//     // );

//     // cuda 执行
    
//     // 分配多维索引输出
//     int* d_output;
//     cudaMalloc(&d_output, num_nonzero * ndim * sizeof(int));

//     // 计算多维索引
//     compute_indices<<<numBlocks, blockSize>>>(d_sort_linear_indices, d_output, d_dims, num_nonzero, ndim);

//     // 拷贝结果回主机
//     std::vector<int> output_flat(num_nonzero * ndim);
//     cudaMemcpy(output_flat.data(), d_output, num_nonzero * ndim * sizeof(int), cudaMemcpyDeviceToHost);

//     // 释放 GPU 内存
//     cudaFree(d_data);
//     cudaFree(d_dims);
//     cudaFree(d_flags);
//     cudaFree(d_linear_indices);
//     cudaFree(d_sort_linear_indices);
//     cudaFree(d_count);
//     cudaFree(d_output);

//     // 整理结果
//     std::vector<std::vector<int>> result(num_nonzero, std::vector<int>(ndim));
//     for (int i = 0; i < num_nonzero; ++i) {
//         for (int j = 0; j < ndim; ++j) {
//             result[i][j] = output_flat[i * ndim + j];
//         }
//     }
//     return result;
// }

// 测试
// int main() {
//     std::vector<int> data = {1, 0, 3, 0, 
//                             5, 0, 7, 8,
//                             1, 0, 3, 0, 
//                             5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8,
//                             1, 0, 3, 0, 5, 0, 7, 8}; // 展平张量
//     std::vector<int> dims = {6, 6, 4};                 // 张量维度

//     auto result = cuda_nonzero(data, dims);

//     std::cout << "Non-zero indices:" << std::endl;
//     for (const auto& idx : result) {
//         for (int dim : idx) {
//             std::cout << dim << " ";
//         }
//         std::cout << std::endl;
//     }

//     return 0;
// }
