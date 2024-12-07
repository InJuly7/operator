# >>> # batched matrix x batched matrix
# >>> tensor1 = torch.randn(10, 3, 4)
# >>> tensor2 = torch.randn(10, 4, 5)
# >>> torch.matmul(tensor1, tensor2).size()
# torch.Size([10, 3, 5])

import torch
from torch.utils.cpp_extension import load

if(torch.cuda.is_available()):
    device = "cuda"
else:
    print("The device is not support GPU ")
    exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pytroch 单算子实现
# 符合标准正态分布Tensor
d_input0 = torch.randn(10,256,1024, device=device)
d_input1 = torch.randn(10,1024,256, device=device)
# print(d_input0)
# print(d_input1)

d_output0 = torch.matmul(d_input0,d_input1)
# print(d_output0.shape) ([10,3,5])
# print(d_output0)

BatchMatmul_extension = load(
    name='bmm_extension',
    sources=['./bmm.cpp', './bmm.cu'],
    with_cuda=True,
    extra_cuda_cflags=["-O2", "-arch=sm_75"],
    build_directory='./load_inline_cuda/',
)

# 查看文档字符串
print(BatchMatmul_extension.bmm.__doc__)
d_output1 = BatchMatmul_extension.bmm(d_input0,d_input1)
print(d_output0)
