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
d_input0 = torch.arange(1, 33, dtype=torch.float, device=device).repeat(32, 1) # 生成 (32, 32)
d_input0 = d_input0.unsqueeze(0)  # 添加 batch 维度，变成 (1, 32, 32)
print(d_input0)
d_input1 = torch.ones(1,32,32, device=device)
# print(d_input0)
# print(d_input1)

d_output0 = torch.matmul(d_input0,d_input1)
print(d_output0)


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
print(d_output1)
print(torch.allclose(d_output0,d_output1,atol=1e-2))
