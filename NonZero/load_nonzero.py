# https://pytorch.org/docs/stable/generated/torch.nonzero.html

import torch
from torch.utils.cpp_extension import load

if(torch.cuda.is_available()):
    device = "cuda"
else:
    print("The device is not support GPU ")
    exit(0)




# Tensor 1D
# d_input0 = torch.tensor([1,0,2,3,0,4,5,0,7,0,8], device=device)

# Tensor 2D
# d_input0 = torch.tensor([[1, 0, 3], [0, 5, 0], [7, 0, 9]], device=device)

# Tensor 3D
# d_input0 = torch.tensor([[[0, 1], [2, 0]], [[3, 0], [0, 4]]], device=device)
# d_input0 = torch.tensor([[0.6, 0.0, 0.0, 0.0],[0.0, 0.4, 0.0, 0.0],[0.0, 0.0, 1.2, 0.0],[0.0, 0.0, 0.0,-0.4]],device=device)

import torch

# 1. 生成一个多维张量
tensor = torch.rand(3, 1024, 1024, device=device)

# 1 --> prob_one , 0 --> (1-prob_one) 按概率生成掩码
prob_one = 0.7
mask = torch.full((3,1024,1024), prob_one, device=device)
mask = torch.bernoulli(mask)
# 逐元素乘法
d_input0 = tensor * mask

# 打印结果
# print("Original Tensor:")
# print(tensor)
# print("\nMask:")
# print(mask)
print("\nTensor with Random Zeros:")
print(d_input0.shape)
# print(d_input0)

print("Torch Nonzero Operator Result")
d_output0 = torch.nonzero(d_input0,as_tuple=False).to(dtype=torch.int)
print(d_output0.shape)
# print(d_output0)

print("Custom Kernel Nonzero Result")


Nonzero_extension = load(
    name='Nonzero_extension',
    sources=['./nonzero.cpp', './nonzero.cu'],
    with_cuda=True,
    extra_cuda_cflags=["-O2", "-arch=sm_75"],
    build_directory='./load_inline_cuda/',
)

# 查看文档字符串
print(Nonzero_extension.nonzero.__doc__)
d_output1 = Nonzero_extension.nonzero(d_input0)
print(d_output1.shape)
# print(d_output1)

# mismatch_indices = (d_output0 != d_output1).nonzero(as_tuple=False)
# if mismatch_indices.numel() > 0:
#     print(f"Found {mismatch_indices.size(0)} mismatched elements.")

# 校验 输出结果是否一致 
print(torch.allclose(d_output0,d_output1,atol=1e-2))

