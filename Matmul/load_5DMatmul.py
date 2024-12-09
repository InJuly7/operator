# >>> # 5D matrix x 5D matrix
# >>> tensor1 = torch.randn(10, 3, 2, 1024, 256)
# >>> tensor2 = torch.randn(10, 3, 2, 256, 1024)
# >>> torch.matmul(tensor1, tensor2).size()

import torch
from torch.utils.cpp_extension import load

if(torch.cuda.is_available()):
    device = "cuda"
else:
    print("The device is not support GPU ")
    exit(0)

d_input0 = torch.randn(3, 3, 2, 1024, 256, device=device)
d_input1 = torch.randn(3, 3, 2, 256, 1024, device=device)

# Pytroch 单算子实现
d_output0 = torch.matmul(d_input0, d_input1)
print(d_output0.shape)
print("Torch Output : ")
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
print("BMM Kernel Output : ")
print(d_output1)

# 校验 输出结果是否一致 
print(torch.allclose(d_output0,d_output1,atol=1e-2))