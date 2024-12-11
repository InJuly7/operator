import torch
device = "cuda"
d_input0 = torch.randn(3, 3, 2, 1024, 256, device=device)
print(d_input0.shape)
print(d_input0.size())