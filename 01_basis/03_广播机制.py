import torch

""" 广播机制：将张量自动调整为可以进行运算的大小，主要是通过行、列的复制实现 """
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)

# 触发广播机制，a按列复制，b按行复制
c = a + b
print(c)