import torch
import numpy as np

""" numpy中的数组与张量互相转换 """
x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
A = x.numpy()
B = torch.tensor(A)

print(type(A))  # <class 'numpy.ndarray'>
print(type(B))  # <class 'torch.Tensor'>