import torch
from numpy.distutils.system_info import x11_info

""" 张量的创建 """
x = torch.arange(12)
#print(x)    # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
#print(x.shape)  # torch.Size([12])
#print(x.numel())    # 12

""" 改变张量的形状 """
X = x.reshape(3, 4) # 3行4列
#x1 = x.reshape(3,-1) # 自动计算列
#print(x1.shape)  # torch.Size([3, 4])
#print(X.shape)  # torch.Size([3, 4])
#print(X)

""" 全0张量 """
Zeros = torch.zeros((3, 4))
#print(Zeros)

""" 全1张量 """
Ones = torch.ones((3, 4))
#print(Ones)

""" 创建一个形状为（3,4）的张量。其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样 """
t1 = torch.randn(3, 4)
print(t1)

""" 通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值 """
t2 = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(t2)