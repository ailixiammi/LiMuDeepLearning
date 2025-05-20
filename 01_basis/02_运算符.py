import torch

""" 按元素运算 """
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
#print(x + y)    # tensor([ 3.,  4.,  6., 10.])
#print(x - y)    # tensor([-1.,  0.,  2.,  6.])
#print(x * y)    # tensor([ 2.,  4.,  8., 16.])
#print(x / y)    # tensor([0.5000, 1.0000, 2.0000, 4.0000])
#print(x ** y)   # tensor([ 1.,  4., 16., 64.])
#print(torch.exp(x))   # tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])

""" 张量拼接 """
o1 = torch.ones(4).reshape(2, -1)
o2 = torch.ones(4).reshape(2, -1)
# 按第一个维度拼接(行增加)
O1 = torch.cat((o1, o2), dim=0)
# print(O1)
# 按第二个维度拼接(列增加)
O2 = torch.cat((o1, o2), dim=1)
# print(O2)

""" 通过逻辑创建bool张量 """
b1 = (x == y)
b2 = (x > y)
b3 = (x < y)
# print(b1) # tensor([False,  True, False, False])
# print(b2) # tensor([False, False,  True,  True])
# print(b3) # tensor([ True, False, False, False])

""" 对张量中的所有元素进行求和，会产生一个单元素张量 """
Tsum =  y.sum()
print(Tsum) # tensor(8)