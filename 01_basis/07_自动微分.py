import torch

# 定义输入变量，并设置 requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)

# f = (x + y) * (x - y)
# 前向传播
a = x + y  # a = x + y
b = x - y  # b = x - y
f = a * b  # f = a * b

# 反向传播
f.backward()  # 自动计算梯度

# 打印梯度
print("f 相对于 x 的梯度:", x.grad)  # 输出 4.0
print("f 相对于 y 的梯度:", y.grad)  # 输出 -2.0