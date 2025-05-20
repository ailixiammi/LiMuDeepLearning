import torch
import numpy as np
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

""" 生成数据集 """
# w:权重，b:偏移，num_examples:样本数量
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 数据符合0~1之间的正态分布，形状为num_examples*len(w)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # 噪音

    return X, y.reshape(-1, 1)

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# print(features)
# print(labels)
# print('features:', features[0],'\nlabel:', labels[0])

""" 使用框架中的API来读取数据集 """
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))


""" 使用torch中的全连接层 """
net = nn.Sequential(nn.Linear(2, 1))

""" 初始化模型参数 """
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

""" 定义损失函数 """
loss = nn.MSELoss()

""" 定义优化算法 """
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

""" 训练 """
num_epochs = 6
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)

    print(f'epoch {epoch + 1}, loss {l:f}')