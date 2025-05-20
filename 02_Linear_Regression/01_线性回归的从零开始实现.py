import torch
import random
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

""" 绘制散点图 """
# 设置图形大小
plt.figure(figsize=(6, 4))
# 绘制散点图
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# 添加标题和标签
plt.title('Scatter Plot of Features vs Labels')
plt.xlabel('Feature 1')
plt.ylabel('Label')
# 显示图形
# plt.show()

""" 读取数据集 """
# 生成大小为batch_size的小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 打乱数据集

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
'''
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
'''

""" 初始化模型参数 """
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

""" 定义模型 """
def linreg(X, w, b):
    return torch.matmul(X, w) + b

""" 定义损失函数 """
def squared_loss(y_hat, y):
    # 均方损失
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

""" 定义优化算法 """
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

""" 模型训练 """
# 学习率
lr = 0.03
# 轮次
num_epochs = 6
# 模型选择为linreg
net = linreg
# 损失函数选择为squared_loss
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in  data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量
        # l中的所有元素被加到一起，并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)

    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')