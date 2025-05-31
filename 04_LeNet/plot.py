""" 下载处理数据集 """
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 下载训练数据集，resize为224*224，并转换为张量格式
train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

# 使用PyTorch的DataLoader类来创建一个数据加载器（DataLoader），用于在训练过程中批量加载和处理数据
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=8)

# 获得一个batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy() # 将4维张量移除第一维，并转换为Numpy数组
batch_y = b_y.numpy() # 将张量转换为Numpy数组
class_label = train_data.classes # 训练集的标签
# print(class_label)

# 可视化一个batch的数据
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()