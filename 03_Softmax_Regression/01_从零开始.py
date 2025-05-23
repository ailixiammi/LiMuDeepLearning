import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision import transforms

""" 读取数据集 """
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./data',
                                                train=True,
                                                transform=trans,
                                                download=True)

mnist_test = torchvision.datasets.FashionMNIST(root='./data',
                                               train=False,
                                               transform=trans,
                                               download=True)
# print(len(mnist_train))
# print(len(mnist_test))
# print(mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=4)