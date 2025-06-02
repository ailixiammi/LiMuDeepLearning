import copy
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import model
from model import ResNet18,Residual

def train_val_data_process():
    """ 划分训练集和验证集，并配置数据加载器 """
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=8)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    """ 模型训练函数 """

    """ 设定训练设备，优化器，损失函数 """
    device = torch.device("cuda")
    # 使用Adam优化器， 学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    """ 初始化参数 """
    # 最高准确度
    best_acc = 0.0
    # 训练集损失值列表
    train_loss_all = []
    # 验证集损失值列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 保存模型开始训练时的时间
    since = time.time()

    """ 模型训练 """
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 20)

        # 初始化参数
        # 本轮开始的时间
        epoch_start_time = time.time()
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch进行训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播：输入为一个batch，输出为一个batch的预测结果
            output = model(b_x) # 结果格式为一个向量，里面对应着每个结果的可能性概率，和为1
            # 查找每一行中最大概率值所对应的索引
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个batch的loss值
            loss = criterion(output, b_y)

            # 梯度初始化为0,防止梯度累积
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则精确度train_corrects+1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        # 模型验证
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()

            # 前向传播：输入为一个batch，输出为一个batch的预测结果
            output = model(b_x) # output 的形状将是 [batch_size, num_classes]
            # 查找每一行中最大概率值所对应的索引
            pre_lab = torch.argmax(output, dim=1)

            # 计算每一个batch的loss值
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则精确度val_corrects+1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一轮次的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)
        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最优模型
        if val_acc_all[-1] > best_acc:
            # 保存最高准确度
            best_acc = val_acc_all[-1]
            # 保存最佳模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算模型完成该轮训练的耗时
        time_use = time.time() - since
        time_epoch = time.time() - epoch_start_time
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))
        print("本轮训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_epoch//60, time_epoch%60))

    # 选择最优参数，保存最优参数的模型
    torch.save(best_model_wts, "/home/ailixia/PycharmProjects/DeepLearning/05_ResNet/best_model.pth")

    # 数据转换为DataFrame格式
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all, })

    return train_process

def matplot_acc_loss(train_process):
    """ 显示每一次迭代后的训练集和验证集的损失函数和准确率 """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # 加载需要的模型
    model_Net = ResNet18(Residual)
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(model_Net, train_data, val_data, num_epochs=20)
    # 绘图
    matplot_acc_loss(train_process)