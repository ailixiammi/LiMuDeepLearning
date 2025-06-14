import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    """ 构造函数：定义所有的工具（层、激活函数等） """
    def __init__(self):
        super(LeNet, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        """ 前向传播 """
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)

        return x

""" 主函数，测试模型是否有问题 """
if __name__ == "__main__":
    device = torch.device("cuda")

    # 模型实例化，并放到设备里面
    model = LeNet().to(device)

    # 获取模型参数，输入数据的大小为(1, 28, 28)
    parameters = summary(model, (1, 28, 28))
    print(parameters)