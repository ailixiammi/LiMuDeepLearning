import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import ResNet18,Residual

def test_data_process():
    """ 处理测试集 """
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))

    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=8)

    return test_dataloader

def test_model_process(model, test_dataloader):
    """ 模型测试 """

    """ 将模型放入设备 """
    device = "cuda"
    model = model.to(device)

    """ 初始化参数 """
    # 测试精度
    test_corrects = 0.0
    # 测试样本数量
    test_num = 0

    """ 模型测试 """
    # 只进行前向传播，不进行梯度计算
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征和标签放入到设备里面
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 设置模型为评估模式
            model.eval()

            # 前向传播：输入为一个batch，输出为一个batch的预测结果
            output = model(test_data_x)  # output 的形状将是 [batch_size, num_classes]
            # 查找每一行中最大概率值所对应的索引
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则精确度test_corrects+1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 累加所有测试样本
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试准确率为：", test_acc)

if __name__ == "__main__":
    # 加载模型
    model = ResNet18(Residual)

    # 加载模型参数
    model.load_state_dict(torch.load("best_model.pth"))

    # 加载测试集
    test_dataloader = test_data_process()

    # 模型测试
    test_model_process(model, test_dataloader)

"""
    # 设置测试设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值：", result, "------", "真实值：", label)
"""
