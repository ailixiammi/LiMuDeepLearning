import os
import pandas as pd
import torch
from sympy.codegen.ast import float32

""" 创建.csv数据集 """

# 创建目录
os.makedirs(os.path.join('.', 'data'), exist_ok=True)

# 定义文件路径
data_file = os.path.join('.', 'data', 'house_tiny.csv')

# 写入数据
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

""" 读取数据 """
data = pd.read_csv(data_file)
print(data)

""" 处理缺失值 """
'''
为了处理缺失的数据，典型的方法包括插值法和删除法，
其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值
'''
# 转换数据类型
data['NumRooms'] = pd.to_numeric(data['NumRooms'], errors='coerce')
data['Alley'] = pd.to_numeric(data['Alley'], errors='coerce')

# 平均替代
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

""" 转换为张量 """
x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x)
print(y)