# -*- coding: utf-8 -*-
"""
@Time: 2023/10/18 15:28
@Auth: 陌尘小小
@File: network.py
@IDE: PyCharm
@EMAIL: wmj142326@163.com
@Note:
"""

import torch
from torchstat import stat
from torch import nn


# 定义网络模型
class LeNet5(nn.Module):
    # 初始化网络
    def __init__(self):
        super(LeNet5, self).__init__()
        # 定义卷积层，ReLU激活函数，平坦层和全连接层
        # conv2d的输入通道为1维，输出为6维，卷积核尺寸为5*5，步长为1，不适用padding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 构建Lenet5架构，x代表网络的输入
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    x = torch.rand(1, 1, 28, 28)
    model = LeNet5()
    y = model(x)
    print(y)

    stat(model, (1, 28, 28))


