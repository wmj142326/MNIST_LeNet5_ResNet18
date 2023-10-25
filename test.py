# -*- coding: utf-8 -*-
"""
@Time: 2023/10/19 11:01
@Auth: 陌尘小小
@File: test.py.py
@IDE: PyCharm
@EMAIL: wmj142326@163.com
@Note:
"""
import os
import torch.utils.data
from torch import nn
import torchvision
from network import LeNet5
from resnet import ResNet18

from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt

# 数据转化为Tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# Test数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 显卡设置,将模型数据转移到GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)
model = ResNet18(10).to(device)

model.load_state_dict(torch.load("./model/best_model_res.pth"))

for batchsize, (image, label) in enumerate(test_dataloader):
    image_batch = torchvision.utils.make_grid(image, padding=2)
    plt.imshow(np.transpose(image_batch.numpy(), (1, 2, 0)), vmin=0, vmax=255)
    plt.show()

    with torch.no_grad():
        image, label = image.to(device), label.to(device)
        output = model(image)
        _, pred = torch.max(output, axis=1)
        print(f"batchsize--{batchsize}")
        print("预测结果:", pred)
        print("真值结果:", label)
