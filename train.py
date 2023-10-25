# -*- coding: utf-8 -*-
"""
@Time: 2023/10/18 16:05
@Auth: 陌尘小小
@File: train.py
@IDE: PyCharm
@EMAIL: wmj142326@163.com
@Note:
"""
import os
import time
import torch.utils.data
from torch import nn
from network import LeNet5
from resnet import ResNet18
from torchvision import transforms, datasets
from torch.optim import lr_scheduler

# 数据转化为Tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载Train和Test数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

val_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)

print(train_dataset)
print(val_dataset)

# 显卡设置,将模型数据转移到GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = LeNet5().to(device)
model = ResNet18(10).to(device)

# 定义损失函数(交叉熵损失)
loss_fun = nn.CrossEntropyLoss()
# 定义优化器
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 定义学习率变化
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, model, loss_fun, optimizer):
    loss, acc, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        cur_loss = loss_fun(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        acc += cur_acc.item()
        n = n + 1
    return loss / n, acc / n


# 定义验证函数
def val(dataloader, model, loss_fun):
    model.eval()
    loss, acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fun(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()
            acc += cur_acc.item()
            n = n + 1

    return loss / n, acc / n


# 训练
time_start = time.time()
epochs = 50
max_acc = 0
for epoch in range(epochs + 1):
    train_loss, train_acc = train(train_dataloader, model, loss_fun, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fun)

    # 保存best训练权重
    if val_acc > max_acc:
        os.makedirs("./model", exist_ok=True)
        max_acc = val_acc
        torch.save(model.state_dict(), "model/best_model_res.pth")
        print(f"---epoch{epoch}---:",
              "train_loss:{:.6f}".format(train_loss), "train_acc:{:.6f}".format(train_acc), "|",
              "val_loss:{:.6f}".format(val_loss), "val_acc:{:.6f}".format(val_acc), "|",
              "......"
              f"save best model in epoch--{epoch}"
              )
    if epoch == epochs:
        torch.save(model.state_dict(), "model/last_model_res.pth")
        print(f"---epoch{epoch}---:",
              "train_loss:{:.6f}".format(train_loss), "train_acc:{:.6f}".format(train_acc), "|",
              "val_loss:{:.6f}".format(val_loss), "val_acc:{:.6f}".format(val_acc), "|",
              "......"
              f"save last model in epoch--{epoch}"
              )
    if val_acc <= max_acc:
        print(f"---epoch{epoch}---:",
              "train_loss:{:.6f}".format(train_loss), "train_acc:{:.6f}".format(train_acc), "|",
              "val_loss:{:.6f}".format(val_loss), "val_acc:{:.6f}".format(val_acc), "|"
              )
time_end = time.time()

time = time_end - time_start

print("loss time：", time)
