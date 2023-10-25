# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/19 下午4:26
@Auth ： 陌尘小小
@File ：demo.py.py
@EMAIL ：wmj142326@163.com
@Note：
"""

import torch.utils.data
from network import LeNet5
from resnet import ResNet18
from PIL import Image
from torchvision import transforms
import numpy as np

# 显卡设置,将模型数据转移到GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet5().to(device)
model = ResNet18(10).to(device)

model.load_state_dict(torch.load("./model/best_model_res.pth"))


# 读取图像与数据处理
def get_image(image_path, threshold=165):
    image = Image.open(image_path)

    # 将图像转换为灰度图
    image = image.convert("L")

    # 将灰度图转换为二值图像
    image = image.point(lambda p: p > threshold and 255)
    image = image.resize((28, 28))
    # 将图像转换为numpy数组
    image_array = np.array(image)
    # 互换0和1
    flipped_image_array = 255 - image_array

    # 将numpy数组转换回图像
    # flipped_image = Image.fromarray(flipped_image_array)
    # 保存二值图像
    # binary_image_path = f"{image_path.split('.')[0]}.png"
    # flipped_image.save(binary_image_path)

    image_tensor = transforms.ToTensor()(flipped_image_array)
    image_tensor[image_tensor != 0] = 1
    return image_tensor


with torch.no_grad():
    image_path = "5.jpg"
    image = get_image(image_path, threshold=190)  # 165,190
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    _, pred = torch.max(output, axis=1)
    print("图片数字:", image_path.split('.')[0],
          "||",
          "预测结果:", int(pred))
