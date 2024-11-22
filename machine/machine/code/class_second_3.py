#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/1 16:55
# @File    : class_second_3.py
# @Author  : lkt
import numpy as np
import torch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from matplotlib import pyplot as plt


class MLP(nn.Module):  # 定义一个多层感知机 (MLP) 类，继承自 nn.Module
    def __init__(self) -> None:  # 初始化方法
        super().__init__()  # 调用父类的初始化方法
        # 定义一个顺序容器，包含多个线性层和激活函数
        self.fc = nn.Sequential(
            nn.Linear(1, 20),  # 输入层到隐藏层，1个输入，20个输出
            nn.LeakyReLU(),    # 使用 Leaky ReLU 激活函数
            nn.Linear(20, 20), # 隐藏层到隐藏层，20个输入，20个输出
            nn.LeakyReLU(),    # 使用 Leaky ReLU 激活函数
            nn.Linear(20, 1),  # 隐藏层到输出层，20个输入，1个输出
            nn.ReLU()          # 使用 ReLU 激活函数
        )

    def forward(self, x):  # 前向传播方法
        return self.fc(x)  # 将输入 x 传递通过全连接层

network = MLP()  # 创建 MLP 类的一个实例


def train():
    eps = 1e-2  # 小的增量，用于生成数据
    data_x = np.arange(-3, 3 + eps, eps)  # 生成从 -3 到 3 的数据点
    data_y = data_x ** 2 * 0.5  # 生成对应的标签，这里是 x 的平方的一半
    data_x = torch.FloatTensor(data_x).unsqueeze(-1)  # 将数据转换为 FloatTensor，并增加一个维度
    data_y = torch.FloatTensor(data_y).unsqueeze(-1)  # 将标签转换为 FloatTensor，并增加一个维度

    iterations = int(2e4)  # 设置训练的迭代次数
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)  # 使用 Adam 优化器，学习率为 1e-4
    loss_function = nn.MSELoss()  # 定义均方误差损失函数

    for t in range(iterations):  # 进行多次迭代
        network_output = network(data_x)  # 获取网络输出
        loss = loss_function(network_output, data_y)  # 计算损失
        loss.backward()  # 反向传播，计算梯度

        optimizer.step()  # 根据梯度调用优化器进行参数更新

        if t % 100 == 0:  # 每 100 次迭代打印一次损失
            print(t, loss.item())  # 打印当前迭代次数和损失值

    torch.save(network.state_dict(), 'square.pth')  # 保存训练好的模型参数


def draw():
    # 加载训练好的网络参数
    network.load_state_dict(torch.load('square.pth', weights_only=True))

    # 设置绘图的精度
    draw_eps = 1e-3

    # 创建绘图所需的 x 轴数据
    draw_x = np.arange(-3, 3 + draw_eps, draw_eps)  # 注意：此处确保包括 3
    draw_y1 = draw_x ** 2 * 0.5  # 计算 y = 0.5 * x^2 的真实值

    # 将 x 数据转换为张量，并增加一个维度
    data_for_drawing = torch.FloatTensor(draw_x).unsqueeze(-1)

    # 使用神经网络预测 y 值，并转换为 NumPy 数组
    draw_y2 = network(data_for_drawing).squeeze().detach().numpy()

    # 获取当前的 Axes 对象
    ax = plt.gca()

    # 设置坐标轴的比例为相等
    ax.set_aspect('equal', adjustable='box')

    # 绘制真实的 y 值
    ax.plot(draw_x, draw_y1, label='True Function: $0.5x^2$')

    # 绘制网络预测的 y 值
    ax.plot(draw_x, draw_y2, label='Predicted Function')

    # 保存绘制的图像为 PNG 文件
    plt.savefig('square.png')

    # 显示绘制的图像
    plt.show()


if __name__ == "__main__":
    train()
    run_code = 0
