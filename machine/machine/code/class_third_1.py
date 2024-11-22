#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/22 15:15
# @File    : class_third_1.py
# @Author  : lkt

import numpy as np
import matplotlib.pyplot as plt

def draw(data, label, n=3, m=4):
    fig = plt.figure()
    for i in range(n * m):  # 显示的图像总数是 n * m
        plt.subplot(n, m, i + 1)
        plt.tight_layout()
        plt.imshow(data[i], cmap='gray', interpolation='none')
        plt.title("Labels: {}".format(label[i]))
        plt.xticks([])  # 去除x轴刻度
        plt.yticks([])  # 去除y轴刻度
    plt.show()

def main_draw():
    data = np.load('../data/mnist_npy_new/mnist_npy/train_data.npy').squeeze()  # 加载训练数据
    label = np.load('../data/mnist_npy_new/mnist_npy/train_label.npy')  # 加载标签
    print(data.shape)
    print(label.shape)
    draw(data[:12], np.argmax(label[:12], axis=-1))  # 只显示前12个样本

if __name__ == "__main__":
    main_draw()