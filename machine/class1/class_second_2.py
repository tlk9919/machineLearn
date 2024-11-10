#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/1 15:16
# @File    : class_second_2.py
# @Author  : lkt

import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


def  kmeans(x, k, eps=1e-2):
    n = x.shape[0]  # 样本数量
    label = np.zeros(n, dtype=int)  # 初始化标签
    center = x[np.random.choice(n, k, replace=False)]  # 随机选择初始中心
    size_of_cluster = np.zeros(k)  # 初始化每个簇的大小

    while True:
        size_of_cluster.fill(0)  # 重置每个簇的大小
        new_center = np.zeros((k, x.shape[1]))  # 初始化新的中心
        dist = np.zeros(k)  # 存储距离

        for i, p in enumerate(x):  # 遍历每个样本
            for j in range(k):  # 计算到每个中心的距离
                dist[j] = norm(p - center[j])
            label[i] = np.argmin(dist)  # 将样本分配到最近的中心
            new_center[label[i]] += p  # 更新新的中心
            size_of_cluster[label[i]] += 1  # 更新簇的大小

        for j in range(k):  # 计算新的中心
            if size_of_cluster[j] > 0:  # 避免除以零
                new_center[j] /= size_of_cluster[j]

        is_changed = False  # 标记中心是否变化
        for j in range(k):  # 检查中心是否变化
            if norm(new_center[j] - center[j]) > eps:
                is_changed = True
                break

        center = new_center  # 更新中心
        if  is_changed==False:  # 如果中心不再变化，退出循环
            break

    return center, label  # 返回中心和标签
# 随机种子
np.random.seed(12)

# 读取数据
df = pd.read_csv('\machine_code\pythonProject\machine\class1\kmeans.csv', header=None)
x = df.to_numpy()

k = 4
center, label = kmeans(x, k)

# 画图
plt.clf()
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

cmap2 = plt.cm.get_cmap('tab20', k)
color = [cmap2(i) for i in range(k)]

# 绘制数据点
for i in range(x.shape[0]):  # 修正了形状索引
    plt.scatter(x[i][0], x[i][1], color=color[label[i]])

# 绘制聚类中心
for i in range(k):
    plt.scatter(center[i][0], center[i][1], color=color[i], marker='*')

plt.show()

if __name__ == "__main__":
    run_code = 0
