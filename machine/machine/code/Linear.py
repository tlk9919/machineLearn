#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 16:41
# @File    : Linear.py
# @Author  : lkt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
def OLS(x,y):
    x_avg=x.mean()
    n=x.shape[0]
    A=(y*(x-x_avg)).sum()
    B=(x*x).sum()-1/n*(x.sum())**2
    w=A/B
    b=1/n*(y-w*x).sum()
    return w,b
df =pd.read_csv('\machine_code\pythonProject\machine\class1\Linear1.csv',header=None).to_numpy()
# 用 pandas 库读取名为 Linear1.csv 的 CSV 文件，并将其转换为 NumPy 数组。 header=None 表示 CSV 文件没有标题行。
x=df[:,0]
# 将 df 数组的第一列（索引为 0 的列）赋值给 x
y=df[:,1]
print(x,y)
w,b=OLS(x,y)
print(w,b)
plt.clf()
ax=plt.gca()
# 获取当前的坐标轴对象，方便进行后续的设置。
ax.set_aspect('equal',adjustable='box')
# 设置坐标轴的纵横比为 1:1，即 x 轴和 y 轴的单位长度在视觉上保持一致
plt.scatter(x,y,color='red')
plt.plot(x,w*x+b,color='blue')
plt.show()
if __name__ == "__main__":
    run_code = 0
