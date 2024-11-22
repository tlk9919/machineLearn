#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 17:36
# @File    : gradient.py
# @Author  : lkt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
def GD(x, y, rendering=False):
    w = 0
    b = 0
    loss = []
    def f(_w, _b, _x):
        return _w * _x +_b
    def MSE(y_gd, y_f):
        return ((y_gd - y_f)**2).mean()
    alpha = 0.05
    T = int(1000)
    for _ in range(T):
        # 求梯度
        partial_w = -(y - f(w, b, x)) * 2 * x
        partial_b = -(y - f(w, b, x)) * 2
        # 梯度下降
        w -= alpha * partial_w.mean()
        b -= alpha * partial_b.mean()
        loss.append(MSE(y, f(w, b, x)))
    if rendering == True:
        plt.plot(loss)
        plt.savefig('1.png')
    return w, b

if __name__ == "__main__":
    run_code = 0
