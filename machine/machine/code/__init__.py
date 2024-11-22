#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 15:10
# @File    : main.py
# @Author  : lkt


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

a = np.array([1,2,3])
b = pd.array([4,5,6])
print(a)
print(b)
x=np.arange(-5,+5.1,0.1)
y=x**3
plt.clf()
# ax=plt.gca()
# ax.set_aspect('equal',adjustable='box')
plt.plot(x,y,color='blue')
plt.scatter(0,0,color='red',marker='*')
plt.show()
# plt.savefig('test.png')
if __name__ == "__main__":
    run_code = 0
