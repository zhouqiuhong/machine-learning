# -*- coding:utf-8 -*-
"""
@author:zhouqiuhong
@file:matplot_test.py
@time:2018/10/22 002215:53
"""
import matplotlib.pyplot as plt
import numpy as np

data = {
    "a": np.arange(50),
    "c": np.random.randint(0, 50, 50),
    "d": np.arange(50)

}
data['b'] = data['a'] + 10*np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter("a", "b", c="c", s="d", data=data)
plt.xlabel("entry a")
plt.ylabel("entry b")
plt.show()