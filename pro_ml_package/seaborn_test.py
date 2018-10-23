# -*- coding:utf-8 -*-
"""
@author:zhouqiuhong
@file:seaborn_test.py
@time:2018/10/23 002310:48
"""
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

sns.set_style("dark")
x1 = np.random.normal(size=1000)
x2 = np.random.randint(0, 50, 200)
sns.distplot(x1, bins=20, kde=True, rug=True)
sns.kdeplot(x2, shade=True)
sns.rugplot(x2)
x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma)

plt.show()