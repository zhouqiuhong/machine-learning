# -*- coding:utf-8 -*-
"""
@author:zhouqiuhong
@file:test_perceptron.py
@time:2018/9/15 001514:13
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
a = np.array([1, 2, 3])
d = np.tile(a, [2])
print(d)
