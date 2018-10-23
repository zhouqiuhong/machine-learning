# -*- coding:utf-8 -*-
"""
@author:zhouqiuhong
@file:sklearn_test_2.py
@time:2018/10/23 00239:36
"""
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
boston_X = boston.data
boston_y = boston.target

X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.25)
line = LinearRegression()
line.fit(X_train, y_train)
print(line.predict(X_test))
print(y_test)
print(line.score(X_test, y_test))