# -*- coding:utf-8 -*-
"""
@author:zhouqiuhong
@file:sklearn_test_1.py
@time:2018/10/23 00239:14
"""
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# plt.scatter(X_train, y_train)
knn = KNeighborsClassifier(algorithm="kd_tree")
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)
print(knn.score(X_test, y_test))
# plt.show()
