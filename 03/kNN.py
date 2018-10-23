# -*- coding:utf-8 -*-
"""
@author:zhouqiuhong
@file:kNN.py
@time:2018/9/15 001515:02
"""
import numpy as np
import operator


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2

