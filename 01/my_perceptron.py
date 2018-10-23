# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
"""
@author:zhouqiuhong
@file:my_perceptron.py
@time:2018/9/15 001513:29
"""
"""
感知机算法的实现
1.python代码实现；
2.调用sklearn包实现

"""


def createdata():
    samples = np.array([[3, -3], [4, -3], [1, 1], [1, 2]])
    labels = [-1, -1, 1, 1]
    return samples, labels


class MyPerceptron:

    def __init__(self, x, y, n=1):
        self.x = x
        self.y = y
        self.w = np.zeros((x.shape[1], 1))
        self.b = 0
        self.n = n #学习率
        self.numSamples = self.x.shape[0]
        self.numFeatures = self.x.shape[1]

    def sign(self, w, b, x):
        y = np.dot(x, w) + b
        return int(y)

    def update(self, label_a, data_a):
        temp = label_a*self.n*data_a
        temp = temp.reshape(self.w.shape)
        self.w = self.w + temp
        self.b = self.b + label_a*self.n

    def train(self):
        isFind = False
        while not isFind:
            count = 0
            for i in range(self.numSamples):
                tmpY = self.sign(self.w, self.b, self.x[i, :])
                if tmpY*self.y[i] <= 0:
                    print("误分类点：{0}".format(self.x[i, :]), "此时w为%s和b为%s"%(self.w, self.b))
                    count += 1
                    self.update(self.y[i], self.x[i, :])
            if count == 0:
                print("训练后的w和b为", self.w, self.b)
                isFind = True
        return self.w, self.b


class Picture:
    def __init__(self, data, w, b):
        self.b = b
        self.w = w
        plt.figure(1)
        plt.title('Perceptron Learning Algorithm', size=14)
        plt.xlabel('x0-axis', size=14)
        plt.ylabel('x1-axis', size=14)

        xData = np.linspace(0, 5, 100)
        yData = self.expression(xData)
        plt.plot(xData, yData, color='r', label='sample data')

        plt.scatter(data[0][0], data[0][1], s=50)
        plt.scatter(data[1][0], data[1][1], s=50)
        plt.scatter(data[2][0], data[2][1], s=50, marker='x')
        plt.scatter(data[3][0], data[3][1], s=50, marker='x')
        plt.savefig('2d.png', dpi=75)

    def expression(self, x):
        y = (-self.b - self.w[0] * x) / self.w[1]  # 注意在此，把x0，x1当做两个坐标轴，把x1当做自变量，x2为因变量
        return y

    def Show(self):
        plt.show()



if __name__ == "__main__":
    samples, labels = createdata()
    myperceptron = MyPerceptron(x=samples, y=labels)
    weights, bias = myperceptron.train()
    Picture = Picture(samples, weights, bias)
    Picture.Show()








