# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:31:09 2022

@author: 95287
"""
#通过对比可以发现，基于sklearn库的逻辑回归效果远远优于自己编辑的，可能与参数选取有关

import numpy as np
import pandas as pd
class LogisticRegression1(object):

    def __init__(self, learning_rate=0.1, max_iter=100, seed=None):
        self.seed = seed
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, x, y):
        np.random.seed(self.seed)
        #用高斯分布初始化（可以用0）
        self.w = np.random.normal(loc=0.0, scale=1.0, size=x.shape[1])
        self.b = np.random.normal(loc=0.0, scale=1.0)
        self.x = x
        self.y = y
        for i in range(self.max_iter):
            self._update_step()
            #print('loss: \t{}'.format(self.loss()))
            #print('score: \t{}'.format(self.score()))
            #print('w: \t{}'.format(self.w))
            #print('b: \t{}'.format(self.b))
    #改变sigmoid函数，变成广义线性回归
    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _f(self, x, w, b):
        #定义线性回归
        z = x.dot(w) + b
        return self._sigmoid(z)

    def predict_proba(self, x=None):
        #输出的y为类后验概率估计p(y=1|x)
        if x is None:
            x = self.x
        y_pred = self._f(x, self.w, self.b)
        return y_pred

    def predict(self, x=None):
        if x is None:
            x = self.x
        y_pred_proba = self._f(x, self.w, self.b)
        y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])
        return y_pred

    def score(self, y_true=None, y_pred=None):
        if y_true is None or y_pred is None:
            y_true = self.y
            y_pred = self.predict()
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

    def loss(self, y_true=None, y_pred_proba=None):
        if y_true is None or y_pred_proba is None:
            y_true = self.y
            y_pred_proba = self.predict_proba()
        #采用对数损失
        return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))
#梯度下降法迭代优化，先求梯度
    def _calc_gradient(self):
        y_pred = self.predict()
        d_w = (y_pred - self.y).dot(self.x) / len(self.y)
        d_b = np.mean(y_pred - self.y)
        return d_w, d_b
#基于梯度，进行迭代
    def _update_step(self):
        d_w, d_b = self._calc_gradient()
        self.w = self.w - self.lr * d_w
        self.b = self.b - self.lr * d_b
        return self.w, self.b
# -*- coding: utf-8 -*-

import numpy as np


def train_test_split(x, y):
    split_index = int(len(y)*0.7)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    return x_train, y_train, x_test, y_test

# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
'''
#导入数据集
from sklearn import datasets
cancer=datasets.load_breast_cancer()
x=cancer.data
y=cancer.target
'''
data_file = open('watermelon_3a.csv')
dataset = np.loadtxt(data_file, delimiter=",")

x = dataset[:,1:3]
y = dataset[:,3]
#数据集划分
x_train, y_train, x_test, y_test = train_test_split(x, y)
# 数据标准化
x_train = (x_train - np.min(x_train, axis=0)) / (np.max(x_train, axis=0) - np.min(x_train, axis=0))
x_test = (x_test - np.min(x_test, axis=0)) / (np.max(x_test, axis=0) - np.min(x_test, axis=0))
#自己建立的逻辑回归分类

clf = LogisticRegression1(learning_rate=0.1, max_iter=500, seed=272)
clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)
y_test_pred_proba = clf.predict_proba(x_test)
print(clf.score(y_test, y_test_pred))
print(clf.loss(y_test, y_test_pred_proba))

#导入sklearn中逻辑回归函数实现分类
def score(y_true, y_pred):
    acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
    return acc

def loss(y_true, y_pred_proba):
        #采用对数损失
    return np.mean(-1.0 * (y_true * np.log(y_pred_proba) + (1.0 - y_true) * np.log(1.0 - y_pred_proba)))

from sklearn.linear_model import LogisticRegression
clf1=LogisticRegression()
clf1.fit(x_train, y_train)
y_test_pred = clf1.predict(x_test)
y_test_pred_proba = clf1.predict_proba(x_test)
print(score(y_test, y_test_pred))
print(loss(y_test, y_test_pred_proba[:,1]))
