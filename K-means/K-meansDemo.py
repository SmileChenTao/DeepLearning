#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import imp
from sklearn import datasets
from sklearn.model_selection import  train_test_split
# 加载数据
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 将数据集划分为训练集和测试集  百分之40作为测试集
x_train ,x_test,y_train,y_test= train_test_split(x,y,test_size=0.4,random_state=42)
# print(x_train.shape)