#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import imp
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target
# 将百分之七十划分为训练集，百分之三十划分为测试集
# 105：45
x_train ,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=42)
# 选择weights='distance' 按距离的倒数计算权重点。在这种情况下，查询点的近邻会比距离较远的邻居影响更大
knn = neighbors.KNeighborsClassifier(n_neighbors=15,
                                     weights='distance')
knn.fit(x_train, y_train)
# y_predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
score = knn.score(x_test,y_test)
print("knn得分为：{}".format(score))






