#!/usr/bin/env python
# -*- coding: UTF-8 -*-

###利用NumPy进行历史股价分析
import sys
import numpy as np
# 读取文件
# usecol=第7行存储到C，第8行存储到V
# c,v= np.loadtxt('../File/data.csv',delimiter=',',usecols=(6,7),unpack=True)
# # print(c)
# import imp
# import nltk
# nltk.download('punkt')
# sentence = "Hello,world"
# tokens = nltk.word_tokenize(sentence)
# print(tokens)
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
y = np.concatenate((a,b))
print(y)
