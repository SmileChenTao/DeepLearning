#!/usr/bin/env python
# -*- coding: UTF-8 -*-
##########tf.keras 实现线性回归

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import pandas as pd

data_path = '../File/Income1.csv'
data = pd.read_csv(data_path)
# print(data)
x = data.Education
y = data.Income
# 序列模型
model = tf.keras.Sequential()
# 添加层 输出单元数为1 输入维度为1
# 完成 ax+b
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
# 输出模型内容
# print(model.summary())
model.compile(optimizer="adam",
              loss='mse')
# epochs所有数据训练10次
history = model.fit(x,y,epochs=10)


