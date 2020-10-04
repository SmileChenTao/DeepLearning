#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers

# 1.导入tf.keras
# tensorflow2推荐使用keras构建网络，常见的神经网络都包含在keras.layer中
print(tf.keras.__version__)
print(tf.__version__)

# 2.构建简单模型
# 2.1 模型堆叠
# 最常见的模型类型是层的堆叠：tf.keras.Sequential 模型
model = tf.keras.Sequential()

# 2.2网络配置
# tf.keras.layers中网络配置：
# activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
# kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。
# kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

# 3.训练和评估
# 构建模型后，调用compile方法配置该模型的学习流程：
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss = tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

# 3.2 输入Numpy数据
import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))

model.fit(train_x,train_y,batch_size=100,epochs=10,
          validation_data=(val_x,val_y))

# 3.3 tf.data 输入数据
# dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
# dataset = dataset.batch(32)
# dataset = dataset.repeat()
# val_dataset = tf.data.Dataset.from_tensor_slices((val_x,val_y))
# val_dataset = val_dataset.batch(32)
# val_dataset = val_dataset.repeat()
#
# model.fit(dataset,epochs=10,steps_per_epoch=30,
#           validation_data=val_dataset,validation_steps=3)

# 3.4 评估与预测
test_x = np.random.random((1000,72))
test_y = np.random.random((1000,10))
model.evaluate(test_x,test_y,batch_size=32)
result = model.predict(test_x,batch_size=32)
print(result)

