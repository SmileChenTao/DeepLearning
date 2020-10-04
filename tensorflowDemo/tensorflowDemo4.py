#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
###########自动微分运算############
# w = tf.constant(3.0)
# with tf.GradientTape() as t:
#     # 跟踪常量
#     t.watch(w)
#     loss = w*w
# dloss_dw = t.gradient(loss,w)
# # 6.0
# print(dloss_dw)
#
# v = tf.Variable(1.0)
# # 记录运算过程
# with tf.GradientTape() as t:
#     loss = v*v
# # 对v求导
# dloss_dv = t.gradient(loss,v)
# print(dloss_dv)

# w = tf.constant(3.0)
# with tf.GradientTape(persistent=True) as t:
#     t.watch(w)
#     y = w*w
#     z = y*y
# value = t.gradient(z,w)
# print(value)

#########自定义训练############
(train_image,train_labels),(test_image,test_labels) =tf.keras.datasets.mnist.load_data()
# 扩充维度
train_image = tf.expand_dims(train_image,-1)
test_image = tf.expand_dims(test_image,-1)

# print(train_image.shape)
# 归一化并转换数据类型
train_image = tf.cast(train_image/255,tf.float32)
train_labels = tf.cast(train_labels,tf.int64)
test_image = tf.cast(test_image/255,tf.float32)
test_labels = tf.cast(test_labels,tf.int64)

dataset = tf.data.Dataset.from_tensor_slices((train_image,train_labels))
dataset = dataset.shuffle(10000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_image,test_labels))
test_dataset = test_dataset.batch(32)

model = keras.Sequential()
model.add(layers.Conv2D(16,[3,3],activation='relu',input_shape=(None,None,1)))
model.add(layers.Conv2D(32,[3,3],activation='relu'))
model.add(layers.GlobalAveragePooling2D())
# 没有激活
model.add(layers.Dense(10))

optimizer = tf.keras.optimizers.Adam()
# 可调用的方法

loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# features,labels = next(iter(dataset))
# (32, 28, 28, 1)
# print(features.shape)
# predictions = model(features)
# (32, 10)
# print(predictions.shape)
# tf.argmax( predictions,axis=1)

def loss(model,x,y):
    """
    方法用于计算损失函数
    :param:
    :return:
    """
    y_ = model(x)
    return loss_func(y,y_)

# 创建计算损失函数和准确率的类
train_loss = tf.keras.metrics.Mean('train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

def train_step(model,images,labels):
    with tf.GradientTape() as t:
        pred = model(images)
        # 计算的是一个批次（32）的损失值
        loss_step = loss_func(labels,pred)
        # loss_step = loss(model,images,labels)
    # 优化损失函数
    grads = t.gradient(loss_step,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    # 计算每一个批次的损失函数和准确率
    train_loss(loss_step)
    train_accuracy(labels,pred)

def test_step(model,images,labels):
    pred = model(images)
    # 计算的是一个批次（32）的损失值
    loss_step = loss_func(labels,pred)
    test_loss(loss_step)
    test_accuracy(labels,pred)

def train():
    """
    方法用于
    :param:
    :return:
    """
    for epoch in range(10):
        for (batch,(images,labels)) in enumerate(dataset):
            train_step(model,images,labels)
        # 训练一个epoch所需要的损失值和准确率
        print('Epoch{} loss is {},accuracy is {}'.format(epoch,train_loss.result(),train_accuracy.result()))
        for (batch,(images,labels)) in enumerate(test_dataset):
            test_step(model,images,labels)
        # 训练一个epoch所需要的损失值和准确率
        print('Epoch{} loss is {},accuracy is {}'.format(epoch,test_loss.result(),test_accuracy.result()))
        test_loss.reset_states()
        test_accuracy.reset_states()
        train_loss.reset_states()
        train_accuracy.reset_states()

train()



