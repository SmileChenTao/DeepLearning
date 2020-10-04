#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import glob
from tensorflow import keras
from tensorflow.keras import layers

#######自定义训练网络##########

# 获取到每张图片的路径
train_image_path = glob.glob('./train/*/*.jpg')
# 给每张图片制作标签
train_image_label = [int(p.split('\\')[1]=='cat') for p in train_image_path]

def load_preprosess_image(path,label):
    """
    方法用于对图片进行预处理
    :param:
    :return:
    """
    # 读取图片
    image = tf.io.read_file(path)
    # 解码
    image = tf.image.decode_jpeg(image,channels=3)
    # 将图片统一大小
    image = tf.image.resize(image,[256,256])
    # tf.image.convert_image_dtype
    image = tf.cast(image,tf.float32)
    image = image/255
    # [1,2,3] --> [[1],[2],[3]]
    # 把每个样本作为单独的维度放置，不然会误以为这一个列表为一个的数据
    label = tf.reshape(label,[1])
    return image,label

train_image_dataset = tf.data.Dataset.from_tensor_slices((train_image_path,train_image_label))
# 根据CPU的个数 自动设置并行运算
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_image_dataset = train_image_dataset.map(load_preprosess_image,num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 32
train_count = len(train_image_path)
train_image_dataset = train_image_dataset.shuffle(train_count).batch(BATCH_SIZE)
# 在数据进行训练过程中，去后台预读取一些数据，加快训练速度
train_image_dataset = train_image_dataset.prefetch(AUTOTUNE)

# 创建模型
model = keras.Sequential([
    layers.Conv2D(64,(3,3),input_shape=(256,256,3),activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256,(3,3),activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(512,(3,3),activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(1024,(3,3),activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256,activation='relu'),
    # 不作激活
    layers.Dense(1)
])

# 定义损失函数
# 最后一层没有激活 from_logits=True
ls = keras.losses.BinaryCrossentropy(from_logits=True)
# 定义优化器
optimizer = tf.keras.optimizers.Adam()
# 定义
epoch_loss_avg = keras.metrics.Mean('train_loss')
train_accuracy = keras.metrics.BinaryAccuracy()

def train_step(model,images,labels):
    """
    方法用于对于一个批次的数据进行优化
    :param:
    :return:
    """
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = ls(labels,pred)
    grads = t.gradient(loss_step,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    epoch_loss_avg(loss_step)
    train_accuracy(labels,tf.cast(pred>0,tf.int32))

train_loss_results = []
train_acc_results = []

num_epochs = 30

for epoch in range(num_epochs) :
    for imgs_,labels_ in train_image_dataset:
        train_step(model,imgs_,labels_)
        print('.',end='')
    print()
    train_loss_results.append(epoch_loss_avg.result())
    train_acc_results.append(train_accuracy.result())
    print('Epoch:{} loss:{:.3f},accuracy:{:.f}'.format(
        epoch+1,epoch_loss_avg.result(),train_accuracy.result()
    ))
    epoch_loss_avg.reset_states()
    train_accuracy.reset_states()

