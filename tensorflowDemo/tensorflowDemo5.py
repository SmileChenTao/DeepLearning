#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 将numpy转换成tensor
a = tf.convert_to_tensor(np.ones([2,3]))
# 创建全部为1的tensor
b = tf.ones([2,2])
# 创建shape为2，2的全部用0填充的tensor
tf.fill([2,2],1)
# 创建形状为b的全部为0的tensor
c = tf.zeros_like(b)
# 正态分布
tf.random.normal([2,2],mean=1,stddev=1)
# 有裁剪的正态分布
tf.random.truncated_normal([2,2],mean=0,stddev=1)
# 均匀分布
tf.random.uniform([2,2],minval=0,maxval=1)
# 自定义切片
x = tf.ones([4,35,8])
m = tf.gather(x,axis=0,indices=[2,1,3,0]).shape
print(m)