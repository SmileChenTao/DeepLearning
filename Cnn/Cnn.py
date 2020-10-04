#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt

data_dir = './2_class'
data_root = pathlib.Path(data_dir)
for item in data_root.iterdir():
    print(item)

all_image_path = list(data_root.glob('*/*'))
all_image_path = [str(path) for path in all_image_path]

import random
random.shuffle(all_image_path)
image_count = len(all_image_path)
label_names = sorted(item.name for item in data_root.glob('*/'))
# 给标签['airplane', 'lake'] 编码 0，1
label_to_index = dict((name,index) for index ,name in enumerate(label_names))
# 给每个图片分类
all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]

def load_preprocess_image(img_path):
    """
    方法用于图片预处理
    :param:
    :return:
    """
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3)
    img_tensor = tf.image.resize(img_tensor,[256,256])
    img_tensor = tf.cast(img_tensor,tf.float32)
    img = img_tensor/255
    return img

image_path = all_image_path[100]

# plt.imshow(load_preprocess_image(image_path))
# plt.show()
path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
image_dataset = path_ds.map(load_preprocess_image)
label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
dataset = tf.data.Dataset.zip((image_dataset,label_dataset))
test_count = int(image_count*0.2)
train_count = image_count - test_count
train_dataset = dataset.skip(test_count)
test_dataset = dataset.take(test_count)
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(train_count).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)