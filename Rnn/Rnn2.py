#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import  keras
import tensorflow_hub as hub
import warnings
warnings.filterwarnings("ignore", category=Warning)

total_words = 10000
max_review_len = 80
batchsize = 128
embedding = 100

(x_train,y_train),(x_test,y_test) = keras.datasets.imdb.load_data(num_words=10000)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

db_train = db_train.shuffle(1000).batch(batchsize, drop_remainder=True)


db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsize, drop_remainder=True)


model = keras.Sequential([
    # [b,80] => [b,80,100]
    layers.Embedding(total_words,embedding,input_length=max_review_len),
    layers.SimpleRNN(64),
    layers.Dense(1,activation='sigmoid')
])
print(model.summary())
model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['acc'])
model.fit(db_train, validation_data=db_test,batch_size=128)

