#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
TensorFlow中Rnn的基础相关概念
"""
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore", category=Warning)
batchsize = 128
# 10000个单词
total_words = 10000
# 80个单词长度
max_review_len = 80
# 维度
embedding_len = 100
# 对常见的10000个单词进行编码，不常见的标记为一个单词
# 加载IMDB数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
# x_train:[b,80]
# x_test: [b,80]
# 设置句子的长度为80个单词，少的会补充，长的会截取
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsize, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsize, drop_remainder=True)

print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)
print('db_train shape',db_train)

# 继承父类Keras.Model
class MyRnn(keras.Model):
    def __init__(self, units):
        super(MyRnn, self).__init__()
        # [b,64]
        self.state0 = [tf.zeros([batchsize, units])]
        # 输入层
        # transform text to embedding representation
        # [b,80] => [b,80,100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        # [b,80,100] , h_dim:64
        # RNN: cell1 , cell2 , cell3
        # SimpleRNN
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        # 输出层  二分类的结果
        # fc,[b,80,100] => [b,64] => [b,1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:[b,80]
        :param training:
        :param mask:
        :return:
        """
        x = inputs
        # embedding: [b,80] => [b,80,100]
        x = self.embedding(x)
        # rnn cell compute
        # [b,80,100] => [b,64]
        state0 = self.state0
        for word in tf.unstack(x, axis=1):  # word: [b,100]
            # h1 = X*Wxh+h*Whh
            out, state1 = self.rnn_cell0(word, state0, training)
            state0 = state1

        # out: [b,64] => [b,1]
        x = self.outlayer(out)
        # p(y is pos /x)
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epoches = 2

    model = MyRnn(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epoches, validation_data=db_test)
    ## 评估模型
    model.evaluate(db_test)


if __name__ == '__main__':
    main()
