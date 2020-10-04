#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from tensorflow.keras import layers
import tensorflow as tf
class Encoder(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dim,enc_units,batch_sz):
        super(Encoder,self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)

    def call(self, inputs,x=None, training=None, mask=None):
        print(inputs)

encoder = Encoder()


