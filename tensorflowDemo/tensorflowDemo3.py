#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_path = '../File/Advertising.csv'

data = pd.read_csv(data_path)

x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

model = keras.Sequential(
    [layers.Dense(10,input_shape=(3,),activation='relu'),
     layers.Dense(1)
     ]
)
print(model.summary())
model.compile(optimizer='adam',
              loss='mse')
model.fit(x,y,epochs=100)
