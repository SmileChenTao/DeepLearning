#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import re
from tensorflow.keras import layers
from tensorflow import keras
from nltk.stem import SnowballStemmer

data_path = '../File/Tweets.csv'
pretreat_data_path = '../File/pretreat_data_tweets.csv'
data = pd.read_csv(data_path)
data = data[['airline_sentiment', 'text']]

# data_positive = data[data.airline_sentiment=='positive']
# data_negative = data[data.airline_sentiment=='negative']
# data_neutral = data[data.airline_sentiment=='negative']
#
# data_negative= data_negative.iloc[:len(data_positive)]
# data_neutral = data_neutral.iloc[:len(data_positive)]
#
# data = pd.concat([data_negative,data_neutral])
# data = pd.concat([data,data_positive])


# print(data.head())
# # ['neutral' 'positive' 'negative']
# print(data['airline_sentiment'].unique())

# negative    9178
# neutral     3099
# positive    2363
# print(data['airline_sentiment'].value_counts())
# 0  negative  1 neutral 2 positive

#############数据预处理##############
def tronsform_review(label):
    """
    方法用于将标签数字化
    :param:label
    :return: int
    """
    if label == 'negative':
        return 0
    elif label == 'neutral':
        return 1
    else:
        return 2


data['review'] = data['airline_sentiment'].apply(tronsform_review)


def get_stem(text):
    """
    方法用于词型还原
    :param:text
    :return:stem
    """
    stem = []
    stemmer = SnowballStemmer("english")  # Choose a language
    for i in text:
        stem.append(stemmer.stem(i))
    return stem


def reg_text(text):
    """
    方法用于删除url，数字，只获取英文单词,最后将所有单词转成小写
    :param:text
    :return:new_text
    """
    token = re.compile('[A-Za-z]+')
    http = r'((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?'
    pattern = re.compile(http)
    new_word = re.sub(pattern, '', text)
    new_text = token.findall(new_word)
    new_text = [word.lower() for word in new_text]
    return new_text


def replace_word(text):
    """
    方法用于缩写词还原
    :param: text
    :return: new_word
    """
    new_word = None
    replacement_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', r'\g<1> will'),
        (r'(\w+)n\'t', r'\g<1> not'),
        (r'(\w+)\'ve', r'\g<1> have'),
        (r'(\w+)\'s', r'\g<1> is'),
        (r'(\w+)\'re', r'\g<1> are'),
        (r'(\w+)\'d', r'\g<1> would')]
    patterns = [(re.compile(pattern), repl) for (pattern, repl) in replacement_patterns]
    for (pattern, repl) in patterns:
        new_word = re.sub(pattern, repl, text)
        text = new_word
    return new_word


def stop_words(text):
    """
    方法用于删除停止词
    :param: text
    :return: new_text
    """
    new_text = []
    stop_words_path = '../File/baidu_stopwords.txt'
    stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]
    for word in text:
        if word not in stopwords:
            new_text.append(word)

    return new_text


def pretreat_data(text):
    """
    方法用于对数据进行预处理
    :param: text
    :return: new_text
    """
    # # 还原缩略词
    new_text = replace_word(text)
    # 删除url，数字，只获取英文单词,最后将所有单词转成小写
    new_text = reg_text(new_text)
    # # 删除停用词
    # new_text = stop_words(new_text)
    # 词型还原
    new_text = get_stem(new_text)
    return new_text


data['text'] = data['text'].apply(pretreat_data)


# 构建词典
word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word)
word_list = list(word_set)
word_dict = dict((word,word_list.index(word)+1) for word in word_list)


def transform_label(text):
    """
    方法用于将单词转换成标签
    :param: text
    :return: label_list
    """
    label_list = []
    for word in text:
        label_list.append(word_dict.get(word,0))
    return label_list


data['text'] = data['text'].apply(transform_label)
max_sequence_length = 30
x_train = keras.preprocessing.sequence.pad_sequences(data['text'].values,maxlen=max_sequence_length)
y_train = data['review']

# data[['review','text']].to_csv(pretreat_data_path)
# # 33
# max_length = max(len(x)for x in data['text'])
# # 9264
max_word_length = len(word_set)+1
embedding = 100

############建立lstm模型
model = keras.Sequential([
    layers.Embedding(max_word_length,embedding,input_length=max_sequence_length),
    layers.LSTM(64,return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(3,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
print(model.summary())
history = model.fit(x_train,y_train,validation_split=0.2,epochs=4)
# print(history.history.keys())
import matplotlib.pyplot as plt
plt.plot(history.epoch,history.history['loss'],label='loss')
plt.plot(history.epoch,history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.plot(history.epoch,history.history['acc'],label='acc')
plt.plot(history.epoch,history.history['val_acc'],label='val_acc')
plt.legend()
plt.show()
