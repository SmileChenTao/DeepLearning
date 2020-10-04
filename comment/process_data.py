#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import jieba
import jieba.analyse
# 读取数据集
data_path = '../File/train.csv'
data = pd.read_csv(data_path)

# 将标签和评论划分开
data[['y_label','x_comment']] = data['label\tcomment'].str.split('\\t',expand = True)

def stop_words_list(filepath):
    """
    方法用于获取停止词
    :param: filepath
    :return: stopwords
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def process_data(sentence):
    """
    方法用于分词，去除非中文词
    :param:
    :return:
    """
    # 正则表达式 符合汉字开头、汉字结尾、中间全是汉字
    cn_reg = '^[\u4e00-\u9fa5]+$'
    sentence_seg = jieba.lcut(sentence.strip())
    # for i in sentence_seg:


    return sentence_seg

data['x_comment1'] = data['x_comment'].apply(process_data)
print(data['x_comment_label'].head())



