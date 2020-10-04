#!/usr/bin/env python
# -*- coding: UTF-8 -*-

################使用word2vec 训练词向量################
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

input_file_name = '../File/凡人修仙之仙界篇分词.txt'
model_file_name = '../File/fiction_model.model'
word2vec_file_name = '../File/fiction_word2vec.txt'
print('开始训练...')
model = Word2Vec(LineSentence(input_file_name),
                 size=300, # 词向量的维度
                 window=5,#词向量上下文最大距离
                 iter=8, # 随机梯度下降法中迭代的最大次数
                 min_count=5,# 计算词向量的最小词频
                 workers=multiprocessing.cpu_count(),
                 sg=1 # 采用skip-gram模型
                 )
print('训练结束...')

print('保存模型...')
model.save(model_file_name)
print('保存模型结束...')

# 加载词向量
# import gensim
# word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_path,binary=True)
print('保存词向量')
model.wv.save_word2vec_format(word2vec_file_name)
print('保存词向量结束...')



