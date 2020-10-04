#!/usr/bin/env python
# -*- coding: UTF-8 -*-

###############使用jieba分词对txt文件的小说进行分词###############

import jieba
import jieba.analyse
import re
print('主程序执行开始...')

input_file_name = '../File/凡人修仙之仙界篇.txt'
output_file_name = '../File/凡人修仙之仙界篇分词.txt'
input_file = open(input_file_name, 'r', encoding='utf-8')
output_file = open(output_file_name, 'w', encoding='utf-8')
print('开始读入数据文件...')
lines = input_file.readlines()
print('读取数据文件结束...')

print('分词程序执行开始...')

# 正则表达式 符合汉字开头、汉字结尾、中间全是汉字
cn_reg = '^[\u4e00-\u9fa5]+$'

def stopwordslist(filepath):
    """
    方法用于获取停止词
    :param: filepath
    :return: stopwords
    """
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    """
    方法用于对每一行进行分词
    :param:
    :return:
    """
    # 用jieba对句子进行分词
    sentence_seged = jieba.cut(sentence.strip())
    # 加载中文停止词
    stopwords = stopwordslist('../File/cn_stopwords.txt')
    out_stence = ''
    for word in sentence_seged:
        # 删去停止词
        if word not in stopwords:
            # \t \r \n都是转义字符。baidu空格就是单纯的空格。
            # word不是非中文词
            if word !='\t' and re.search(cn_reg,word)  :
                out_stence += word
                out_stence += " "
    return out_stence

for line in lines:
    if line == '\n':
        continue
    line_seg = seg_sentence(line)
    if line_seg != '':
        output_file.write(line_seg+'\n')

print('分词程序执行结束！')


