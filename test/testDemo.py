#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time

# 时间戳
time.time()

# 结构化时间
# print( time.localtime())
# print(time.gmtime())

# # 将结构化时间转换成时间戳
# print(time.mktime(time.localtime()))

# 将结构化时间转换成字符串时间
# print(time.strftime('%Y-%m-%d %X',time.localtime()))

# 将字符串时间转成结构化时间
# print(time.strptime('2020-07-25 12:38:13','%Y-%m-%d %X'))

# a = eval('[1,2,3]')

import random
# print(random.random())
# print(random.randint(1,3))
# print(random.randrange(1,6))
# print(random.choice([1,2]))
# print(random.sample([1,3,4],2))
# a = [1,3,4]
# random.shuffle(a)
# print(a)

# import json
# a = {'a':20}
# data = json.dumps(a)
# print(type(data))

import re
# a = re.findall('a..x','xaldxlll')
# print(a)
b = re.findall(r'(?:ax)+','axaxsdsdsaxax')
print(b)