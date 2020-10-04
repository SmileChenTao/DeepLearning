#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np

"""
# numpy 一维数组
a = np.arange(5)
# 输出数组类型
print(type(a))
# 数组数据类型
print(a.dtype)
# 输出数组形状
print(a.shape)
"""

"""
# 创建多维数组
b = np.array([[1,2,4],[4,5,6]])
m = np.array([np.arange(3),np.arange(3)])
print(b)
print(m)

# 生成全是0的数组
# print(np.zeros(5))
# print(np.zeros([3,6]))
# 创建新数组，只分配内存空间，不填充任何值
# print(np.empty(4))
# 生成全是1的数组
# print(np.ones(4))
"""

"""
# 转换数据类型
# arr = np.array([1,2,3,4])
# print(arr.dtype)
# float_arr = arr.astype(np.float64)
# print(float_arr.dtype)
# item = np.array([("b",20),("c",10)])
# print(item.dtype)

# # 数组与标量的运算
# arr = np.array([1,2,3])
# print(arr*2)
# print(1/arr)

# 一维数组的索引与切片
# a = np.arange(9)
# [0 1 2 3 4 5 6 7 8]
# print(a)
# 从3到6 左闭右开
# print(a[3:7])
# 从0到6 步长为2
# print(a[:7:2])
# -1为最后一个 从0到最后一个之前
# [0 1 2 3 4 5 6 7]
# print(a[:-1])
# 逆序输出，从最后一个到3  [8 7 6 5 4 3]
# print(a[:2:-1])
"""

"""
# 多维数组的索引与切片
# 2行3列
# y = np.arange(6).reshape(2,3)
# print(y)

# 2个3行4列(看成2层楼，每层3行4列)
# b = np.arange(24).reshape(2,3,4)
# print(b)
# print(b[0,:,:])
# print(b[0,...])
# print(b[0,1])
# print(b[:,2,1])
# print(b[...,1])
# 选2层 第1楼所有
# print(b[:,1])
# print(b[0,:,-1])
# print(b[0,::-1,-1])

# 数组转置
# arr = np.arange(15).reshape(3,5)
# #  转置1
# # print(arr.T)
# #  转置2
# # print(arr.transpose())
# # 将多维数组转换成一维数组
# print(arr.ravel())
# # 将多维数组转换成一维数组
# arr2 = arr.flatten()
# print(arr2)
"""

"""
# 组合数组
# a = np.arange(9).reshape(3, 3)
# b = 2 * a
# # 横向拼接数组
# print(np.hstack((a,b)))
# # 横向拼接数组
# print(np.concatenate((a,b),axis=1))
# # 竖向拼接数组
# print(np.vstack((a,b)))
# # 竖向拼接数组
# print(np.concatenate((a,b),axis=0))
# 深度组合成三维数组
# print(np.dstack((a, b)))
"""
"""
# 数组的分割
# a = np.arange(9).reshape(3,3)
# print(a)
# # 按照水平方向分割
# print(np.hsplit(a,3))
# # 按照水平方向分割
# print(np.split(a,3,axis=1))
# # 按照垂直方向分割
# print(np.vsplit(a,3))
# print(np.split(a,3,axis=0))
"""

# # 数组的转换
# b = np.array([[1,2],[3,4]])
# # print(b.tolist())
# # print(b.tostring())
# print(b)
# print(b.mean(axis=1))
