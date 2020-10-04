#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf

a = tf.random.normal([3,3])
indices = [[0,0],[2,0]]
print(a.indices)
