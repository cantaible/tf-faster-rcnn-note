# -*- coding:utf-8 -*-

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length

def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
  shift_x = tf.range(width) * feat_stride # width
  # tf.range创建数字序列，x偏移的距离
  shift_y = tf.range(height) * feat_stride # height
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  # tf.meshgrid用于从数组a和b产生网格
  # 用法: [A,B]=Meshgrid(a,b)，生成size(b)Xsize(a)大小的矩阵A和B。
  # 它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列。
  sx = tf.reshape(shift_x, shape=(-1,))#转换成列向量
  sy = tf.reshape(shift_y, shape=(-1,))
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  # 把这四个列向量合并tf.transpose转置
  K = tf.multiply(width, height)
  # 两个矩阵中对应元素相乘
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))
  # shape=(?,4,1),?应该是9
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  # A=9
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)
  # 创建一个常数张量

  length = K * A
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
  # anchor_constant(9*4)是以[0 0 15 15]为中心生成的9个anchor，shift是偏移规则(9,4,1)
  # tf.cast转换类型
  return tf.cast(anchors_tf, dtype=tf.float32), length
