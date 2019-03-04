# -*- coding:utf-8 -*-



# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  '''
  A simplified version compared to fast/er RCNN
     For details please see the technical report
  :param rpn_cls_prob:
  :param rpn_bbox_pred:
  :param im_info: [M,N,scale_factor]保存了将任意图像缩放到M×N的所有信息
  :param cfg_key:
  :param _feat_stride:feat_stride=16用于计算anchor的偏移量
  :param anchors:
  :param num_anchors:
  :return:
  '''

  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
  scores = scores.reshape((-1, 1))
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  # 计算得到bbox四个顶点坐标
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  '''
  按照输入的foreground softmax降序排列，提取前pre_nms_topN（6000）的结果
  提取修正后的foreground anchor
  '''
  order = scores.ravel().argsort()[::-1]
  # ravel数组扁平化，降序排列
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
  proposals = proposals[order, :]
  # anchor坐标
  scores = scores[order]
  # anchor分数

  # Non-maximal suppression
  keep = nms(np.hstack((proposals, scores)), nms_thresh)

  # Pick th top region proposals after NMS
  '''
  再次按照nms后的foreground softmax由大到小排列，提取前post_nms_topN（300）结果作为proposals的输出
  '''
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep]

  # Only support single image as input
  batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
  blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

  return blob, scores


def proposal_layer_tf(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  if type(cfg_key) == bytes:
    cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  # shape=(1,?,?,18)
  scores = tf.reshape(scores, shape=(-1,))
  # bbox
  rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

  proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
  # rpn_bbox_pred是dx,dy,dh,dw
  # proposals是经过dx,dy,dh,dw修正后得到的bbox角点坐标
  proposals = clip_boxes_tf(proposals, im_info[:2])
  # 裁剪掉超出图像边界的部分

  # Non-maximal suppression
  indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)
  # 非最大值抑制
  # 去除掉与这个概率最大的边界框的loU大于一个阈值的其他边界框
  #　按照scores降序选择边界框的子集
  # 返回的是选出来，留下来的边框下标

  boxes = tf.gather(proposals, indices)
  # 得到proposals中第indices个索引对应的值
  # boxes是选出来的边框
  boxes = tf.to_float(boxes)
  scores = tf.gather(scores, indices)
  # scores是选出来框对应的得分
  scores = tf.reshape(scores, shape=(-1, 1))

  # Only support single image as input
  batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
  blob = tf.concat([batch_inds, boxes], 1)
  # 链接bath_inds和boxes
  # blob是边框坐标，前面加了bath_inds貌似是为了让索引号从1开始
  # scores是边框对应的分数

  return blob, scores


