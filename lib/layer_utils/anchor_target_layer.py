# -*- coding:utf-8 -*-



# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.cython_bbox import bbox_overlaps
from model.bbox_transform import bbox_transform

def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchors):
  """Same as the anchor target layer in original Fast/er RCNN """
  A = num_anchors
  total_anchors = all_anchors.shape[0]
  # anchor的总数
  K = total_anchors / num_anchors

  # allow boxes to sit over the edge by a small amount
  _allowed_border = 0

  # map of shape (..., H, W)
  height, width = rpn_cls_score.shape[1:3]

  # only keep anchors inside the image
  # 筛选出all_anchors中所有满足条件的anchor的索引
  inds_inside = np.where(
    (all_anchors[:, 0] >= -_allowed_border) &
    (all_anchors[:, 1] >= -_allowed_border) &
    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
  )[0]

  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]

  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  # 建立一个随机生成的数组，维度指定
  labels.fill(-1)
  # labels中的内容用-1初始化（1：前景，0：背景，-1：忽略）

  # overlaps between the anchors and the gt boxes
  # overlaps (ex, gt)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(anchors, dtype=np.float),
    np.ascontiguousarray(gt_boxes, dtype=np.float))
  # 计算rpn得到的anchor和groundtrue_box的重叠面积shape=(len(anchors),len(gx_boxes))
  # overlaps[i][j]代表了第i个anchor与第j个gtbox的重叠面积
  argmax_overlaps = overlaps.argmax(axis=1)
  # 返回每个anchor对应的最匹配的gt_box的编号
  # axis=1：找每一行的最大值，拿出第1+1维度进行比较
  max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
  # 根据索引得到值
  # max_overlap是满足要求的anchor的分数
  gt_argmax_overlaps = overlaps.argmax(axis=0)
  # 取每一列的最大值，返回与每个gt_box最匹配的anchor
  gt_max_overlaps = overlaps[gt_argmax_overlaps,
                             np.arange(overlaps.shape[1])]
  # 返回与每个gt_box最匹配的anchor
  gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
  # np.where输出overlaps中满足条件的元素的位置索引。[0]是第0维坐标
  # ！！特么返回的是gt_max_overlaps按照降序排列后在overlaps中位置的第多少行
  # 返回每个gt_boxes对应的overlap最大的anchor的序号，降序排列

  if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
      # 记录anchor与gt_box的ioU值小于RPN_NEGATIVE_OVERLAP的为负样本

  # fg label: for each gt, anchor with highest overlap
  labels[gt_argmax_overlaps] = 1
  # 记录anchor与gt_box的ioU值最大的为正样本

  # fg label: above threshold IOU
  labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
  # 记录anchor与gt_box的ioU值大于RPN_POSITIVE_OVERLAP的为正样本

  if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
    # assign bg labels last so that negative labels can clobber positives
    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

  # subsample positive labels if we have too many
  # 如果正样本过多，就进行采样。采样比例由RPN_FG_FRACTION和RPN_BATCHSIZE控制
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)# 0.5*256
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:
    disable_inds = npr.choice(
      fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1
      # numpy.random.choice  参数size表示输出的shape，

  # subsample negative labels if we have too many
  # 如果负样本过多，就进行采样。采样比例由
  num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(
      bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1

  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
  # anchor是所有满足条件的anchor，argmax_overlaps是每个anchor对应的最匹配的gt_box的编号
  # gt_boxes是ground truth边界框
  # gt_boxes[argmax_overlaps, :]是每个anchor对应ioU最大的gt_boxes的边界框,
  # _compute_targets返回gt框和anchor框相差的dxdydhdw
  bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  # only the positive ones have regression targets
  bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
  # RPN_BBOX_INSIDE_WEIGHTS=[1,1,1,1]

  bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
  else:
    assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
            (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
    positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                        np.sum(labels == 1))
    negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                        np.sum(labels == 0))

  # 计算正样本/负样本和anchor总数的比值
  bbox_outside_weights[labels == 1, :] = positive_weights
  bbox_outside_weights[labels == 0, :] = negative_weights

  # map up to original set of anchors
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
  bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

  # labels
  labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, 1, A * height, width))
  rpn_labels = labels

  # bbox_targets
  bbox_targets = bbox_targets \
    .reshape((1, height, width, A * 4))

  rpn_bbox_targets = bbox_targets
  # bbox_inside_weights
  bbox_inside_weights = bbox_inside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_inside_weights = bbox_inside_weights

  # bbox_outside_weights
  bbox_outside_weights = bbox_outside_weights \
    .reshape((1, height, width, A * 4))

  rpn_bbox_outside_weights = bbox_outside_weights
  # rpn_bbox_inside_weights用于把是object的box过滤出来，
  # 因为并不是所有的anchors都是有object的。
  # rpn_bbox_inside_weights用于设置标记为1的box和标记为0的box的权值比率

  # rpn_bbox_targets是计算出来的dxdydhdw

  # rpn_labels是标签值，1,0,-1
  return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
  """ Unmap a subset of item (data) back to the original set of items (of
  size count) """
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds, :] = data
  return ret


def _compute_targets(ex_rois, gt_rois):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 5

  return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
