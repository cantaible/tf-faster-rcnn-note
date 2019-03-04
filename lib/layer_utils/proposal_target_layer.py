# -*- coding:utf-8 -*-




# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps


def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
  """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  将目标检测方案分配给ground truth目标。生成建议分类标签和边界框回归目标。
  """

  # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
  # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
  all_rois = rpn_rois
  # 边框角点坐标
  all_scores = rpn_scores
  # 边框得分

  # Include ground-truth boxes in the set of candidate rois
  # 将ground-truth boxes包括进候选框集
  if cfg.TRAIN.USE_GT:
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack(
      (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )
    # vstack竖直方向堆叠，hstack水平堆叠
    # 将gt_boxes加入到all_rois，hstack是为了整格式
    # gt_boxes[:,:-1]取每行除了最后一个之外的其他元素

    # not sure if it a wise appending, but anyway i am not using it
    # 不确定它是否是一个明智的附加物，但无论如何，我没有使用它
    all_scores = np.vstack((all_scores, zeros))
    # 将gt_boxes的scores也加上，但为啥是0？

  num_images = 1
  rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
  fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

  # Sample rois with classification labels and bounding box regression
  # 采样roi带有分类标签和边界框
  # targets
  labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
    all_rois, all_scores, gt_boxes, fg_rois_per_image,
    rois_per_image, _num_classes)
  # 调用sample_rois函数对roi做抽样，目的是让roi总数保持为rois_per_image（默认是128），

  rois = rois.reshape(-1, 5)
  roi_scores = roi_scores.reshape(-1)
  labels = labels.reshape(-1, 1)
  bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
  bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
  bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

  return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
  """Bounding-box regression targets (bbox_target_data) are stored in a
  compact form N x (class, tx, ty, tw, th)

  This function expands those targets into the 4-of-4*K representation used
  by the network (i.e. only one class has non-zero targets).

  Returns:
      bbox_target (ndarray): N x 4K blob of regression targets
      bbox_inside_weights (ndarray): N x 4K blob of loss weights
  """

  clss = bbox_target_data[:, 0]
  bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
  bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
  inds = np.where(clss > 0)[0]
  for ind in inds:
    cls = clss[ind]
    start = int(4 * cls)
    end = start + 4
    bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
  return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
  """Compute bounding-box regression targets for an image."""

  assert ex_rois.shape[0] == gt_rois.shape[0]
  assert ex_rois.shape[1] == 4
  assert gt_rois.shape[1] == 4

  targets = bbox_transform(ex_rois, gt_rois)
  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
    targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
               / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
  return np.hstack(
    (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
  """Generate a random sample of RoIs comprising foreground and background
  examples.
  生成包含前景和背景示例的随机ROI样本。
  调用sample_rois函数对roi做抽样，目的是让roi总数保持为rois_per_image（默认是128）
  同时正负roi的比例是1:3左右
  """
  # overlaps: (rois x gt_boxes)
  overlaps = bbox_overlaps(
    np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
    np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
  gt_assignment = overlaps.argmax(axis=1)
  # 返回每个anchor对应的最匹配的gt_box的编号
  # axis=1：找每一行的最大值，拿出第1+1维度进行比较
  max_overlaps = overlaps.max(axis=1)
  # 返回每个anchor对应的最匹配的gt_box的overlap值
  labels = gt_boxes[gt_assignment, 4]
  # 对每个rois，找到归属的类别(-1,0,1)

  # Select foreground RoIs as those with >= FG_THRESH overlap
  fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
  # Guard against the case when an image has fewer than fg_rois_per_image
  # 找到属于前景的roi
  # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  # 选择在BG_THRESH_LO, BG_THRESH_HI之间的
  bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                     (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
  # 找到属于背景的rois（与gt_box重叠率介于0和0.5之间的）

  # Small modification to the original version where we ensure a fixed number of regions are sampled
  if fg_inds.size > 0 and bg_inds.size > 0:
    fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
    # fg_rois_per_image：每张图片上选取的属于前景的rois的数量
    fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
    # 如果有必要，则随机选取一部分
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    to_replace = bg_inds.size < bg_rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
  elif fg_inds.size > 0:
    to_replace = fg_inds.size < rois_per_image
    fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = rois_per_image
  elif bg_inds.size > 0:
    to_replace = bg_inds.size < rois_per_image
    bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
    fg_rois_per_image = 0
  else:
    import pdb
    pdb.set_trace()

  # The indices that we're selecting (both fg and bg)
  keep_inds = np.append(fg_inds, bg_inds)
  # 记录下最终保存的框
  # Select sampled values from various arrays:
  labels = labels[keep_inds]
  # 记录下最终保留的框对应的label
  # Clamp labels for the background RoIs to 0
  labels[int(fg_rois_per_image):] = 0
  # 把背景框的label置为0
  rois = all_rois[keep_inds]
  roi_scores = all_scores[keep_inds]

  bbox_target_data = _compute_targets(
    rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
  # 得到最终保留的框的类别ground truth值和坐标变换gt值
  # _compute_targets计算坐标回归标签

  bbox_targets, bbox_inside_weights = \
    _get_bbox_regression_labels(bbox_target_data, num_classes)
  # 得到最终计算loss时使用的gt边框回归值和bbox_inside_weights
  # _get_bbox_regression_labels函数将坐标标签扩充，变成训练所需的格式


  return labels, rois, roi_scores, bbox_targets, bbox_inside_weights
