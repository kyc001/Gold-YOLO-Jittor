"""
GOLD-YOLO Jittor版本 - Task Aligned Assigner
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import jittor as jt
import jittor.nn as nn
from yolov6.assigners.assigner_utils import select_candidates_in_gts, select_highest_overlaps, iou_calculator, \
    dist_calculator


class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 topk=13,
                 num_classes=80,
                 alpha=1.0,
                 beta=6.0,
                 eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    
    def execute(self,
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        
        if self.n_max_boxes == 0:
            return jt.full_like(pd_scores[..., 0], self.bg_idx), \
                jt.zeros_like(pd_bboxes), \
                jt.zeros_like(pd_scores), \
                jt.zeros_like(pd_scores[..., 0])
        
        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
        target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst = [], [], [], []
        # loop batch dim in case of numerous object box
        for i in range(cycle):
            start, end = i * step, (i + 1) * step
            pd_scores_ = pd_scores[start:end, ...]
            pd_bboxes_ = pd_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            mask_gt_ = mask_gt[start:end, ...]
            
            mask_pos, align_metric, overlaps = self.get_pos_mask(
                    pd_scores_, pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_)
            
            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                    mask_pos, overlaps, self.n_max_boxes)
            
            # assigned target
            target_labels, target_bboxes, target_scores = self.get_targets(
                    gt_labels_, gt_bboxes_, target_gt_idx, fg_mask)
            
            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.max(dim=-1, keepdim=True)[0]
            pos_overlaps = (overlaps * mask_pos).max(dim=-1, keepdim=True)[0]
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            target_scores = target_scores * norm_align_metric
            
            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)
        
        # concat
        target_labels = jt.concat(target_labels_lst, 0)
        target_bboxes = jt.concat(target_bboxes_lst, 0)
        target_scores = jt.concat(target_scores_lst, 0)
        fg_mask = jt.concat(fg_mask_lst, 0)
        
        return target_labels, target_bboxes, target_scores, fg_mask.bool()
    
    def get_pos_mask(self,
                     pd_scores,
                     pd_bboxes,
                     gt_labels,
                     gt_bboxes,
                     anc_points,
                     mask_gt):
        
        # get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask, (b, max_num_obj, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        
        return mask_pos, align_metric, overlaps
    
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        
        gt_labels = gt_labels.astype(jt.int64)
        ind = jt.zeros([2, self.bs, self.n_max_boxes], dtype=jt.int64)
        ind[0] = jt.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w
        
        overlaps = iou_calculator(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1))
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps
    
    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = jt.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).repeat([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs = jt.where(topk_mask, topk_idxs, jt.zeros_like(topk_idxs))
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        is_in_topk = jt.nn.one_hot(topk_idxs, metrics.shape[-1]).sum(-2)
        # avoid multiple gts assign to the same anchor
        is_in_topk = jt.where(is_in_topk > 1, jt.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.astype(metrics.dtype)
    
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        
        # assigned target labels, (b, 1)
        batch_ind = jt.arange(end=self.bs, dtype=jt.int64).unsqueeze(-1)
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)
        
        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        
        # assigned target scores
        target_labels = jt.where(fg_mask, target_labels, jt.full_like(target_labels, self.bg_idx))
        target_scores = jt.nn.one_hot(target_labels, self.num_classes + 1).float()
        target_scores = target_scores[:, :, :self.num_classes]
        
        return target_labels, target_bboxes, target_scores
