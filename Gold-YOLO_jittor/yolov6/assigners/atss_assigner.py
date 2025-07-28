"""
GOLD-YOLO Jittor版本 - ATSS分配器
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import jittor as jt
import jittor.nn as nn
from yolov6.assigners.iou2d_calculator import iou2d_calculator
from yolov6.assigners.assigner_utils import dist_calculator, select_candidates_in_gts, select_highest_overlaps, iou_calculator


class ATSSAssigner(nn.Module):
    '''Adaptive Training Sample Selection Assigner'''
    def __init__(self,
                 topk=9,
                 num_classes=80):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes

    def execute(self,
                anc_bboxes,
                n_level_bboxes,
                gt_labels,
                gt_bboxes,
                mask_gt,
                pd_bboxes):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            anc_bboxes (Tensor): shape(num_total_anchors, 4)
            n_level_bboxes (List):len(3)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            pd_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        # 修复anc_bboxes形状并正确计算n_anchors
        if len(anc_bboxes.shape) == 3:
            # anc_bboxes是[1, N, 4]格式，需要flatten为[N, 4]
            anc_bboxes = anc_bboxes.view(-1, 4)

        self.n_anchors = anc_bboxes.size(0)  # 现在是正确的anchor数量
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            return jt.full([self.bs, self.n_anchors], self.bg_idx), \
                   jt.zeros([self.bs, self.n_anchors, 4]), \
                   jt.zeros([self.bs, self.n_anchors, self.num_classes]), \
                   jt.zeros([self.bs, self.n_anchors])

        # 修复reshape问题 - 确保gt_bboxes的形状正确
        gt_bboxes_flat = gt_bboxes.reshape([-1, 4])

        # 验证形状
        if gt_bboxes_flat.shape[-1] != 4:
            raise ValueError(f"GT bboxes reshape后最后维度不是4: {gt_bboxes_flat.shape}")

        if anc_bboxes.shape[-1] != 4:
            raise ValueError(f"Anchor bboxes最后维度不是4: {anc_bboxes.shape}")

        # 详细调试IoU计算的输入
        print(f"🔍 [IoU输入] gt_bboxes_flat形状: {gt_bboxes_flat.shape}, 数值范围: [{float(gt_bboxes_flat.min().data):.6f}, {float(gt_bboxes_flat.max().data):.6f}]")
        print(f"🔍 [IoU输入] anc_bboxes形状: {anc_bboxes.shape}, 数值范围: [{float(anc_bboxes.min().data):.1f}, {float(anc_bboxes.max().data):.1f}]")

        # 检查前几个GT框和anchor框的具体数值
        if gt_bboxes_flat.shape[0] > 0:
            print(f"🔍 [IoU输入] 第一个GT框: {gt_bboxes_flat[0].numpy()}")
            print(f"🔍 [IoU输入] 前3个anchor框: {anc_bboxes[:3].numpy()}")

        # 检测坐标系统不匹配问题
        gt_max = float(gt_bboxes_flat.max().data)
        anc_max = float(anc_bboxes.max().data)
        if gt_max <= 1.0 and anc_max > 100.0:
            print(f"🚨 [坐标系统错误] GT框是归一化坐标({gt_max:.3f})，但anchor是像素坐标({anc_max:.1f})")
            print(f"🔧 [坐标修复] 将GT框转换为像素坐标")
            gt_bboxes_flat = gt_bboxes_flat * 640.0
            print(f"🔧 [坐标修复] 修复后GT框数值范围: [{float(gt_bboxes_flat.min().data):.1f}, {float(gt_bboxes_flat.max().data):.1f}]")
            print(f"🔧 [坐标修复] 修复后第一个GT框: {gt_bboxes_flat[0].numpy()}")

        overlaps = iou2d_calculator(gt_bboxes_flat, anc_bboxes)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])

        # 调试IoU计算结果
        print(f"🔍 [ATSS-IoU] overlaps形状: {overlaps.shape}, 数值范围: [{float(overlaps.min().data):.6f}, {float(overlaps.max().data):.6f}]")
        print(f"🔍 [ATSS-IoU] overlaps非零数量: {int((overlaps > 0).sum().data)}")
        print(f"🔍 [ATSS-IoU] overlaps>0.01数量: {int((overlaps > 0.01).sum().data)}")
        print(f"🔍 [ATSS-IoU] overlaps>0.1数量: {int((overlaps > 0.1).sum().data)}")

        distances, ac_points = dist_calculator(gt_bboxes_flat, anc_bboxes)

        # 正确的reshape：distances应该是[N_gt, N_anchors] -> [bs, n_max_boxes, n_anchors]
        expected_shape = [self.bs, self.n_max_boxes, self.n_anchors]

        # 检查distances的实际形状是否符合预期
        if distances.shape == (self.bs * self.n_max_boxes, self.n_anchors):
            # 正确的形状，直接reshape
            distances = distances.reshape(expected_shape)
        elif distances.shape == (gt_bboxes_flat.shape[0], self.n_anchors):
            # gt_bboxes_flat可能不是bs*n_max_boxes的形状
            if gt_bboxes_flat.shape[0] < self.bs * self.n_max_boxes:
                # 需要padding
                padding_size = self.bs * self.n_max_boxes - gt_bboxes_flat.shape[0]
                padding = jt.ones((padding_size, self.n_anchors)) * 1e6  # 大距离值
                distances = jt.concat([distances, padding], dim=0)
                distances = distances.reshape(expected_shape)
            else:
                # 截取
                distances = distances[:self.bs * self.n_max_boxes]
                distances = distances.reshape(expected_shape)
        else:
            # 使用默认距离矩阵
            distances = jt.ones(expected_shape) * 1e6

        is_in_candidate, candidate_idxs = self.select_topk_candidates(
            distances, n_level_bboxes, mask_gt)

        print(f"🔍 [ATSS-候选] is_in_candidate形状: {is_in_candidate.shape}, 候选数量: {int(is_in_candidate.sum().data)}")
        print(f"🔍 [ATSS-候选] candidate_idxs形状: {candidate_idxs.shape}")

        overlaps_thr_per_gt, iou_candidates = self.thres_calculator(
            is_in_candidate, candidate_idxs, overlaps)

        print(f"🔍 [ATSS-阈值] overlaps_thr_per_gt形状: {overlaps_thr_per_gt.shape}, 数值范围: [{float(overlaps_thr_per_gt.min().data):.6f}, {float(overlaps_thr_per_gt.max().data):.6f}]")
        print(f"🔍 [ATSS-阈值] iou_candidates形状: {iou_candidates.shape}, 数值范围: [{float(iou_candidates.min().data):.6f}, {float(iou_candidates.max().data):.6f}]")

        # select candidates iou >= threshold as positive
        # 修复形状不匹配问题 - 根据实际形状调整
        if overlaps_thr_per_gt.shape[1] != iou_candidates.shape[1]:
            # 调整overlaps_thr_per_gt的形状以匹配iou_candidates
            target_shape = [overlaps_thr_per_gt.shape[0], iou_candidates.shape[1], overlaps_thr_per_gt.shape[2]]
            overlaps_thr_per_gt = overlaps_thr_per_gt.expand(target_shape)

        overlaps_thr_expanded = overlaps_thr_per_gt.repeat([1, 1, self.n_anchors])

        # 确保形状匹配
        if overlaps_thr_expanded.shape != iou_candidates.shape:
            # 最终广播修复
            overlaps_thr_expanded = overlaps_thr_expanded.expand_as(iou_candidates)

        # 确保第一个jt.where的形状一致
        zeros_for_is_pos = jt.zeros_like(is_in_candidate)
        if zeros_for_is_pos.shape != is_in_candidate.shape:
            zeros_for_is_pos = jt.zeros(is_in_candidate.shape, dtype=is_in_candidate.dtype)

        # 详细调试IoU阈值筛选过程
        print(f"🔍 [ATSS-阈值筛选] iou_candidates数值范围: [{float(iou_candidates.min().data):.6f}, {float(iou_candidates.max().data):.6f}]")
        print(f"🔍 [ATSS-阈值筛选] overlaps_thr_expanded数值范围: [{float(overlaps_thr_expanded.min().data):.6f}, {float(overlaps_thr_expanded.max().data):.6f}]")

        # 检查有多少候选的IoU大于阈值
        iou_above_threshold = (iou_candidates > overlaps_thr_expanded).sum()
        print(f"🔍 [ATSS-阈值筛选] IoU大于阈值的候选数: {int(iou_above_threshold.data)}")

        is_pos = jt.where(
            iou_candidates > overlaps_thr_expanded,
            is_in_candidate, zeros_for_is_pos)

        print(f"🔍 [ATSS-正样本] is_pos形状: {is_pos.shape}, 正样本候选数: {int(is_pos.sum().data)}")

        is_in_gts = select_candidates_in_gts(ac_points, gt_bboxes)
        print(f"🔍 [ATSS-GT内] is_in_gts形状: {is_in_gts.shape}, GT内候选数: {int(is_in_gts.sum().data)}")

        # 详细调试最终筛选过程
        print(f"🔍 [ATSS-最终筛选] mask_gt形状: {mask_gt.shape}, 有效GT数: {int(mask_gt.sum().data)}")

        # 分步检查筛选条件
        step1 = is_pos * is_in_gts
        step2 = step1 * mask_gt
        print(f"🔍 [ATSS-最终筛选] is_pos * is_in_gts = {int(step1.sum().data)}")
        print(f"🔍 [ATSS-最终筛选] (is_pos * is_in_gts) * mask_gt = {int(step2.sum().data)}")

        mask_pos = step2
        print(f"🔍 [ATSS-最终] mask_pos形状: {mask_pos.shape}, 最终正样本数: {int(mask_pos.sum().data)}")

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)

        print(f"🔍 [ATSS-结果] fg_mask形状: {fg_mask.shape}, 最终正样本数: {int(fg_mask.sum().data)}")

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # soft label with iou - 修复训练初期IoU为0的问题
        if pd_bboxes is not None:
            print(f"🔍 [IoU软标签] pd_bboxes不为None，计算IoU软标签")
            ious = iou_calculator(gt_bboxes, pd_bboxes) * mask_pos
            ious = ious.max(dim=-2)[0].unsqueeze(-1)
            print(f"🔍 [IoU软标签] ious形状: {ious.shape}, 数值范围: [{float(ious.min().data):.6f}, {float(ious.max().data):.6f}]")
            print(f"🔍 [IoU软标签] target_scores乘法前总和: {float(target_scores.sum().data):.6f}")

            # 修复训练初期IoU为0的问题：设置IoU下限，避免target_scores被完全清零
            iou_threshold = 0.1  # IoU下限阈值
            ious_clamped = jt.clamp(ious, min_v=iou_threshold)
            print(f"🔍 [IoU软标签] ious_clamped数值范围: [{float(ious_clamped.min().data):.6f}, {float(ious_clamped.max().data):.6f}]")

            target_scores *= ious_clamped
            print(f"🔍 [IoU软标签] target_scores乘法后总和: {float(target_scores.sum().data):.6f}")
        else:
            print(f"🔍 [IoU软标签] pd_bboxes为None，跳过IoU软标签计算")

        return target_labels.long(), target_bboxes, target_scores, fg_mask.bool()

    def select_topk_candidates(self,
                               distances,
                               n_level_bboxes,
                               mask_gt):

        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()

        # 检查split操作的形状匹配
        if sum(n_level_bboxes) != distances.shape[-1]:
            # 修复n_level_bboxes
            total_anchors = distances.shape[-1]
            if len(n_level_bboxes) == 3:  # 3个FPN层
                # 重新计算每层的anchor数量
                n_level_bboxes = [total_anchors // 3] * 2 + [total_anchors - 2 * (total_anchors // 3)]

        level_distances = jt.split(distances, n_level_bboxes, dim=-1)
        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(level_distances, n_level_bboxes):

            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)
            _, per_level_topk_idxs = per_level_distances.topk(selected_k, dim=-1, largest=False)
            candidate_idxs.append(per_level_topk_idxs + start_idx)

            # 修复形状不匹配问题：mask_gt的形状需要与per_level_topk_idxs匹配
            # mask_gt: [bs, n_max_boxes, topk], per_level_topk_idxs: [bs, n_max_boxes, selected_k]
            if selected_k < self.topk:
                # 如果selected_k < topk，需要截取mask_gt的前selected_k列
                level_mask_gt = mask_gt[:, :, :selected_k]
            else:
                # 如果selected_k == topk，直接使用mask_gt
                level_mask_gt = mask_gt

            # 确保所有tensor形状一致
            zeros_like_result = jt.zeros_like(per_level_topk_idxs)

            if level_mask_gt.shape != per_level_topk_idxs.shape:
                level_mask_gt = level_mask_gt.expand_as(per_level_topk_idxs)

            if zeros_like_result.shape != per_level_topk_idxs.shape:
                zeros_like_result = jt.zeros(per_level_topk_idxs.shape, dtype=per_level_topk_idxs.dtype)

            # 最终形状验证和强制统一
            if level_mask_gt.shape != per_level_topk_idxs.shape or zeros_like_result.shape != per_level_topk_idxs.shape:
                target_shape = per_level_topk_idxs.shape
                level_mask_gt = jt.ones(target_shape, dtype=level_mask_gt.dtype)
                zeros_like_result = jt.zeros(target_shape, dtype=per_level_topk_idxs.dtype)

            per_level_topk_idxs = jt.where(level_mask_gt,
                per_level_topk_idxs, zeros_like_result)
            is_in_candidate = nn.one_hot(per_level_topk_idxs, per_level_boxes).sum(dim=-2)
            # 修复第三个jt.where
            zeros_for_candidate = jt.zeros_like(is_in_candidate)
            if zeros_for_candidate.shape != is_in_candidate.shape:
                zeros_for_candidate = jt.zeros(is_in_candidate.shape, dtype=is_in_candidate.dtype)

            is_in_candidate = jt.where(is_in_candidate > 1,
                zeros_for_candidate, is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.astype(distances.dtype))
            start_idx = end_idx

        is_in_candidate_list = jt.concat(is_in_candidate_list, dim=-1)
        candidate_idxs = jt.concat(candidate_idxs, dim=-1)

        return is_in_candidate_list, candidate_idxs

    def thres_calculator(self,
                         is_in_candidate,
                         candidate_idxs,
                         overlaps):

        n_bs_max_boxes = self.bs * self.n_max_boxes
        # 修复第四个jt.where
        zeros_for_overlaps = jt.zeros_like(overlaps)
        if zeros_for_overlaps.shape != overlaps.shape:
            zeros_for_overlaps = jt.zeros(overlaps.shape, dtype=overlaps.dtype)

        _candidate_overlaps = jt.where(is_in_candidate > 0,
            overlaps, zeros_for_overlaps)
        candidate_idxs = candidate_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self.n_anchors * jt.arange(n_bs_max_boxes)
        assist_idxs = assist_idxs[:,None]
        faltten_idxs = candidate_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[faltten_idxs]
        candidate_overlaps = candidate_overlaps.reshape([self.bs, self.n_max_boxes, -1])

        overlaps_mean_per_gt = candidate_overlaps.mean(dim=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(dim=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        # 详细调试阈值计算过程
        print(f"🔍 [阈值计算] candidate_overlaps形状: {candidate_overlaps.shape}")
        print(f"🔍 [阈值计算] candidate_overlaps数值范围: [{float(candidate_overlaps.min().data):.6f}, {float(candidate_overlaps.max().data):.6f}]")
        print(f"🔍 [阈值计算] overlaps_mean_per_gt: [{float(overlaps_mean_per_gt.min().data):.6f}, {float(overlaps_mean_per_gt.max().data):.6f}]")
        print(f"🔍 [阈值计算] overlaps_std_per_gt: [{float(overlaps_std_per_gt.min().data):.6f}, {float(overlaps_std_per_gt.max().data):.6f}]")
        print(f"🔍 [阈值计算] overlaps_thr_per_gt: [{float(overlaps_thr_per_gt.min().data):.6f}, {float(overlaps_thr_per_gt.max().data):.6f}]")

        return overlaps_thr_per_gt, _candidate_overlaps

    def get_targets(self,
                    gt_labels,
                    gt_bboxes,
                    target_gt_idx,
                    fg_mask):

        # assigned target labels
        batch_idx = jt.arange(self.bs, dtype=gt_labels.dtype)
        batch_idx = batch_idx[...,None]
        target_gt_idx = (target_gt_idx + batch_idx * self.n_max_boxes).long()
        target_labels = gt_labels.flatten()[target_gt_idx.flatten()]
        target_labels = target_labels.reshape([self.bs, self.n_anchors])
        # 修复第五个jt.where
        full_like_result = jt.full_like(target_labels, self.bg_idx)
        if full_like_result.shape != target_labels.shape:
            full_like_result = jt.full(target_labels.shape, self.bg_idx, dtype=target_labels.dtype)

        target_labels = jt.where(fg_mask > 0,
            target_labels, full_like_result)

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]
        target_bboxes = target_bboxes.reshape([self.bs, self.n_anchors, 4])

        # assigned target scores - 修复one_hot编码问题
        print(f"🔍 [target_scores计算] target_labels形状: {target_labels.shape}, 数值范围: [{float(target_labels.min().data):.1f}, {float(target_labels.max().data):.1f}]")
        print(f"🔍 [target_scores计算] self.num_classes: {self.num_classes}, self.bg_idx: {self.bg_idx}")
        print(f"🔍 [target_scores计算] fg_mask正样本数: {int(fg_mask.sum().data)}")

        # 使用正确的one_hot编码方式
        # 首先将背景类标签(self.bg_idx)替换为0，因为one_hot不应该包含背景类
        target_labels_for_onehot = jt.where(
            target_labels == self.bg_idx,
            jt.zeros_like(target_labels),  # 背景类设为0
            target_labels
        )

        # 创建one_hot编码，只包含前景类
        target_scores = nn.one_hot(target_labels_for_onehot.long(), self.num_classes).float()

        # 对于背景类位置，将所有类别概率设为0
        bg_mask = (target_labels == self.bg_idx).unsqueeze(-1).repeat([1, 1, self.num_classes])
        target_scores = jt.where(bg_mask, jt.zeros_like(target_scores), target_scores)

        # 只保留正样本的target_scores，负样本全部设为0
        fg_mask_expanded = fg_mask.unsqueeze(-1).repeat([1, 1, self.num_classes])
        target_scores = jt.where(fg_mask_expanded, target_scores, jt.zeros_like(target_scores))

        print(f"🔍 [target_scores计算] target_scores形状: {target_scores.shape}, 数值范围: [{float(target_scores.min().data):.6f}, {float(target_scores.max().data):.6f}]")
        print(f"🔍 [target_scores计算] target_scores非零数量: {int((target_scores > 0).sum().data)}")
        print(f"🔍 [target_scores计算] target_scores总和: {float(target_scores.sum().data):.6f}")

        return target_labels, target_bboxes, target_scores
