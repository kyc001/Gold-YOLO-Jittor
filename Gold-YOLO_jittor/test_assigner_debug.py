#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试分配器广播错误的最小重现脚本
"""

import jittor as jt
import numpy as np
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner

# Set Jittor flags
jt.flags.use_cuda = 1

def test_atss_assigner():
    """测试ATSS分配器"""
    print("测试ATSS分配器...")
    try:
        assigner = ATSSAssigner(9, num_classes=80)
        
        # 创建测试数据
        batch_size = 1
        n_anchors = 100
        num_gt = 2
        
        # 创建anchor boxes
        anchors = jt.randn(n_anchors, 4) * 100 + 200  # [n_anchors, 4]
        n_anchors_list = [n_anchors]
        
        # 创建GT数据
        gt_labels = jt.array([[1, 2]]).float().unsqueeze(-1)  # [1, 2, 1]
        gt_bboxes = jt.array([[[100, 100, 200, 200], [300, 300, 400, 400]]]).float()  # [1, 2, 4]
        mask_gt = jt.ones(1, 2, 1).bool()  # [1, 2, 1]
        
        # 创建预测数据
        pred_bboxes = jt.randn(n_anchors, 4) * 50 + 250  # [n_anchors, 4]
        
        print(f"输入形状:")
        print(f"  anchors: {anchors.shape}")
        print(f"  gt_labels: {gt_labels.shape}")
        print(f"  gt_bboxes: {gt_bboxes.shape}")
        print(f"  mask_gt: {mask_gt.shape}")
        print(f"  pred_bboxes: {pred_bboxes.shape}")
        
        # 调用分配器
        result = assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes)
        print("ATSS分配器测试成功!")
        return True
        
    except Exception as e:
        print(f"ATSS分配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tal_assigner():
    """测试TAL分配器"""
    print("\n测试TAL分配器...")
    try:
        assigner = TaskAlignedAssigner(topk=13, num_classes=80, alpha=1.0, beta=6.0)
        
        # 创建测试数据
        batch_size = 1
        n_anchors = 100
        num_gt = 2
        
        # 创建预测数据
        pred_scores = jt.randn(batch_size, n_anchors, 80)  # [1, n_anchors, 80]
        pred_bboxes = jt.randn(batch_size, n_anchors, 4) * 50 + 250  # [1, n_anchors, 4]
        anchor_points = jt.randn(n_anchors, 2) * 100 + 200  # [n_anchors, 2]
        
        # 创建GT数据
        gt_labels = jt.array([[1, 2]]).float().unsqueeze(-1)  # [1, 2, 1]
        gt_bboxes = jt.array([[[100, 100, 200, 200], [300, 300, 400, 400]]]).float()  # [1, 2, 4]
        mask_gt = jt.ones(1, 2, 1).bool()  # [1, 2, 1]
        
        print(f"输入形状:")
        print(f"  pred_scores: {pred_scores.shape}")
        print(f"  pred_bboxes: {pred_bboxes.shape}")
        print(f"  anchor_points: {anchor_points.shape}")
        print(f"  gt_labels: {gt_labels.shape}")
        print(f"  gt_bboxes: {gt_bboxes.shape}")
        print(f"  mask_gt: {mask_gt.shape}")
        
        # 调用分配器
        result = assigner(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt)
        print("TAL分配器测试成功!")
        return True
        
    except Exception as e:
        print(f"TAL分配器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔧 分配器广播错误调试测试")
    print("=" * 50)
    
    # 测试ATSS分配器
    atss_success = test_atss_assigner()
    
    # 测试TAL分配器
    tal_success = test_tal_assigner()
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"  ATSS分配器: {'✅ 成功' if atss_success else '❌ 失败'}")
    print(f"  TAL分配器: {'✅ 成功' if tal_success else '❌ 失败'}")
    
    if atss_success and tal_success:
        print("\n🎉 所有分配器测试通过!")
    else:
        print("\n⚠️ 存在分配器问题，需要进一步调试")

if __name__ == "__main__":
    main()
