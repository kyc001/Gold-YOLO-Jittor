#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
精确测试分配器调用的脚本
"""

import jittor as jt
import numpy as np
import traceback

# Set Jittor flags
jt.flags.use_cuda = 1

def test_exact_assigner_call():
    """精确测试分配器调用"""
    print("🔧 精确测试分配器调用...")
    
    try:
        from yolov6.assigners.tal_assigner import TaskAlignedAssigner
        from yolov6.utils.general import generate_anchors
        
        # 创建分配器
        assigner = TaskAlignedAssigner(topk=13, num_classes=80, alpha=1.0, beta=6.0)
        
        # 创建与损失函数中完全相同的数据
        batch_size = 1
        n_anchors = 8400
        
        # 模拟特征图
        feats = [
            jt.randn(1, 256, 80, 80),   # P3
            jt.randn(1, 512, 40, 40),   # P4  
            jt.randn(1, 1024, 20, 20)   # P5
        ]
        
        # 生成anchor_points和stride_tensor（与损失函数中相同）
        fpn_strides = [8, 16, 32]
        anchor_points, stride_tensor = generate_anchors(feats, fpn_strides, 0.5, 0.0, is_eval=True)
        
        print(f"  anchor_points形状: {anchor_points.shape}")
        print(f"  stride_tensor形状: {stride_tensor.shape}")
        
        # 创建预测数据
        pred_scores = jt.randn(batch_size, n_anchors, 80)
        pred_distri = jt.randn(batch_size, n_anchors, 68)
        
        # 解码bbox（与损失函数中相同）
        from yolov6.models.losses.loss import ComputeLoss

        # 创建损失函数实例来使用其bbox_decode方法
        criterion = ComputeLoss(
            num_classes=80,
            ori_img_size=640,
            warmup_epoch=0,
            use_dfl=True,
            reg_max=16,
            iou_type='giou'
        )

        pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
        
        print(f"  pred_scores形状: {pred_scores.shape}")
        print(f"  pred_bboxes形状: {pred_bboxes.shape}")
        
        # 创建GT数据
        gt_labels = jt.ones((batch_size, 2, 1), dtype=jt.int64)
        gt_bboxes = jt.array([[[100, 100, 200, 200], [300, 300, 400, 400]]], dtype=jt.float32)
        mask_gt = jt.ones((batch_size, 2, 1), dtype=jt.bool)
        
        print(f"  gt_labels形状: {gt_labels.shape}")
        print(f"  gt_bboxes形状: {gt_bboxes.shape}")
        print(f"  mask_gt形状: {mask_gt.shape}")
        
        # 确保stride_tensor形状正确
        if stride_tensor.shape[0] != n_anchors:
            print(f"  ⚠️ stride_tensor形状不匹配，重新生成...")
            stride_tensor = jt.concat([
                jt.full((n_anchors // 3,), fpn_strides[0], dtype=jt.float32),
                jt.full((n_anchors // 3,), fpn_strides[1], dtype=jt.float32),
                jt.full((n_anchors - 2 * (n_anchors // 3),), fpn_strides[2], dtype=jt.float32)
            ]).unsqueeze(-1)
            print(f"  修正后stride_tensor形状: {stride_tensor.shape}")
        
        # 计算scaled_pred_bboxes（与损失函数中相同）
        scaled_pred_bboxes = pred_bboxes.detach() * stride_tensor
        
        print(f"  scaled_pred_bboxes形状: {scaled_pred_bboxes.shape}")
        
        # 调用分配器（与损失函数中完全相同的调用）
        print("  调用TAL分配器...")
        result = assigner(
            pred_scores.detach(),
            scaled_pred_bboxes,
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        print(f"  ✅ TAL分配器调用成功")
        print(f"    输出形状: {[x.shape for x in result]}")
        return True
        
    except Exception as e:
        print(f"  ❌ TAL分配器调用失败: {e}")
        print("  详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔧 精确测试分配器调用")
    print("=" * 50)
    
    success = test_exact_assigner_call()
    
    print("\n" + "=" * 50)
    print(f"测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
