#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
逐步排查分配器问题
"""

import jittor as jt
import numpy as np
import traceback

# Set Jittor flags
jt.flags.use_cuda = 1

def test_tal_assigner_step_by_step():
    """逐步测试TAL分配器"""
    print("🔧 逐步测试TAL分配器...")
    
    try:
        from yolov6.assigners.tal_assigner import TaskAlignedAssigner
        
        # 创建分配器
        assigner = TaskAlignedAssigner(topk=13, num_classes=80, alpha=1.0, beta=6.0)
        
        # 创建测试数据
        batch_size = 1
        n_anchors = 100  # 使用更小的数据
        num_gt = 1  # 只用1个GT
        
        pred_scores = jt.randn(batch_size, n_anchors, 80)
        pred_bboxes = jt.randn(batch_size, n_anchors, 4)
        anchor_points = jt.randn(n_anchors, 2)
        
        # GT数据
        gt_labels = jt.ones((batch_size, num_gt, 1), dtype=jt.int64)
        gt_bboxes = jt.randn(batch_size, num_gt, 4)
        mask_gt = jt.ones((batch_size, num_gt, 1), dtype=jt.bool)
        
        print(f"  测试数据形状:")
        print(f"    pred_scores: {pred_scores.shape}")
        print(f"    pred_bboxes: {pred_bboxes.shape}")
        print(f"    gt_labels: {gt_labels.shape}")
        print(f"    gt_bboxes: {gt_bboxes.shape}")
        print(f"    mask_gt: {mask_gt.shape}")
        
        # 调用分配器
        print("  调用TAL分配器...")
        result = assigner(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt)
        
        print(f"  ✅ TAL分配器成功")
        print(f"    输出形状: {[x.shape for x in result]}")
        return True
        
    except Exception as e:
        print(f"  ❌ TAL分配器失败: {e}")
        print("  详细错误信息:")
        traceback.print_exc()
        return False

def test_loss_with_small_data():
    """使用小数据测试损失函数"""
    print("\n🔧 使用小数据测试损失函数...")
    
    try:
        from yolov6.models.losses.loss import ComputeLoss
        
        # 创建损失函数
        criterion = ComputeLoss(
            num_classes=80,
            ori_img_size=640,
            warmup_epoch=0,  # 强制使用formal_assigner
            use_dfl=True,
            reg_max=16,
            iou_type='giou'
        )
        
        # 创建小数据
        batch_size = 1
        n_anchors = 100  # 使用更小的anchor数量
        
        # 模拟模型输出
        feats = [
            jt.randn(1, 256, 10, 10),   # P3 - 更小的特征图
            jt.randn(1, 512, 5, 5),    # P4  
            jt.randn(1, 1024, 3, 3)    # P5
        ]
        
        pred_scores = jt.randn(batch_size, n_anchors, 80)
        pred_distri = jt.randn(batch_size, n_anchors, 68)  # 4 * (reg_max + 1)
        
        outputs = (feats, pred_scores, pred_distri)
        
        # 创建目标数据
        targets = [{
            'cls': jt.array([1]),  # 只有1个目标
            'bboxes': jt.array([[100, 100, 200, 200]]),
            'batch_idx': jt.array([0])
        }]
        
        print(f"  输入形状:")
        print(f"    pred_scores: {pred_scores.shape}")
        print(f"    pred_distri: {pred_distri.shape}")
        print(f"    targets: {len(targets)} 个目标")
        
        # 调用损失函数
        print("  调用损失函数...")
        loss, loss_items = criterion(outputs, targets, epoch_num=1, step_num=0)
        
        print(f"  ✅ 损失函数调用成功: {loss}")
        return True
        
    except Exception as e:
        print(f"  ❌ 损失函数调用失败: {e}")
        print("  详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔧 逐步排查分配器问题")
    print("=" * 50)
    
    # 测试分配器本身
    assigner_success = test_tal_assigner_step_by_step()
    
    # 测试损失函数
    loss_success = test_loss_with_small_data()
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"  分配器测试: {'✅ 成功' if assigner_success else '❌ 失败'}")
    print(f"  损失函数测试: {'✅ 成功' if loss_success else '❌ 失败'}")
    
    return assigner_success and loss_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
