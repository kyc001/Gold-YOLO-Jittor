#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
专门测试完整分配器的脚本
"""

import jittor as jt
import numpy as np
import traceback

# Set Jittor flags
jt.flags.use_cuda = 1

def test_complete_assigners():
    """测试完整分配器"""
    print("🔧 测试完整分配器...")
    
    try:
        from yolov6.assigners.tal_assigner import TaskAlignedAssigner
        from yolov6.assigners.atss_assigner import ATSSAssigner
        
        # 创建分配器
        tal_assigner = TaskAlignedAssigner(topk=13, num_classes=80, alpha=1.0, beta=6.0)
        atss_assigner = ATSSAssigner(9, num_classes=80)
        
        # 创建测试数据
        batch_size = 1
        n_anchors = 8400
        num_gt = 2
        
        # TAL分配器测试数据
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
        
        # 测试TAL分配器
        print("\n  测试TAL分配器...")
        try:
            tal_result = tal_assigner(pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt)
            print(f"  ✅ TAL分配器成功")
            print(f"    输出形状: {[x.shape for x in tal_result]}")
        except Exception as e:
            print(f"  ❌ TAL分配器失败: {e}")
            traceback.print_exc()
            return False
        
        # ATSS分配器测试数据
        anchors = jt.randn(n_anchors, 4)
        n_anchors_list = [2800, 2800, 2800]  # 每个层级的anchor数量
        
        # 测试ATSS分配器
        print("\n  测试ATSS分配器...")
        try:
            atss_result = atss_assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes)
            print(f"  ✅ ATSS分配器成功")
            print(f"    输出形状: {[x.shape for x in atss_result]}")
        except Exception as e:
            print(f"  ❌ ATSS分配器失败: {e}")
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ 分配器测试失败: {e}")
        traceback.print_exc()
        return False

def test_loss_with_complete_assigners():
    """测试使用完整分配器的损失函数"""
    print("\n🔧 测试使用完整分配器的损失函数...")
    
    try:
        from yolov6.models.losses.loss import ComputeLoss
        
        # 强制使用完整分配器
        criterion = ComputeLoss(
            num_classes=80,
            ori_img_size=640,
            warmup_epoch=0,  # 使用formal_assigner
            use_dfl=True,
            reg_max=16,
            iou_type='giou'
        )
        
        # 创建测试数据
        batch_size = 1
        n_anchors = 8400
        
        # 模拟模型输出
        feats = [
            jt.randn(1, 256, 80, 80),   # P3
            jt.randn(1, 512, 40, 40),   # P4  
            jt.randn(1, 1024, 20, 20)   # P5
        ]
        
        pred_scores = jt.randn(batch_size, n_anchors, 80)
        pred_distri = jt.randn(batch_size, n_anchors, 68)  # 4 * (reg_max + 1)
        
        outputs = (feats, pred_scores, pred_distri)
        
        # 创建目标数据
        targets = [{
            'cls': jt.array([1, 2]),
            'bboxes': jt.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
            'batch_idx': jt.array([0, 0])
        }]
        
        print(f"  输入形状:")
        print(f"    pred_scores: {pred_scores.shape}")
        print(f"    pred_distri: {pred_distri.shape}")
        print(f"    targets: {len(targets)} 个目标")
        
        # 调用损失函数，强制使用完整分配器
        print("  调用损失函数（epoch_num=1，强制使用完整分配器）...")
        loss, loss_items = criterion(outputs, targets, epoch_num=1, step_num=0)
        
        print(f"  ✅ 完整分配器损失函数调用成功: {loss}")
        return True
        
    except Exception as e:
        print(f"  ❌ 完整分配器损失函数调用失败: {e}")
        print("  详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔧 完整分配器专项测试")
    print("=" * 50)
    
    # 测试分配器本身
    assigner_success = test_complete_assigners()
    
    # 测试损失函数中的完整分配器
    loss_success = test_loss_with_complete_assigners()
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"  分配器测试: {'✅ 成功' if assigner_success else '❌ 失败'}")
    print(f"  损失函数测试: {'✅ 成功' if loss_success else '❌ 失败'}")
    
    if assigner_success and loss_success:
        print("\n🎉 完整分配器测试通过!")
        return True
    else:
        print("\n⚠️ 完整分配器存在问题")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
