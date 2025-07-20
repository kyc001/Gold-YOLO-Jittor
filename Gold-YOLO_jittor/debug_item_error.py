#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
调试item()错误的详细脚本
"""

import jittor as jt
import numpy as np
import traceback
from pathlib import Path

# Set Jittor flags
jt.flags.use_cuda = 1

def test_loss_function():
    """测试损失函数中的item()调用"""
    print("🔧 测试损失函数中的item()调用...")
    
    try:
        from yolov6.models.losses.loss import ComputeLoss
        
        # 创建损失函数
        criterion = ComputeLoss(
            num_classes=80,
            ori_img_size=640,
            warmup_epoch=0,
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
        
        # 调用损失函数
        print("  调用损失函数...")
        loss, loss_items = criterion(outputs, targets, epoch_num=0, step_num=0)
        
        print(f"  ✅ 损失函数调用成功: {loss}")
        return True
        
    except Exception as e:
        print(f"  ❌ 损失函数调用失败: {e}")
        print("  详细错误信息:")
        traceback.print_exc()
        return False

def test_simple_case():
    """测试简单情况"""
    print("\n🔧 测试简单的item()调用...")
    
    try:
        # 测试标量张量
        scalar = jt.array(5.0)
        print(f"  标量张量: {scalar}, 形状: {scalar.shape}, numel: {scalar.numel()}")
        val = scalar.item()
        print(f"  ✅ 标量.item()成功: {val}")
        
        # 测试向量张量
        vector = jt.array([1.0, 2.0, 3.0])
        print(f"  向量张量: 形状: {vector.shape}, numel: {vector.numel()}")
        
        # 测试sum()操作
        sum_result = vector.sum()
        print(f"  sum()结果: 形状: {sum_result.shape}, numel: {sum_result.numel()}")
        sum_val = sum_result.item()
        print(f"  ✅ sum().item()成功: {sum_val}")
        
        # 测试大张量的sum
        large_tensor = jt.ones(8400)
        print(f"  大张量: 形状: {large_tensor.shape}, numel: {large_tensor.numel()}")
        large_sum = large_tensor.sum()
        print(f"  大张量sum(): 形状: {large_sum.shape}, numel: {large_sum.numel()}")
        large_sum_val = large_sum.item()
        print(f"  ✅ 大张量sum().item()成功: {large_sum_val}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 简单测试失败: {e}")
        traceback.print_exc()
        return False

def test_mask_operations():
    """测试mask相关操作"""
    print("\n🔧 测试mask相关操作...")
    
    try:
        # 创建类似fg_mask的张量
        batch_size = 1
        n_anchors = 8400
        
        fg_mask = jt.zeros((batch_size, n_anchors), dtype=jt.bool)
        print(f"  fg_mask形状: {fg_mask.shape}")
        
        # 设置一些为True
        fg_mask[0, :10] = True
        
        # 测试sum操作
        total_fg = fg_mask.sum()
        print(f"  fg_mask.sum()形状: {total_fg.shape}, numel: {total_fg.numel()}")
        
        if total_fg.numel() == 1:
            total_fg_val = total_fg.item()
            print(f"  ✅ fg_mask.sum().item()成功: {total_fg_val}")
        else:
            print(f"  ⚠️ fg_mask.sum()不是标量!")
            total_fg_val = int(total_fg.data[0])
            print(f"  使用.data[0]: {total_fg_val}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ mask测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🔧 item()错误详细调试")
    print("=" * 50)
    
    # 测试简单情况
    simple_success = test_simple_case()
    
    # 测试mask操作
    mask_success = test_mask_operations()
    
    # 测试损失函数
    loss_success = test_loss_function()
    
    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"  简单测试: {'✅ 成功' if simple_success else '❌ 失败'}")
    print(f"  Mask测试: {'✅ 成功' if mask_success else '❌ 失败'}")
    print(f"  损失函数测试: {'✅ 成功' if loss_success else '❌ 失败'}")
    
    if simple_success and mask_success and loss_success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️ 存在问题，需要进一步调试")

if __name__ == "__main__":
    main()
