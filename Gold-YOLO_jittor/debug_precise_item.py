#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
精确调试item()错误的脚本
"""

import jittor as jt
import numpy as np
import traceback

# Set Jittor flags
jt.flags.use_cuda = 1

# 重写item方法来捕获错误
original_item = jt.Var.item

def debug_item(self):
    """调试版本的item方法"""
    if self.numel() != 1:
        print(f"❌ 错误的item()调用: 张量形状={self.shape}, numel={self.numel()}")
        print("调用栈:")
        traceback.print_stack()
        raise RuntimeError(f"Item var size should be 1, but got {self.numel()}")
    return original_item(self)

# 替换item方法
jt.Var.item = debug_item

def test_loss_function_debug():
    """调试损失函数"""
    print("🔧 调试损失函数中的item()调用...")
    
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
        loss, loss_items = criterion(outputs, targets, epoch_num=1, step_num=0)
        
        print(f"  ✅ 损失函数调用成功: {loss}")
        return True
        
    except Exception as e:
        print(f"  ❌ 损失函数调用失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🔧 精确调试item()错误")
    print("=" * 50)
    
    success = test_loss_function_debug()
    
    print("\n" + "=" * 50)
    print(f"测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
