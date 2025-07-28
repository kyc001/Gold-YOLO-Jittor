#!/usr/bin/env python3
"""
调试.item()错误的最小化测试脚本
"""

import jittor as jt
import numpy as np
import sys
import os

# 添加路径
sys.path.append('.')
sys.path.append('..')

def test_loss_function():
    """测试损失函数中的.item()调用"""
    print("🔍 开始测试损失函数...")
    
    try:
        from yolov6.models.losses import ComputeLoss
        
        # 创建损失函数
        loss_fn = ComputeLoss(
            num_classes=20,
            ori_img_size=640,
            warmup_epoch=4,
            use_dfl=True,
            reg_max=16,
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        
        print("✅ 损失函数创建成功")
        
        # 创建虚拟输出
        batch_size = 1
        feats = [
            jt.randn((batch_size, 32, 80, 80)),   # stride 8
            jt.randn((batch_size, 64, 40, 40)),   # stride 16  
            jt.randn((batch_size, 128, 20, 20))   # stride 32
        ]
        
        pred_scores = jt.randn((batch_size, 8400, 20))
        pred_distri = jt.randn((batch_size, 8400, 68))  # 4 * (reg_max + 1)
        
        outputs = [feats, pred_scores, pred_distri]
        
        # 创建虚拟目标 - 这里可能是问题所在
        targets = jt.array([[0, 1, 0.5, 0.5, 0.3, 0.3, 0]])  # [batch_idx, class, x, y, w, h, extra]
        
        print(f"🔍 targets形状: {targets.shape}")
        print(f"🔍 targets内容: {targets.numpy()}")
        
        # 调用损失函数 - 这里应该会触发错误
        print("🔍 开始调用损失函数...")
        loss, loss_items = loss_fn(outputs, targets, epoch_num=0, step_num=0)
        
        print(f"✅ 损失计算成功: {loss}")
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_preprocess_function():
    """单独测试preprocess函数"""
    print("\n🔍 开始测试preprocess函数...")
    
    try:
        from yolov6.models.losses import ComputeLoss
        
        loss_fn = ComputeLoss(
            num_classes=20,
            ori_img_size=640,
            warmup_epoch=4,
            use_dfl=True,
            reg_max=16,
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        
        # 测试不同的targets
        test_cases = [
            jt.array([[0, 1, 0.5, 0.5, 0.3, 0.3, 0]]),  # 1个目标
            jt.array([[0, 1, 0.5, 0.5, 0.3, 0.3, 0], [0, 2, 0.7, 0.7, 0.2, 0.2, 0]]),  # 2个目标
            jt.array([]),  # 空目标
        ]
        
        for i, targets in enumerate(test_cases):
            print(f"\n🔍 测试用例 {i+1}: targets形状 {targets.shape}")
            try:
                batch_size = 1
                scale_tensor = jt.array([640, 640, 640, 640], dtype='float32')
                
                result = loss_fn.preprocess(targets, batch_size, scale_tensor)
                print(f"✅ preprocess成功: 结果形状 {result.shape}")
                
            except Exception as e:
                print(f"❌ preprocess失败: {e}")
                import traceback
                traceback.print_exc()
                break
                
    except Exception as e:
        print(f"❌ preprocess测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 开始调试.item()错误...")
    
    # 首先测试preprocess函数
    test_preprocess_function()
    
    # 然后测试完整的损失函数
    test_loss_function()
    
    print("🏁 调试完成")
