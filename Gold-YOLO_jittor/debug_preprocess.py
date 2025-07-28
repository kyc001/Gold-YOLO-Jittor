#!/usr/bin/env python3
"""
调试preprocess函数的数据重组过程
"""

import jittor as jt
import numpy as np
import sys
import os

# 添加路径
sys.path.append('.')
sys.path.append('..')

def test_preprocess():
    """测试preprocess函数的数据重组"""
    print("🔍 开始测试preprocess函数...")
    
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
        
        # 创建测试数据 - 模拟真实的数据格式
        # 格式: [batch_idx, class, x, y, w, h, extra]
        test_targets = jt.array([
            [0, 10, 0.5, 0.5, 0.2, 0.3, 0],  # 第一个目标
            [0, 14, 0.7, 0.3, 0.1, 0.2, 0],  # 第二个目标
        ])
        
        print(f"🔍 输入targets形状: {test_targets.shape}")
        print(f"🔍 输入targets内容: {test_targets.numpy()}")
        
        # 调用preprocess函数
        batch_size = 1
        scale_tensor = jt.array([640, 640, 640, 640], dtype='float32')
        
        result = loss_fn.preprocess(test_targets, batch_size, scale_tensor)
        
        print(f"🔍 输出结果形状: {result.shape}")
        print(f"🔍 输出结果内容: {result.numpy()}")
        
        # 分析结果
        print(f"\n📊 结果分析:")
        print(f"   类别: {result[0, :, 0].numpy()}")
        print(f"   坐标: {result[0, :, 1:].numpy()}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocess()
