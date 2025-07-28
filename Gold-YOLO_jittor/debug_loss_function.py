#!/usr/bin/env python3
"""
调试损失函数的目标过滤问题
"""

import os
import sys
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model

def debug_loss_function():
    """调试损失函数"""
    print("🔧 调试损失函数的目标过滤问题...")
    
    # 创建模型
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    model.train()
    
    # 创建损失函数
    import importlib.util
    losses_file = os.path.join(os.path.dirname(__file__), 'yolov6', 'models', 'losses.py')
    spec = importlib.util.spec_from_file_location("losses", losses_file)
    losses_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses_module)
    
    loss_fn = losses_module.ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        iou_type='siou',
        loss_weight={
            'class': 1.0,
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 创建简单的输入数据
    print("🔧 创建测试数据...")
    
    # 图像：[1, 3, 640, 640]
    images = jt.randn(1, 3, 640, 640)
    
    # 标签：[1, 6] - [cls, x_center, y_center, width, height, 0]
    targets = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    
    print(f"输入图像形状: {images.shape}")
    print(f"输入标签形状: {targets.shape}")
    print(f"输入标签内容: {targets.numpy()}")
    
    # 模型前向传播
    print("🔧 模型前向传播...")
    predictions = model(images)
    
    print(f"模型输出数量: {len(predictions)}")
    for i, pred in enumerate(predictions):
        if hasattr(pred, 'shape'):
            print(f"输出 {i}: 形状={pred.shape}")
        else:
            print(f"输出 {i}: 类型={type(pred)}")
    
    # 损失计算
    print("🔧 损失计算...")
    try:
        loss, loss_items = loss_fn(predictions, targets, epoch_num=1, step_num=1)
        
        if loss is not None:
            print(f"✅ 损失计算成功: {float(loss.data):.6f}")
            if loss_items is not None:
                print(f"损失详情: {loss_items}")
        else:
            print("❌ 损失计算返回None")
            
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    debug_loss_function()
