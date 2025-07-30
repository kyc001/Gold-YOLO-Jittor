#!/usr/bin/env python3
"""
测试修复后的训练
使用更小的学习率，验证TaskAlignedAssigner修复效果
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import time
import math

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.pytorch_aligned_losses import ComputeLoss

def test_fixed_training():
    """测试修复后的训练"""
    print(f"🔧 测试修复后的训练")
    print("=" * 80)
    print(f"验证TaskAlignedAssigner修复效果")
    print("=" * 80)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    # 读取标注
    annotations = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append([cls_id, x_center, y_center, width, height])
    
    # 读取图像
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    # 准备标签
    targets = []
    for ann in annotations:
        cls_id, x_center, y_center, width, height = ann
        targets.append([0, cls_id, x_center, y_center, width, height])
    targets_tensor = jt.array(targets, dtype=jt.float32).unsqueeze(0)
    
    print(f"📊 数据准备:")
    print(f"   图像张量: {img_tensor.shape}")
    print(f"   标签张量: {targets_tensor.shape}")
    print(f"   目标数量: {len(annotations)}个")
    
    # 创建模型
    print(f"\n🎯 创建模型:")
    model = create_perfect_gold_yolo_model()
    model.train()
    
    # 创建损失函数
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=500,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    
    # 使用小学习率
    lr = 0.01
    optimizer = jt.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    
    print(f"\n🔧 开始训练测试 (学习率: {lr}):")
    print("-" * 60)
    
    # 记录训练过程
    loss_history = []
    
    for epoch in range(10):
        try:
            # 前向传播
            outputs = model(img_tensor)
            
            # 计算损失
            loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
            
            # 检查损失值
            loss_value = float(loss.data.item())
            loss_items_values = [float(item.data.item()) for item in loss_items]
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            loss_history.append(loss_value)
            
            print(f"   轮次 {epoch+1:2d}: 总损失={loss_value:.6f}, IoU={loss_items_values[0]:.4f}, DFL={loss_items_values[1]:.4f}, 分类={loss_items_values[2]:.4f}")
            
            # 检查分类损失是否>0
            if loss_items_values[2] > 0:
                print(f"     ✅ 分类损失>0，目标分配正常！")
            else:
                print(f"     ❌ 分类损失=0，目标分配仍有问题")
            
        except Exception as e:
            print(f"     轮次 {epoch+1:2d}: ❌ 训练异常: {e}")
            break
    
    # 分析结果
    if len(loss_history) >= 10:
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\n📊 训练结果:")
        print(f"   初始损失: {initial_loss:.6f}")
        print(f"   最终损失: {final_loss:.6f}")
        print(f"   损失下降: {loss_reduction:.1f}%")
        
        if loss_reduction > 10:
            print(f"   🎉 训练效果良好！")
            
            # 检查分类损失
            final_cls_loss = loss_items_values[2]
            if final_cls_loss > 0:
                print(f"   🎉 分类损失正常: {final_cls_loss:.6f}")
                print(f"   ✅ TaskAlignedAssigner修复成功！")
                return True
            else:
                print(f"   ❌ 分类损失仍为0，需要进一步修复")
                return False
        else:
            print(f"   ⚠️ 训练效果不佳")
            return False
    else:
        print(f"\n❌ 训练失败")
        return False

def main():
    print("🔧 测试修复后的训练")
    print("=" * 80)
    
    try:
        success = test_fixed_training()
        
        if success:
            print(f"\n🎉 修复成功！可以进行正式训练了！")
            print(f"   建议使用学习率0.01进行单张图片过拟合测试")
        else:
            print(f"\n❌ 修复未完成，需要继续调试TaskAlignedAssigner")
            
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
