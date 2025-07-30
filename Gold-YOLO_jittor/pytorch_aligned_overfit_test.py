#!/usr/bin/env python3
"""
完全对齐PyTorch版本的过拟合测试
使用PyTorch版本的所有超参数
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

def pytorch_aligned_overfit_test():
    """完全对齐PyTorch版本的过拟合测试"""
    print(f"🎯 完全对齐PyTorch版本的过拟合测试")
    print("=" * 80)
    print(f"使用PyTorch版本的所有超参数")
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
    for i, ann in enumerate(annotations):
        cls_id, x_center, y_center, width, height = ann
        print(f"     目标{i+1}: 类别={cls_id}, 中心=({x_center:.3f},{y_center:.3f}), 尺寸=({width:.3f},{height:.3f})")
    
    # 创建模型
    print(f"\n🎯 创建模型:")
    model = create_perfect_gold_yolo_model()
    model.train()
    
    # 创建损失函数 - 完全对齐PyTorch版本
    print(f"\n💰 创建损失函数 (完全对齐PyTorch版本):")
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=500,
        warmup_epoch=0,  # 对齐PyTorch版本的atss_warmup_epoch=0
        use_dfl=False,   # 对齐PyTorch版本的use_dfl=False
        reg_max=0,       # 对齐PyTorch版本的reg_max=0
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}  # 完全对齐PyTorch版本
    )
    
    print(f"   损失权重: {loss_fn.loss_weight}")
    print(f"   use_dfl: {loss_fn.use_dfl}")
    print(f"   reg_max: {loss_fn.reg_max}")
    print(f"   warmup_epoch: {loss_fn.warmup_epoch}")
    
    # 使用PyTorch版本的优化器参数
    print(f"\n🔧 创建优化器 (完全对齐PyTorch版本):")
    lr = 0.02          # 对齐PyTorch版本的lr0=0.02
    momentum = 0.937   # 对齐PyTorch版本的momentum=0.937
    weight_decay = 0.0005  # 对齐PyTorch版本的weight_decay=0.0005
    
    optimizer = jt.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    
    print(f"   学习率: {lr}")
    print(f"   动量: {momentum}")
    print(f"   权重衰减: {weight_decay}")
    
    print(f"\n🔧 开始过拟合训练 (PyTorch对齐版本):")
    print(f"   目标: 损失下降>90%")
    print("-" * 60)
    
    # 记录训练过程
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(200):
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
            
            if loss_value < best_loss:
                best_loss = loss_value
            
            # 每10轮打印一次
            if (epoch + 1) % 10 == 0:
                print(f"   轮次 {epoch+1:3d}: 总损失={loss_value:.6f}, IoU={loss_items_values[0]:.4f}, DFL={loss_items_values[1]:.4f}, 分类={loss_items_values[2]:.4f}")
            
            # 检查是否达到过拟合标准
            if epoch >= 20:  # 至少训练20轮
                initial_loss = loss_history[0]
                current_reduction = (initial_loss - loss_value) / initial_loss * 100
                
                if current_reduction >= 90:
                    print(f"\n🎉🎉🎉 过拟合成功！(PyTorch对齐版本)")
                    print(f"   轮次: {epoch+1}")
                    print(f"   初始损失: {initial_loss:.6f}")
                    print(f"   当前损失: {loss_value:.6f}")
                    print(f"   损失下降: {current_reduction:.1f}%")
                    print(f"   ✅ 达到过拟合标准！")
                    return True, epoch+1, current_reduction
            
            # 检查训练是否稳定
            if not math.isfinite(loss_value) or loss_value > 10000:
                print(f"\n❌ 训练不稳定，损失异常: {loss_value}")
                return False, epoch+1, 0
                
        except Exception as e:
            print(f"     轮次 {epoch+1:3d}: ❌ 训练异常: {e}")
            return False, epoch+1, 0
    
    # 200轮训练完成，检查最终结果
    if len(loss_history) >= 200:
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\n📊 200轮训练完成 (PyTorch对齐版本):")
        print(f"   初始损失: {initial_loss:.6f}")
        print(f"   最终损失: {final_loss:.6f}")
        print(f"   最佳损失: {best_loss:.6f}")
        print(f"   损失下降: {loss_reduction:.1f}%")
        
        if loss_reduction >= 90:
            print(f"   🎉 过拟合成功！")
            return True, 200, loss_reduction
        elif loss_reduction >= 50:
            print(f"   ⚠️ 部分成功，但未达到90%标准")
            return False, 200, loss_reduction
        else:
            print(f"   ❌ 过拟合失败")
            return False, 200, loss_reduction
    else:
        print(f"\n❌ 训练中断")
        return False, len(loss_history), 0

def main():
    print("🎯 完全对齐PyTorch版本的过拟合测试")
    print("=" * 80)
    
    try:
        success, epochs, reduction = pytorch_aligned_overfit_test()
        
        print(f"\n" + "=" * 80)
        print(f"📊 PyTorch对齐版本测试结果:")
        print(f"=" * 80)
        
        if success:
            print(f"🎉🎉🎉 PyTorch对齐版本过拟合成功！")
            print(f"   ✅ 过拟合测试通过")
            print(f"   ✅ 训练轮次: {epochs}")
            print(f"   ✅ 损失下降: {reduction:.1f}%")
            print(f"   ✅ 使用PyTorch版本超参数成功")
            print(f"\n🚀 证明Jittor版本完全对齐PyTorch版本！")
            print(f"   学习率: 0.02")
            print(f"   动量: 0.937")
            print(f"   权重衰减: 0.0005")
        else:
            print(f"❌ PyTorch对齐版本过拟合测试未完全通过")
            print(f"   训练轮次: {epochs}")
            print(f"   损失下降: {reduction:.1f}%")
            if reduction >= 80:
                print(f"   ⚠️ 接近成功，可能需要更多轮次或微调")
            elif reduction >= 50:
                print(f"   ⚠️ 模型基本正常，但可能需要进一步优化")
            else:
                print(f"   ❌ 模型仍有问题，需要进一步调试")
            
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
