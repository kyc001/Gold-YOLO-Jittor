#!/usr/bin/env python3
"""
稳定训练测试
使用更小的学习率，逐步排查问题
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

def pytorch_exact_initialization(model):
    """完全照抄PyTorch版本的初始化"""
    for module in model.modules():
        if hasattr(module, 'initialize_biases'):
            module.initialize_biases()
            break
    return model

def stable_training_test():
    """稳定训练测试"""
    print(f"🔧 稳定训练测试")
    print("=" * 80)
    print(f"使用更小的学习率，逐步排查训练不稳定问题")
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
    model = pytorch_exact_initialization(model)
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
    
    # 测试不同学习率
    learning_rates = [0.001, 0.005, 0.01, 0.02]
    
    for lr in learning_rates:
        print(f"\n🔧 测试学习率: {lr}")
        print("-" * 60)
        
        # 重新创建模型和优化器
        model = create_perfect_gold_yolo_model()
        model = pytorch_exact_initialization(model)
        model.train()
        
        optimizer = jt.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
        
        # 记录训练过程
        loss_history = []
        stable_training = True
        
        print(f"   开始训练 (20轮):")
        
        for epoch in range(20):
            try:
                # 前向传播
                outputs = model(img_tensor)
                
                # 计算损失
                loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
                
                # 检查损失值
                loss_value = float(loss.data.item())
                loss_items_values = [float(item.data.item()) for item in loss_items]
                
                # 检查是否稳定
                if not math.isfinite(loss_value):
                    print(f"     轮次 {epoch+1:2d}: ❌ 损失为NaN或Inf")
                    stable_training = False
                    break
                elif loss_value > 10000:
                    print(f"     轮次 {epoch+1:2d}: ❌ 损失爆炸 ({loss_value:.2f})")
                    stable_training = False
                    break
                elif any(not math.isfinite(v) for v in loss_items_values):
                    print(f"     轮次 {epoch+1:2d}: ❌ 损失项有NaN/Inf")
                    stable_training = False
                    break
                elif any(v > 100000 for v in loss_items_values):
                    print(f"     轮次 {epoch+1:2d}: ❌ 损失项爆炸")
                    stable_training = False
                    break
                
                # 反向传播
                optimizer.zero_grad()
                optimizer.backward(loss)
                
                # 检查梯度
                max_grad_norm = 0.0
                nan_grads = 0
                
                for param in model.parameters():
                    if param.requires_grad:
                        try:
                            grad = param.opt_grad(optimizer)
                            if grad is not None:
                                grad_norm = float(grad.norm())
                                if not math.isfinite(grad_norm):
                                    nan_grads += 1
                                else:
                                    max_grad_norm = max(max_grad_norm, grad_norm)
                        except:
                            pass
                
                if nan_grads > 0:
                    print(f"     轮次 {epoch+1:2d}: ❌ {nan_grads}个梯度为NaN")
                    stable_training = False
                    break
                elif max_grad_norm > 1000:
                    print(f"     轮次 {epoch+1:2d}: ❌ 梯度过大 ({max_grad_norm:.2f})")
                    stable_training = False
                    break
                
                # 更新参数
                optimizer.step()
                
                loss_history.append(loss_value)
                
                # 每5轮打印一次
                if (epoch + 1) % 5 == 0:
                    print(f"     轮次 {epoch+1:2d}: 损失={loss_value:.6f}, 损失项={[f'{x:.4f}' for x in loss_items_values]}, 最大梯度={max_grad_norm:.4f}")
                
            except Exception as e:
                print(f"     轮次 {epoch+1:2d}: ❌ 训练异常: {e}")
                stable_training = False
                break
        
        # 分析结果
        if stable_training and len(loss_history) >= 20:
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100
            
            print(f"   ✅ 训练稳定:")
            print(f"     初始损失: {initial_loss:.6f}")
            print(f"     最终损失: {final_loss:.6f}")
            print(f"     损失下降: {loss_reduction:.1f}%")
            
            # 判断是否有学习效果
            if loss_reduction > 5:
                print(f"     🎉 学习率 {lr} 效果良好！")
                
                # 继续训练更多轮次
                print(f"   继续训练到50轮:")
                
                for epoch in range(20, 50):
                    try:
                        outputs = model(img_tensor)
                        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
                        loss_value = float(loss.data.item())
                        
                        if not math.isfinite(loss_value) or loss_value > 10000:
                            print(f"     轮次 {epoch+1:2d}: ❌ 训练变不稳定")
                            break
                        
                        optimizer.zero_grad()
                        optimizer.backward(loss)
                        optimizer.step()
                        
                        loss_history.append(loss_value)
                        
                        if (epoch + 1) % 10 == 0:
                            print(f"     轮次 {epoch+1:2d}: 损失={loss_value:.6f}")
                    
                    except Exception as e:
                        print(f"     轮次 {epoch+1:2d}: ❌ 训练异常: {e}")
                        break
                
                # 最终结果
                if len(loss_history) >= 50:
                    final_loss = loss_history[-1]
                    total_reduction = (initial_loss - final_loss) / initial_loss * 100
                    print(f"   🎯 50轮训练结果:")
                    print(f"     总损失下降: {total_reduction:.1f}%")
                    print(f"     最终损失: {final_loss:.6f}")
                    
                    if total_reduction > 50:
                        print(f"     🎉🎉🎉 学习率 {lr} 训练成功！")
                        return lr, True
            else:
                print(f"     ⚠️ 学习率 {lr} 学习效果不佳")
        else:
            print(f"   ❌ 学习率 {lr} 训练不稳定")
    
    print(f"\n📊 测试总结:")
    print(f"   所有测试的学习率都有问题")
    print(f"   建议进一步降低学习率或检查损失函数")
    
    return None, False

def main():
    print("🔧 稳定训练测试")
    print("=" * 80)
    
    try:
        best_lr, success = stable_training_test()
        
        if success:
            print(f"\n🎉 找到稳定的学习率: {best_lr}")
            print(f"   可以使用此学习率进行正式训练")
        else:
            print(f"\n❌ 没有找到稳定的学习率")
            print(f"   需要进一步调试模型或损失函数")
            
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
