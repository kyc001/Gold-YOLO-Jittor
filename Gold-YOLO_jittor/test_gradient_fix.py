#!/usr/bin/env python3
"""
测试梯度爆炸修复效果
验证VarifocalLoss修复后的梯度是否正常
"""

import os
import sys
import numpy as np
import jittor as jt
from jittor import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model

def test_gradient_fix():
    """测试梯度爆炸修复效果"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              测试梯度爆炸修复效果                             ║
    ║                                                              ║
    ║  🔧 验证VarifocalLoss修复后的梯度是否正常                    ║
    ║  📊 对比修复前后的梯度大小                                   ║
    ║  🎯 确保分类头能够正常训练                                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建模型
    print("🔧 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                jt.init.constant_(module.bias, -2.0)
    
    # 创建修复后的损失函数
    print("🔧 创建修复后的损失函数...")
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
    
    # 创建测试数据
    print("🔧 创建测试数据...")
    images = jt.randn(1, 3, 640, 640)
    targets = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    
    # 创建优化器
    optimizer = nn.SGD(model.parameters(), lr=0.01)
    
    print("🔍 测试修复后的梯度...")
    
    # 前向传播
    model.train()
    outputs = model(images)
    
    # 计算损失
    loss, loss_items = loss_fn(outputs, targets, epoch_num=1, step_num=1)
    
    print(f"修复后损失计算结果:")
    if loss is not None:
        print(f"   总损失: {float(loss.numpy()):.6f}")
        if loss_items is not None:
            try:
                loss_values = [float(item.numpy()) for item in loss_items]
                print(f"   损失详情: {loss_values}")
                print(f"   分类损失: {loss_values[0]:.6f}")
                print(f"   IoU损失: {loss_values[1]:.6f}")
                print(f"   DFL损失: {loss_values[2]:.6f}")
            except:
                print(f"   损失详情: {loss_items}")
    else:
        print(f"   ❌ 损失为None")
        return False
    
    # 计算梯度
    print(f"\n🔍 分析修复后的梯度...")
    optimizer.zero_grad()
    optimizer.backward(loss)
    
    # 检查分类头梯度
    gradient_normal = True
    max_gradient = 0.0
    
    for name, param in model.named_parameters():
        if 'cls_pred' in name:
            grad = param.opt_grad(optimizer)
            if grad is not None:
                grad_min = float(grad.min().numpy())
                grad_max = float(grad.max().numpy())
                grad_mean = float(grad.mean().numpy())
                grad_std = float(grad.std().numpy())
                grad_abs_max = max(abs(grad_min), abs(grad_max))
                max_gradient = max(max_gradient, grad_abs_max)
                
                print(f"   {name}:")
                print(f"     梯度范围: [{grad_min:.6f}, {grad_max:.6f}]")
                print(f"     梯度均值: {grad_mean:.6f}")
                print(f"     梯度标准差: {grad_std:.6f}")
                print(f"     梯度绝对值最大: {grad_abs_max:.6f}")
                
                # 检查梯度是否正常
                if grad_abs_max > 10.0:
                    print(f"     ❌ 梯度仍然过大！")
                    gradient_normal = False
                elif grad_abs_max > 1.0:
                    print(f"     ⚠️ 梯度偏大但可接受")
                else:
                    print(f"     ✅ 梯度正常")
                
                # 检查梯度方向分布
                grad_numpy = grad.numpy().flatten()
                positive_ratio = np.sum(grad_numpy > 0) / len(grad_numpy) * 100
                negative_ratio = np.sum(grad_numpy < 0) / len(grad_numpy) * 100
                zero_ratio = np.sum(grad_numpy == 0) / len(grad_numpy) * 100
                
                print(f"     梯度方向分布: 正值{positive_ratio:.1f}%, 负值{negative_ratio:.1f}%, 零值{zero_ratio:.1f}%")
                
                # 检查梯度方向是否合理
                if positive_ratio == 100.0 or negative_ratio == 100.0:
                    print(f"     ⚠️ 梯度方向单一，可能有问题")
                else:
                    print(f"     ✅ 梯度方向分布合理")
    
    print(f"\n📊 梯度修复效果总结:")
    print(f"   最大梯度绝对值: {max_gradient:.6f}")
    
    if gradient_normal and max_gradient < 10.0:
        print(f"   ✅ 梯度爆炸问题已修复！")
        print(f"   🎯 分类头现在可以正常训练")
        return True
    elif max_gradient < 100.0:
        print(f"   ⚠️ 梯度有所改善但仍需优化")
        return None
    else:
        print(f"   ❌ 梯度爆炸问题仍然存在")
        return False

def test_training_stability():
    """测试训练稳定性"""
    print(f"\n🔧 测试训练稳定性...")
    
    # 创建模型
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                jt.init.constant_(module.bias, -2.0)
    
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
    
    # 创建优化器
    optimizer = nn.SGD(model.parameters(), lr=0.01)
    
    # 创建测试数据
    images = jt.randn(1, 3, 640, 640)
    targets = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    
    print(f"进行10步训练测试...")
    
    model.train()
    loss_history = []
    cls_output_history = []
    
    for step in range(10):
        # 前向传播
        outputs = model(images)
        
        # 记录分类输出
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            cls_output = outputs[1]  # [1, 8400, 20]
            cls_min = float(cls_output.min().numpy())
            cls_max = float(cls_output.max().numpy())
            cls_range = cls_max - cls_min
            cls_output_history.append(cls_range)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets, epoch_num=step+1, step_num=1)
        
        if loss is not None:
            loss_value = float(loss.numpy())
            loss_history.append(loss_value)
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            print(f"   步骤{step+1}: 损失={loss_value:.6f}, 分类范围={cls_range:.6f}")
        else:
            print(f"   步骤{step+1}: 损失为None")
            return False
    
    # 分析训练稳定性
    print(f"\n📊 训练稳定性分析:")
    print(f"   初始损失: {loss_history[0]:.6f}")
    print(f"   最终损失: {loss_history[-1]:.6f}")
    print(f"   初始分类范围: {cls_output_history[0]:.6f}")
    print(f"   最终分类范围: {cls_output_history[-1]:.6f}")
    
    # 检查是否稳定
    if cls_output_history[-1] > 0.001:
        print(f"   ✅ 分类输出保持变化，训练稳定")
        return True
    else:
        print(f"   ❌ 分类输出趋向于0，仍有问题")
        return False

if __name__ == "__main__":
    print("🚀 开始测试梯度爆炸修复效果...")
    
    # 测试梯度修复
    gradient_result = test_gradient_fix()
    
    # 测试训练稳定性
    stability_result = test_training_stability()
    
    print("\n" + "="*70)
    print("🎉 梯度爆炸修复测试完成！")
    print("="*70)
    
    if gradient_result is True and stability_result is True:
        print("✅ 梯度爆炸问题完全修复！")
        print("🎯 分类头现在可以正常训练")
        print("📋 建议进行完整的500轮训练验证")
    elif gradient_result is not False and stability_result is not False:
        print("⚠️ 梯度问题有所改善，但需要进一步优化")
        print("🔧 建议调整学习率或损失权重")
    else:
        print("❌ 梯度爆炸问题仍然存在")
        print("🔧 需要进一步修复损失函数")
    
    print(f"\n📊 测试结果:")
    print(f"   梯度修复: {'✅ 成功' if gradient_result is True else '⚠️ 部分' if gradient_result is None else '❌ 失败'}")
    print(f"   训练稳定性: {'✅ 稳定' if stability_result is True else '❌ 不稳定'}")
