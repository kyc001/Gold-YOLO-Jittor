#!/usr/bin/env python3
"""
深入分析分类头被训练坏的根本原因
不绕开任何问题，彻底解决分类头问题
"""

import os
import sys
import numpy as np
import jittor as jt
from jittor import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model

def deep_analyze_classification_loss():
    """深入分析分类损失计算"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              深入分析分类头被训练坏的根本原因                 ║
    ║                                                              ║
    ║  🔍 分析分类损失的计算过程                                   ║
    ║  🎯 找出导致分类头输出变为0的具体原因                        ║
    ║  🔧 不绕开任何问题，彻底解决                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建模型
    print("🔧 创建模型进行深入分析...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                jt.init.constant_(module.bias, -2.0)
    
    # 创建损失函数
    print("🔧 创建损失函数...")
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
    
    print(f"测试图像形状: {images.shape}")
    print(f"测试标签形状: {targets.shape}")
    print(f"测试标签内容: {targets.numpy()}")
    
    # 分析初始状态
    print("\n🔍 分析初始状态...")
    model.eval()
    with jt.no_grad():
        initial_outputs = model(images)
    
    if isinstance(initial_outputs, (list, tuple)) and len(initial_outputs) >= 2:
        initial_cls = initial_outputs[1]  # [1, 8400, 20]
        initial_cls_min = float(initial_cls.min().numpy())
        initial_cls_max = float(initial_cls.max().numpy())
        initial_cls_mean = float(initial_cls.mean().numpy())
        initial_cls_range = initial_cls_max - initial_cls_min
        
        print(f"初始分类输出统计:")
        print(f"   范围: [{initial_cls_min:.6f}, {initial_cls_max:.6f}]")
        print(f"   均值: {initial_cls_mean:.6f}")
        print(f"   变化范围: {initial_cls_range:.6f}")
        
        # 分析分类输出的分布
        cls_numpy = initial_cls.numpy().flatten()
        print(f"   标准差: {np.std(cls_numpy):.6f}")
        print(f"   零值比例: {np.sum(cls_numpy == 0) / len(cls_numpy) * 100:.2f}%")
        print(f"   正值比例: {np.sum(cls_numpy > 0) / len(cls_numpy) * 100:.2f}%")
        print(f"   负值比例: {np.sum(cls_numpy < 0) / len(cls_numpy) * 100:.2f}%")
    
    # 分析损失计算过程
    print("\n🔍 分析损失计算过程...")
    model.train()
    
    # 前向传播
    outputs = model(images)
    print(f"训练模式输出类型: {type(outputs)}")
    print(f"训练模式输出数量: {len(outputs)}")
    
    # 手动调用损失函数并监控内部过程
    print("\n🔍 手动调用损失函数...")
    
    # 修改损失函数以添加调试信息
    original_forward = loss_fn.__call__
    
    def debug_loss_forward(predictions, targets, epoch_num, step_num):
        print(f"\n🔍 损失函数内部调试:")
        print(f"   predictions类型: {type(predictions)}")
        print(f"   predictions长度: {len(predictions) if isinstance(predictions, (list, tuple)) else 'N/A'}")
        print(f"   targets形状: {targets.shape}")
        
        # 调用原始损失函数
        result = original_forward(predictions, targets, epoch_num, step_num)
        
        return result
    
    loss_fn.__call__ = debug_loss_forward
    
    # 计算损失
    loss, loss_items = loss_fn(outputs, targets, epoch_num=1, step_num=1)
    
    print(f"\n🔍 损失计算结果:")
    if loss is not None:
        print(f"   总损失: {float(loss.numpy()):.6f}")
        if loss_items is not None:
            try:
                print(f"   损失详情: {[float(item.numpy()) for item in loss_items]}")
            except:
                print(f"   损失详情: {loss_items}")
    else:
        print(f"   ❌ 损失为None")
    
    # 分析梯度
    print("\n🔍 分析梯度...")
    
    # 创建优化器
    optimizer = nn.SGD(model.parameters(), lr=0.01)
    
    if loss is not None:
        # 计算梯度
        optimizer.zero_grad()
        optimizer.backward(loss)
        
        # 分析分类头的梯度
        print(f"分类头梯度分析:")
        for name, param in model.named_parameters():
            if 'cls_pred' in name:
                grad = param.opt_grad(optimizer)
                if grad is not None:
                    grad_min = float(grad.min().numpy())
                    grad_max = float(grad.max().numpy())
                    grad_mean = float(grad.mean().numpy())
                    grad_std = float(grad.std().numpy())
                    
                    print(f"   {name}:")
                    print(f"     梯度范围: [{grad_min:.6f}, {grad_max:.6f}]")
                    print(f"     梯度均值: {grad_mean:.6f}")
                    print(f"     梯度标准差: {grad_std:.6f}")
                    
                    # 检查梯度是否异常
                    if abs(grad_mean) > 1.0:
                        print(f"     ⚠️ 梯度均值过大！")
                    if grad_std > 1.0:
                        print(f"     ⚠️ 梯度标准差过大！")
                    if grad_min == grad_max == 0.0:
                        print(f"     ❌ 梯度全为0！")
                    
                    # 分析梯度方向
                    grad_numpy = grad.numpy().flatten()
                    positive_ratio = np.sum(grad_numpy > 0) / len(grad_numpy) * 100
                    negative_ratio = np.sum(grad_numpy < 0) / len(grad_numpy) * 100
                    zero_ratio = np.sum(grad_numpy == 0) / len(grad_numpy) * 100
                    
                    print(f"     梯度方向分布: 正值{positive_ratio:.1f}%, 负值{negative_ratio:.1f}%, 零值{zero_ratio:.1f}%")
                else:
                    print(f"   {name}: 梯度为None")
        
        # 执行一步优化
        print(f"\n🔍 执行一步优化...")
        optimizer.step()
        
        # 检查优化后的分类输出
        model.eval()
        with jt.no_grad():
            after_outputs = model(images)
        
        if isinstance(after_outputs, (list, tuple)) and len(after_outputs) >= 2:
            after_cls = after_outputs[1]  # [1, 8400, 20]
            after_cls_min = float(after_cls.min().numpy())
            after_cls_max = float(after_cls.max().numpy())
            after_cls_mean = float(after_cls.mean().numpy())
            after_cls_range = after_cls_max - after_cls_min
            
            print(f"优化后分类输出统计:")
            print(f"   范围: [{after_cls_min:.6f}, {after_cls_max:.6f}]")
            print(f"   均值: {after_cls_mean:.6f}")
            print(f"   变化范围: {after_cls_range:.6f}")
            
            # 对比优化前后的变化
            range_change = after_cls_range - initial_cls_range
            mean_change = after_cls_mean - initial_cls_mean
            
            print(f"\n📊 优化前后对比:")
            print(f"   变化范围变化: {range_change:.6f}")
            print(f"   均值变化: {mean_change:.6f}")
            
            if range_change < -0.001:
                print(f"   ❌ 变化范围显著减小，分类头正在被训练坏！")
                return False
            elif abs(range_change) < 0.0001:
                print(f"   ⚠️ 变化范围几乎不变，可能学习率过小")
                return None
            else:
                print(f"   ✅ 变化范围正常变化")
                return True
    
    return None

def analyze_loss_function_internals():
    """分析损失函数内部实现"""
    print("\n🔍 分析损失函数内部实现...")
    
    # 检查损失函数的分类损失计算部分
    losses_file = os.path.join(os.path.dirname(__file__), 'yolov6', 'models', 'losses.py')
    
    print(f"损失函数文件: {losses_file}")
    
    # 查找分类损失相关的代码
    with open(losses_file, 'r') as f:
        content = f.read()
    
    # 查找关键函数
    if 'class_loss' in content:
        print("✅ 找到class_loss相关代码")
    else:
        print("❌ 未找到class_loss相关代码")
    
    if 'sigmoid' in content:
        print("✅ 找到sigmoid相关代码")
    else:
        print("❌ 未找到sigmoid相关代码")
    
    if 'bce' in content.lower():
        print("✅ 找到BCE损失相关代码")
    else:
        print("❌ 未找到BCE损失相关代码")
    
    # 检查是否有标签平滑或其他可能导致问题的设置
    if 'label_smooth' in content:
        print("⚠️ 发现标签平滑设置")
    
    if 'focal' in content.lower():
        print("⚠️ 发现Focal Loss设置")
    
    return True

def propose_solutions():
    """提出解决方案"""
    print("\n🎯 提出解决方案:")
    print("基于深入分析，可能的解决方案包括:")
    print("1. 修改分类损失的计算方式")
    print("2. 调整分类头的激活函数")
    print("3. 修改标签的编码方式")
    print("4. 使用不同的损失函数")
    print("5. 调整损失权重的平衡")
    print("6. 修改优化器的设置")
    print("7. 使用梯度累积或其他训练技巧")

if __name__ == "__main__":
    print("🚀 开始深入分析分类头问题...")
    
    # 1. 深入分析分类损失
    print("\n" + "="*70)
    print("步骤1：深入分析分类损失计算")
    print("="*70)
    result = deep_analyze_classification_loss()
    
    # 2. 分析损失函数内部
    print("\n" + "="*70)
    print("步骤2：分析损失函数内部实现")
    print("="*70)
    analyze_loss_function_internals()
    
    # 3. 提出解决方案
    print("\n" + "="*70)
    print("步骤3：提出解决方案")
    print("="*70)
    propose_solutions()
    
    print("\n" + "="*70)
    print("🎉 深入分析完成！")
    print("="*70)
    
    if result is False:
        print("❌ 确认分类头正在被训练坏")
        print("🔧 需要立即修复损失函数或训练策略")
    elif result is True:
        print("✅ 分类头训练正常")
        print("🎯 问题可能在其他地方")
    else:
        print("⚠️ 需要进一步调查")
    
    print("\n📋 下一步行动:")
    print("   1. 根据分析结果修复具体问题")
    print("   2. 实施最有效的解决方案")
    print("   3. 验证修复效果")
    print("   4. 确保分类头能够正常学习")
