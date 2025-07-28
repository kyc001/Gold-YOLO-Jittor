#!/usr/bin/env python3
"""
深入调试损失函数的标签预处理流程
找出坐标变成0的具体原因
"""

import os
import sys
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model

def debug_loss_preprocessing():
    """调试损失函数的标签预处理"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              深入调试损失函数标签预处理流程                   ║
    ║                                                              ║
    ║  🔍 逐步跟踪标签处理过程                                     ║
    ║  🎯 找出坐标变成0的具体原因                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建模型和损失函数
    print("🔧 创建模型和损失函数...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    model.train()
    
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
            'class': 5.0,
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 准备测试数据
    print("\n📦 准备测试数据...")
    images = jt.randn(1, 3, 640, 640)
    targets = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    
    print(f"输入图像形状: {images.shape}")
    print(f"输入标签形状: {targets.shape}")
    print(f"输入标签内容: {targets.numpy()}")
    
    # 模型前向传播
    print("\n🔧 模型前向传播...")
    predictions = model(images)

    if isinstance(predictions, (list, tuple)):
        print(f"模型输出数量: {len(predictions)}")
        for i, p in enumerate(predictions):
            if hasattr(p, 'shape'):
                print(f"  输出{i}形状: {p.shape}")
            else:
                print(f"  输出{i}类型: {type(p)}")
    else:
        print(f"模型输出形状: {predictions.shape}")
    
    # 手动调用损失函数的预处理部分
    print("\n🔍 手动调试损失函数预处理...")
    
    # 模拟损失函数内部的处理流程
    print("\n1. 初始标签处理:")
    print(f"   targets形状: {targets.shape}")
    print(f"   targets内容: {targets.numpy()}")
    
    # 检查targets的维度
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)  # 添加batch维度
        print(f"   添加batch维度后: {targets.shape}")
    
    print("\n2. 提取坐标和类别:")
    batch_size = targets.shape[0]
    print(f"   batch_size: {batch_size}")
    
    # 提取各个部分
    gt_cls = targets[:, :, 0:1]  # 类别
    gt_bboxes = targets[:, :, 1:5]  # 坐标 [x_center, y_center, width, height]
    
    print(f"   gt_cls形状: {gt_cls.shape}, 内容: {gt_cls.numpy()}")
    print(f"   gt_bboxes形状: {gt_bboxes.shape}, 内容: {gt_bboxes.numpy()}")
    
    print("\n3. 坐标缩放:")
    ori_img_size = 640
    scale_tensor = jt.array([ori_img_size] * 4, dtype='float32')
    print(f"   scale_tensor: {scale_tensor.numpy()}")
    
    # 应用缩放
    gt_bboxes_scaled = gt_bboxes * scale_tensor
    print(f"   缩放后gt_bboxes: {gt_bboxes_scaled.numpy()}")
    print(f"   缩放后数值范围: [{float(gt_bboxes_scaled.min().numpy()):.6f}, {float(gt_bboxes_scaled.max().numpy()):.6f}]")
    
    print("\n4. 坐标格式转换 (xywh -> xyxy):")
    from yolov6.utils.general import xywh2xyxy
    
    # 复制一份用于转换
    gt_bboxes_xyxy = gt_bboxes_scaled.clone()
    print(f"   转换前: {gt_bboxes_xyxy.numpy()}")
    
    # 应用坐标转换
    gt_bboxes_xyxy = xywh2xyxy(gt_bboxes_xyxy)
    print(f"   转换后: {gt_bboxes_xyxy.numpy()}")
    print(f"   转换后数值范围: [{float(gt_bboxes_xyxy.min().numpy()):.6f}, {float(gt_bboxes_xyxy.max().numpy()):.6f}]")
    
    print("\n5. 检查是否有有效目标:")
    # 检查坐标是否有效
    valid_mask = (gt_bboxes_xyxy[..., 2] > gt_bboxes_xyxy[..., 0]) & (gt_bboxes_xyxy[..., 3] > gt_bboxes_xyxy[..., 1])
    print(f"   有效坐标掩码: {valid_mask.numpy()}")
    
    # 检查类别是否有效
    valid_cls_mask = (gt_cls >= 0) & (gt_cls < 20)
    print(f"   有效类别掩码: {valid_cls_mask.numpy()}")
    
    # 综合有效性
    overall_valid = valid_mask & valid_cls_mask.squeeze(-1)
    print(f"   综合有效掩码: {overall_valid.numpy()}")
    print(f"   有效目标数量: {int(overall_valid.sum().numpy())}")
    
    print("\n6. 模拟损失函数的完整调用:")
    try:
        loss, loss_items = loss_fn(predictions, targets, epoch_num=1, step_num=1)
        
        if loss is not None:
            print(f"   ✅ 损失计算成功: {float(loss.numpy()):.6f}")
            if loss_items is not None:
                print(f"   损失详情: {loss_items}")
        else:
            print(f"   ❌ 损失计算返回None")
            
    except Exception as e:
        print(f"   ❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 问题诊断:")
    
    # 检查各个步骤的结果
    if float(gt_bboxes_scaled.min().numpy()) == 0 and float(gt_bboxes_scaled.max().numpy()) == 0:
        print("   ❌ 问题在步骤3：坐标缩放后变成0")
        print("   🔧 可能原因：输入坐标本身为0或缩放因子有问题")
    elif float(gt_bboxes_xyxy.min().numpy()) == 0 and float(gt_bboxes_xyxy.max().numpy()) == 0:
        print("   ❌ 问题在步骤4：坐标转换后变成0")
        print("   🔧 可能原因：xywh2xyxy函数有问题")
    elif int(overall_valid.sum().numpy()) == 0:
        print("   ❌ 问题在步骤5：所有目标被标记为无效")
        print("   🔧 可能原因：坐标或类别验证逻辑有问题")
    else:
        print("   ✅ 预处理流程正常")
    
    return True

def test_with_different_inputs():
    """测试不同的输入格式"""
    print("\n🔧 测试不同的输入格式...")
    
    # 测试用例1：标准格式
    print("\n测试用例1：标准格式")
    targets1 = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    test_single_input(targets1, "标准格式")
    
    # 测试用例2：小目标
    print("\n测试用例2：小目标")
    targets2 = jt.array([[0, 0.5, 0.5, 0.1, 0.1, 0]], dtype='float32')
    test_single_input(targets2, "小目标")
    
    # 测试用例3：边角目标
    print("\n测试用例3：边角目标")
    targets3 = jt.array([[0, 0.1, 0.1, 0.2, 0.2, 0]], dtype='float32')
    test_single_input(targets3, "边角目标")
    
    # 测试用例4：多个目标
    print("\n测试用例4：多个目标")
    targets4 = jt.array([
        [0, 0.3, 0.3, 0.4, 0.4, 0],
        [1, 0.7, 0.7, 0.4, 0.4, 0]
    ], dtype='float32')
    test_single_input(targets4, "多个目标")

def test_single_input(targets, description):
    """测试单个输入"""
    print(f"   {description}: {targets.numpy()}")
    
    # 坐标缩放
    gt_bboxes = targets[:, 1:5] if targets.ndim == 2 else targets[:, :, 1:5]
    scale_tensor = jt.array([640, 640, 640, 640], dtype='float32')
    gt_bboxes_scaled = gt_bboxes * scale_tensor
    
    # 坐标转换
    from yolov6.utils.general import xywh2xyxy
    gt_bboxes_xyxy = xywh2xyxy(gt_bboxes_scaled.clone())
    
    # 检查有效性
    valid_mask = (gt_bboxes_xyxy[..., 2] > gt_bboxes_xyxy[..., 0]) & (gt_bboxes_xyxy[..., 3] > gt_bboxes_xyxy[..., 1])
    valid_count = int(valid_mask.sum().numpy())
    
    print(f"     缩放后: {gt_bboxes_scaled.numpy()}")
    print(f"     转换后: {gt_bboxes_xyxy.numpy()}")
    print(f"     有效目标数: {valid_count}")

if __name__ == "__main__":
    print("🚀 开始深入调试损失函数标签预处理...")
    
    # 主要调试
    debug_loss_preprocessing()
    
    # 测试不同输入
    test_with_different_inputs()
    
    print("\n🎉 损失函数标签预处理调试完成！")
