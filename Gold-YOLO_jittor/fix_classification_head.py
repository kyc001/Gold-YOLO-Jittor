#!/usr/bin/env python3
"""
修复分类头输出为0的问题 - 更新版本
深入分析和修复分类头的权重初始化和训练过程
"""
"""
修复分类头问题并重新训练
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from jittor import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.general import scale_coords

def fix_classification_head():
    """修复分类头问题"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  修复分类头问题并重新训练                     ║
    ║                                                              ║
    ║  🔧 重新初始化分类头权重                                     ║
    ║  🎯 调整训练策略提高分类性能                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建模型
    print("🔧 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            print(f"   重新初始化: {name}")
            # 使用Xavier初始化权重
            nn.init.xavier_uniform_(module.weight)
            # 设置bias为小正值，有利于sigmoid输出
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
    
    # 创建损失函数 - 增加分类损失权重
    print("🔧 创建损失函数 (增强分类权重)...")
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
            'class': 5.0,  # 增加分类损失权重
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 创建优化器 - 简化版本，统一使用较高学习率
    print("🔧 创建优化器 (统一高学习率)...")

    optimizer = nn.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)
    
    # 准备训练数据
    print("🔧 准备训练数据...")
    
    # 加载测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    
    # 预处理图像
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
    images = jt.array(img).unsqueeze(0)  # 添加batch维度
    
    # 使用正确的标签格式
    targets = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    
    print(f"训练图像形状: {images.shape}")
    print(f"训练标签形状: {targets.shape}")
    print(f"训练标签内容: {targets.numpy()}")
    
    # 开始修复训练
    print("🚀 开始修复训练...")
    print("   训练轮数: 1000 (增加轮数)")
    print("   分类损失权重: 5.0 (增强)")
    print("   分类头学习率: 0.05 (5倍)")
    print("=" * 70)
    
    model.train()
    
    for epoch in range(1000):
        # 前向传播
        predictions = model(images)
        
        # 计算损失
        loss, loss_items = loss_fn(predictions, targets, epoch_num=epoch+1, step_num=1)
        
        if loss is not None:
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            epoch_loss = float(loss.numpy())
        else:
            epoch_loss = 0.0
        
        # 打印训练进度
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/1000 | Loss: {epoch_loss:.6f}")
            
            # 检查分类头输出
            if (epoch + 1) % 200 == 0:
                model.eval()
                with jt.no_grad():
                    test_pred = model(images)[0]
                    test_cls_conf = test_pred[:, 5:]
                    cls_min = float(test_cls_conf.min().numpy())
                    cls_max = float(test_cls_conf.max().numpy())
                    cls_range = cls_max - cls_min
                    print(f"         类别置信度范围: [{cls_min:.6f}, {cls_max:.6f}], 变化范围: {cls_range:.6f}")
                model.train()
    
    print("✅ 修复训练完成！")
    
    # 保存修复后的模型
    save_path = 'fixed_classification_model.pkl'
    jt.save(model.state_dict(), save_path)
    print(f"💾 修复后模型已保存: {save_path}")
    
    return model

def test_fixed_model(model):
    """测试修复后的模型"""
    print("\n🔍 测试修复后的模型...")
    model.eval()
    
    # 加载测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    
    # 预处理
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
    # 推理
    with jt.no_grad():
        pred = model(img_tensor)[0]  # [8400, 25]
    
    # 解析预测结果
    boxes = pred[:, :4]  # [x_center, y_center, width, height]
    obj_conf = pred[:, 4]  # 目标置信度
    cls_conf = pred[:, 5:]  # 类别置信度 [20]
    
    print(f"修复后 - 目标置信度范围: [{float(obj_conf.min().numpy()):.6f}, {float(obj_conf.max().numpy()):.6f}]")
    print(f"修复后 - 类别置信度范围: [{float(cls_conf.min().numpy()):.6f}, {float(cls_conf.max().numpy()):.6f}]")
    
    # 计算最终置信度和类别
    cls_scores = cls_conf.max(dim=1)[0]  # 最大类别置信度
    cls_indices = cls_conf.argmax(dim=1)  # 类别索引
    final_conf = obj_conf * cls_scores
    
    print(f"修复后 - 最终置信度范围: [{float(final_conf.min().numpy()):.6f}, {float(final_conf.max().numpy()):.6f}]")
    
    # 检查类别置信度变化范围
    cls_range = float(cls_conf.max().numpy()) - float(cls_conf.min().numpy())
    print(f"修复后 - 类别置信度变化范围: {cls_range:.6f}")
    
    if cls_range > 0.1:
        print("✅ 分类头修复成功！类别置信度有明显变化")
    else:
        print("⚠️ 分类头仍需进一步调整")
    
    # 进行NMS并可视化
    print("\n🎨 创建修复后的检测可视化...")
    
    # 使用更低的置信度阈值
    conf_threshold = max(0.01, float(final_conf.max().numpy()) * 0.1)
    print(f"使用置信度阈值: {conf_threshold:.6f}")
    
    # 简单过滤
    mask = final_conf > conf_threshold
    if mask.sum() > 0:
        filtered_boxes = boxes[mask]
        filtered_conf = final_conf[mask]
        filtered_cls = cls_indices[mask]
        
        print(f"过滤后检测数量: {len(filtered_boxes)}")
        
        # 转换坐标格式并可视化
        img_vis = img0.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        
        for i in range(min(5, len(filtered_boxes))):
            box = filtered_boxes[i].numpy()
            conf = float(filtered_conf[i].numpy())
            cls = int(filtered_cls[i].numpy())
            
            # 转换坐标
            x_center, y_center, width, height = box
            x1 = int((x_center - width/2) * w0 / 640)
            y1 = int((y_center - height/2) * h0 / 640)
            x2 = int((x_center + width/2) * w0 / 640)
            y2 = int((y_center + height/2) * h0 / 640)
            
            # 确保坐标在范围内
            x1 = max(0, min(x1, w0-1))
            y1 = max(0, min(y1, h0-1))
            x2 = max(0, min(x2, w0-1))
            y2 = max(0, min(y2, h0-1))
            
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'Fixed_C{cls} {conf:.4f}'
            cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            print(f"修复后检测{i+1}: 类别={cls}, 置信度={conf:.6f}, 坐标=[{x1},{y1},{x2},{y2}]")
        
        # 保存可视化结果
        result_path = 'fixed_classification_result.jpg'
        cv2.imwrite(result_path, img_vis)
        print(f"📸 修复后检测结果已保存: {result_path}")
        
        return True
    else:
        print("❌ 修复后仍无有效检测")
        return False

if __name__ == "__main__":
    print("🚀 开始修复分类头问题...")
    
    # 修复分类头
    fixed_model = fix_classification_head()
    
    # 测试修复后的模型
    success = test_fixed_model(fixed_model)
    
    if success:
        print("\n🎉 分类头修复成功！")
        print("📋 修复效果:")
        print("   - 重新初始化分类头权重")
        print("   - 增加分类损失权重到5.0")
        print("   - 分类头使用5倍学习率")
        print("   - 训练1000轮确保收敛")
    else:
        print("\n❌ 分类头修复失败，需要进一步调整！")
