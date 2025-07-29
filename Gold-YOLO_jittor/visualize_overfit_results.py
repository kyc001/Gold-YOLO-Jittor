#!/usr/bin/env python3
"""
可视化单张图片过拟合结果
绘制检测框，显示置信度和类别
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
import time
from pathlib import Path

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.losses import ComputeLoss
from yolov6.utils.nms import non_max_suppression

# VOC数据集类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 类别颜色
COLORS = np.array([
    [255, 178, 50], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
    [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 192, 192], [128, 128, 128],
    [255, 165, 0], [255, 20, 147], [0, 191, 255], [255, 105, 180], [34, 139, 34]
], dtype=np.uint8)

def pytorch_exact_initialization(model):
    """完全照抄PyTorch版本的初始化"""
    for name, module in model.named_modules():
        if hasattr(module, 'initialize_biases'):
            module.initialize_biases()
            break
    return model

def draw_detection_box(img, box, label, confidence, color):
    """绘制检测框"""
    x1, y1, x2, y2 = map(int, box)
    
    # 绘制检测框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 准备标签文本
    label_text = f'{label}: {confidence:.3f}'
    
    # 计算文本大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    # 绘制标签背景
    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    
    # 绘制标签文本
    cv2.putText(img, label_text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

def draw_ground_truth_box(img, box, label, color=(0, 255, 0)):
    """绘制真实标注框"""
    x1, y1, x2, y2 = map(int, box)
    
    # 绘制真实框（虚线效果）
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 绘制标签
    label_text = f'GT: {label}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    cv2.rectangle(img, (x1, y2), (x1 + text_width, y2 + text_height + 5), color, -1)
    cv2.putText(img, label_text, (x1, y2 + text_height), font, font_scale, (255, 255, 255), thickness)

def visualize_overfit_results():
    """可视化过拟合结果"""
    print(f"🎨 可视化单张图片过拟合结果")
    print("=" * 50)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    # 读取真实标注
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
    
    target_counts = {}
    for ann in annotations:
        cls_name = VOC_CLASSES[ann[0]]
        target_counts[cls_name] = target_counts.get(cls_name, 0) + 1
    
    print(f"📋 期望检测结果: {target_counts}")
    print(f"   总目标数: {len(annotations)}")
    
    # 读取原始图像
    original_img = cv2.imread(img_path)
    img_height, img_width = original_img.shape[:2]
    
    # 准备输入
    img = letterbox(original_img, new_shape=640, stride=32, auto=False)[0]
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
    
    # 创建模型
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    model.train()
    
    # 创建损失函数和优化器
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=4,
        use_dfl=False,
        reg_max=0,
        iou_type='giou',
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    optimizer = jt.optim.AdamW(model.parameters(), lr=0.05)
    
    # 创建保存目录
    save_dir = Path("runs/visualization_overfit")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 快速训练100轮并可视化:")
    
    # 训练循环
    for epoch in range(100):
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch+1, step_num=1)
        
        # 优化
        optimizer.step(loss)
        
        epoch_loss = float(loss.numpy())
        
        # 每25轮可视化一次
        if (epoch + 1) % 25 == 0:
            print(f"\n   Epoch {epoch+1}: Loss {epoch_loss:.6f}")
            
            # 推理模式
            model.eval()
            with jt.no_grad():
                test_outputs = model(img_tensor)
                
                # 分析模型输出
                coords = test_outputs[..., :4]
                objectness = test_outputs[..., 4]
                classes = test_outputs[..., 5:]
                
                print(f"     模型输出分析:")
                print(f"       坐标范围: [{coords.min():.3f}, {coords.max():.3f}]")
                print(f"       objectness范围: [{objectness.min():.3f}, {objectness.max():.3f}]")
                print(f"       类别分数范围: [{classes.min():.6f}, {classes.max():.6f}]")
                
                # 检查期望类别的分数
                expected_classes = [3, 11, 14]  # boat, dog, person
                print(f"     期望类别分数:")
                for cls_id in expected_classes:
                    cls_scores = classes[0, :, cls_id]
                    max_score = float(cls_scores.max())
                    argmax_result = cls_scores.argmax(dim=0)
                    if isinstance(argmax_result, tuple):
                        max_idx = int(argmax_result[0])
                    else:
                        max_idx = int(argmax_result)
                    print(f"       {VOC_CLASSES[cls_id]}(类别{cls_id}): 最大{max_score:.6f} (位置{max_idx})")

                # 检查aeroplane的分数
                aero_scores = classes[0, :, 0]
                aero_max_score = float(aero_scores.max())
                aero_argmax_result = aero_scores.argmax(dim=0)
                if isinstance(aero_argmax_result, tuple):
                    aero_max_idx = int(aero_argmax_result[0])
                else:
                    aero_max_idx = int(aero_argmax_result)
                print(f"       aeroplane(类别0): 最大{aero_max_score:.6f} (位置{aero_max_idx})")
                
                # **关键调试：手动检查最高分数的类别**
                print(f"\n     🔍 手动检查最高分数的类别:")
                all_max_scores = classes[0].max(dim=1)  # 每个anchor的最大分数
                if isinstance(all_max_scores, tuple):
                    max_scores, max_indices = all_max_scores
                else:
                    max_scores = all_max_scores
                    max_indices = classes[0].argmax(dim=1)
                
                # 找到全局最高分数
                global_max_score = float(max_scores.max())
                global_max_anchor_result = max_scores.argmax(dim=0)
                if isinstance(global_max_anchor_result, tuple):
                    global_max_anchor = int(global_max_anchor_result[0])
                else:
                    global_max_anchor = int(global_max_anchor_result)

                if isinstance(max_indices, tuple):
                    max_indices_tensor = max_indices[0] if len(max_indices) > 0 else max_indices
                else:
                    max_indices_tensor = max_indices

                global_max_class_result = max_indices_tensor[global_max_anchor]
                if isinstance(global_max_class_result, tuple):
                    global_max_class = int(global_max_class_result[0])
                else:
                    global_max_class = int(global_max_class_result)
                global_max_class_name = VOC_CLASSES[global_max_class] if global_max_class < len(VOC_CLASSES) else f'Class{global_max_class}'
                
                print(f"       全局最高分数: {global_max_score:.6f}")
                print(f"       对应anchor: {global_max_anchor}")
                print(f"       对应类别: {global_max_class_name}(类别{global_max_class})")
                
                # NMS处理
                pred = non_max_suppression(test_outputs, conf_thres=0.01, iou_thres=0.45, max_det=100)
                
                if len(pred) > 0 and len(pred[0]) > 0:
                    detections = pred[0]
                    det_count = len(detections)
                    print(f"     NMS后检测数量: {det_count}")
                    
                    # 转换为numpy
                    if hasattr(detections, 'numpy'):
                        detections_np = detections.numpy()
                    else:
                        detections_np = detections
                    
                    # 确保检测结果是2维的
                    if detections_np.ndim == 3:
                        detections_np = detections_np.reshape(-1, detections_np.shape[-1])
                    
                    # 创建可视化图像
                    vis_img = original_img.copy()
                    
                    # 绘制真实标注框
                    for ann in annotations:
                        cls_id, x_center, y_center, width, height = ann
                        
                        # 转换为像素坐标
                        x1 = int((x_center - width/2) * img_width)
                        y1 = int((y_center - height/2) * img_height)
                        x2 = int((x_center + width/2) * img_width)
                        y2 = int((y_center + height/2) * img_height)
                        
                        cls_name = VOC_CLASSES[cls_id]
                        draw_ground_truth_box(vis_img, [x1, y1, x2, y2], cls_name)
                    
                    # 统计检测结果
                    detected_counts = {}
                    confidence_info = []
                    
                    # 绘制检测框
                    for i, detection in enumerate(detections_np):
                        if len(detection) >= 6:
                            x1, y1, x2, y2, conf, cls_id = detection[:6]
                            cls_id = int(cls_id)
                            cls_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f'Class{cls_id}'
                            
                            detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
                            confidence_info.append((cls_name, float(conf)))
                            
                            # 只绘制前10个检测框
                            if i < 10:
                                color = COLORS[cls_id % len(COLORS)].tolist()
                                
                                # 缩放坐标到原图尺寸
                                scale_x = img_width / 640
                                scale_y = img_height / 640
                                x1_scaled = int(x1 * scale_x)
                                y1_scaled = int(y1 * scale_y)
                                x2_scaled = int(x2 * scale_x)
                                y2_scaled = int(y2 * scale_y)
                                
                                draw_detection_box(vis_img, [x1_scaled, y1_scaled, x2_scaled, y2_scaled], 
                                                 cls_name, float(conf), color)
                    
                    print(f"     检测类别统计: {detected_counts}")
                    
                    # 显示置信度最高的前5个检测
                    confidence_info.sort(key=lambda x: x[1], reverse=True)
                    print(f"     置信度最高的5个检测:")
                    for i, (cls_name, conf) in enumerate(confidence_info[:5]):
                        print(f"       {i+1}. {cls_name}: {conf:.6f}")
                    
                    # 保存可视化结果
                    save_path = save_dir / f'epoch_{epoch+1:03d}_visualization.jpg'
                    cv2.imwrite(str(save_path), vis_img)
                    print(f"     💾 可视化结果已保存: {save_path}")
                    
                    # 检查是否检测到期望类别
                    expected_class_names = set(target_counts.keys())
                    detected_class_names = set(detected_counts.keys())
                    correct_classes = expected_class_names.intersection(detected_class_names)
                    
                    if len(correct_classes) > 0:
                        print(f"     ✅ 检测到正确类别: {correct_classes}")
                        species_accuracy = len(correct_classes) / len(expected_class_names)
                        print(f"     种类准确率: {species_accuracy*100:.1f}%")
                        
                        # **关键检查：dog是否是置信度最高的**
                        if confidence_info and confidence_info[0][0] == 'dog':
                            print(f"     🎉 dog是置信度最高的类别！")
                            if species_accuracy >= 0.8:
                                print(f"\n🎉 完美过拟合成功！")
                                return True
                        else:
                            print(f"     ❌ dog不是置信度最高的类别，需要继续训练")
                    else:
                        expected_class_names_list = list(expected_class_names)
                        print(f"     ❌ 未检测到正确类别，期望: {expected_class_names_list}")
                else:
                    print(f"     ❌ 没有检测结果")
            
            model.train()
    
    print(f"\n⚠️ 100轮训练完成，需要进一步分析")
    return False

def main():
    print("🔥 可视化单张图片过拟合结果")
    print("=" * 70)
    print("目标：可视化检测结果，确保dog是置信度最高的类别")
    print("=" * 70)
    
    success = visualize_overfit_results()
    
    if success:
        print(f"\n🎉🎉🎉 可视化成功！🎉🎉🎉")
        print(f"✅ dog是置信度最高的类别")
        print(f"✅ 单张图片过拟合成功")
    else:
        print(f"\n⚠️ 需要深入修复置信度问题")

if __name__ == "__main__":
    main()
