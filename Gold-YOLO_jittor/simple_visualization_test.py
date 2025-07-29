#!/usr/bin/env python3
"""
简化的可视化测试脚本
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
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

def pytorch_exact_initialization(model):
    """完全照抄PyTorch版本的初始化"""
    for name, module in model.named_modules():
        if hasattr(module, 'initialize_biases'):
            module.initialize_biases()
            break
    return model

def simple_visualization_test():
    """简化的可视化测试"""
    print(f"🔧 简化的可视化测试")
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
    
    # 准备输入
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=640, stride=32, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
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
    
    print(f"\n🚀 快速训练100轮:")
    
    # 训练循环
    for epoch in range(100):
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch+1, step_num=1)
        
        # 优化
        optimizer.step(loss)
        
        epoch_loss = float(loss.numpy())
        
        # 每25轮检测一次
        if (epoch + 1) % 25 == 0:
            print(f"\n   Epoch {epoch+1}: Loss {epoch_loss:.6f}")
            
            # 检测测试
            model.eval()
            with jt.no_grad():
                test_outputs = model(img_tensor)
                
                # 检查期望类别的分数
                coords = test_outputs[..., :4]
                objectness = test_outputs[..., 4]
                classes = test_outputs[..., 5:]
                
                expected_classes = [3, 11, 14]  # boat, dog, person
                print(f"     期望类别分数:")
                for cls_id in expected_classes:
                    cls_scores = classes[0, :, cls_id]
                    max_score = float(cls_scores.max())
                    print(f"       {VOC_CLASSES[cls_id]}(类别{cls_id}): 最大{max_score:.6f}")
                
                # NMS处理
                pred = non_max_suppression(test_outputs, conf_thres=0.01, iou_thres=0.45, max_det=100)
                
                if len(pred) > 0 and len(pred[0]) > 0:
                    detections = pred[0]
                    det_count = len(detections)
                    print(f"     检测数量: {det_count}")
                    
                    # 转换为numpy
                    if hasattr(detections, 'numpy'):
                        detections_np = detections.numpy()
                    else:
                        detections_np = detections
                    
                    # 确保检测结果是2维的
                    if detections_np.ndim == 3:
                        detections_np = detections_np.reshape(-1, detections_np.shape[-1])
                    
                    # 统计检测到的类别
                    detected_counts = {}
                    confidence_info = []
                    
                    for detection in detections_np:
                        if len(detection) >= 6:
                            conf = float(detection[4])
                            cls_id = int(detection[5])
                            cls_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f'Class{cls_id}'
                            
                            detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
                            confidence_info.append((cls_name, conf))
                    
                    print(f"     检测类别: {detected_counts}")
                    
                    # 显示置信度最高的前5个检测
                    confidence_info.sort(key=lambda x: x[1], reverse=True)
                    print(f"     置信度最高的5个检测:")
                    for i, (cls_name, conf) in enumerate(confidence_info[:5]):
                        print(f"       {i+1}. {cls_name}: {conf:.6f}")
                    
                    # 检查是否检测到期望类别
                    expected_class_names = set(target_counts.keys())
                    detected_class_names = set(detected_counts.keys())
                    correct_classes = expected_class_names.intersection(detected_class_names)
                    
                    if len(correct_classes) > 0:
                        print(f"     ✅ 检测到正确类别: {correct_classes}")
                        species_accuracy = len(correct_classes) / len(expected_class_names)
                        print(f"     种类准确率: {species_accuracy*100:.1f}%")
                        
                        if species_accuracy >= 0.8:
                            print(f"\n🎉 种类识别成功！可以开始200轮完整训练！")
                            return True
                    else:
                        expected_class_names_list = list(expected_class_names)
                        print(f"     ❌ 未检测到正确类别，期望: {expected_class_names_list}")
                else:
                    print(f"     ❌ 没有检测结果")
            
            model.train()
    
    print(f"\n⚠️ 100轮训练完成")
    return False

def main():
    print("🔥 简化的可视化测试脚本")
    print("=" * 70)
    print("功能：测试检测结果，分析置信度和数量问题")
    print("=" * 70)
    
    success = simple_visualization_test()
    
    if success:
        print(f"\n🎉🎉🎉 测试成功！🎉🎉🎉")
        print(f"✅ 检测结果正常")
        print(f"✅ 置信度分析完整")
        print(f"✅ 可以开始200轮完整训练")
    else:
        print(f"\n⚠️ 需要进一步分析置信度问题")

if __name__ == "__main__":
    main()
