#!/usr/bin/env python3
"""
可视化检测结果脚本 - 对齐PyTorch版本
支持训练过程和推理过程的可视化
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
import math
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

# 类别颜色 - 对齐PyTorch版本
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

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """绘制检测框和标签 - 完全对齐PyTorch版本"""
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 
                   cv2.FONT_HERSHEY_COMPLEX, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

def visualize_detections(image, detections, conf_thres=0.25, hide_labels=False, hide_conf=False):
    """可视化检测结果 - 对齐PyTorch版本"""
    img_vis = image.copy()
    lw = max(round(sum(img_vis.shape) / 2 * 0.003), 2)  # line width
    
    detection_count = 0
    class_counts = {}
    
    if len(detections) > 0:
        # 转换为numpy
        if hasattr(detections, 'numpy'):
            detections = detections.numpy()

        # 确保检测结果是2维的
        if detections.ndim == 3:
            detections = detections.reshape(-1, detections.shape[-1])

        # 处理检测结果
        for detection in detections:
            if len(detection) >= 6:
                xyxy = detection[:4]
                conf = detection[4]
                cls = detection[5]
            if conf >= conf_thres:
                detection_count += 1
                class_num = int(cls)
                class_name = VOC_CLASSES[class_num] if class_num < len(VOC_CLASSES) else f'Class{class_num}'
                
                # 统计类别数量
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # 生成标签
                if hide_labels:
                    label = None
                elif hide_conf:
                    label = class_name
                else:
                    label = f'{class_name} {conf:.2f}'
                
                # 获取颜色
                color = COLORS[class_num % len(COLORS)].tolist()
                
                # 绘制检测框和标签
                plot_box_and_label(img_vis, lw, xyxy, label, color=color)
            else:
                print(f"   ⚠️ 检测结果格式错误: {detection}")
    
    return img_vis, detection_count, class_counts

def analyze_detection_results(detections, conf_thres=0.25):
    """分析检测结果 - 详细统计"""
    if len(detections) == 0:
        return {
            'total_detections': 0,
            'class_counts': {},
            'confidence_stats': {},
            'confidence_distribution': []
        }
    
    # 转换为numpy
    if hasattr(detections, 'numpy'):
        detections = detections.numpy()
    
    total_detections = 0
    class_counts = {}
    confidence_stats = {}
    confidence_distribution = []
    
    for detection in detections:
        if len(detection) >= 6:
            xyxy = detection[:4]
            conf = detection[4]
            cls = detection[5]
        else:
            continue

        if conf >= conf_thres:
            total_detections += 1
            class_num = int(cls)
            class_name = VOC_CLASSES[class_num] if class_num < len(VOC_CLASSES) else f'Class{class_num}'
            
            # 统计类别数量
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # 统计置信度
            if class_name not in confidence_stats:
                confidence_stats[class_name] = []
            confidence_stats[class_name].append(float(conf))
            confidence_distribution.append((class_name, float(conf)))
    
    # 计算置信度统计
    for class_name in confidence_stats:
        confs = confidence_stats[class_name]
        confidence_stats[class_name] = {
            'count': len(confs),
            'max': max(confs),
            'min': min(confs),
            'mean': sum(confs) / len(confs),
            'std': np.std(confs)
        }
    
    return {
        'total_detections': total_detections,
        'class_counts': class_counts,
        'confidence_stats': confidence_stats,
        'confidence_distribution': sorted(confidence_distribution, key=lambda x: x[1], reverse=True)
    }

def visualize_training_progress(model, img_tensor, targets_tensor, annotations, epoch, save_dir):
    """可视化训练进度"""
    model.eval()
    
    with jt.no_grad():
        # 前向传播
        outputs = model(img_tensor)
        
        # NMS处理
        pred = non_max_suppression(outputs, conf_thres=0.01, iou_thres=0.45, max_det=100)
        
        if len(pred) > 0 and len(pred[0]) > 0:
            detections = pred[0]
        else:
            detections = jt.array([])
        
        # 准备原始图像
        img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
        original_img = cv2.imread(img_path)
        
        # 可视化检测结果
        img_vis, detection_count, class_counts = visualize_detections(
            original_img, detections, conf_thres=0.01, hide_conf=False
        )
        
        # 分析检测结果
        analysis = analyze_detection_results(detections, conf_thres=0.01)
        
        # 添加训练信息
        info_text = [
            f"Epoch: {epoch}",
            f"Detections: {detection_count}",
            f"Expected: {len(annotations)}",
            f"Classes: {list(class_counts.keys())}"
        ]
        
        # 绘制信息文本
        y_offset = 30
        for text in info_text:
            cv2.putText(img_vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 25
        
        # 保存可视化结果
        save_path = save_dir / f'epoch_{epoch:03d}_detection.jpg'
        cv2.imwrite(str(save_path), img_vis)
        
        return analysis, str(save_path)

def test_current_model_visualization():
    """测试当前模型的可视化效果"""
    print(f"🔧 测试当前模型的可视化效果")
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
    
    # 创建保存目录
    save_dir = Path("runs/visualization_test")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 快速训练并可视化:")
    
    # 训练并可视化
    for epoch in range(0, 201, 50):  # 0, 50, 100, 150, 200
        if epoch > 0:
            # 训练50轮
            model.train()  # 确保训练模式
            for _ in range(50):
                outputs = model(img_tensor)
                loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=1)
                optimizer.step(loss)
        
        # 可视化当前状态
        analysis, save_path = visualize_training_progress(
            model, img_tensor, targets_tensor, annotations, epoch, save_dir
        )
        
        print(f"\n   Epoch {epoch}:")
        print(f"     检测数量: {analysis['total_detections']}")
        print(f"     检测类别: {list(analysis['class_counts'].keys())}")
        print(f"     置信度统计:")
        
        for class_name, stats in analysis['confidence_stats'].items():
            print(f"       {class_name}: 最大{stats['max']:.3f}, 平均{stats['mean']:.3f}, 数量{stats['count']}")
        
        print(f"     可视化保存: {save_path}")
        
        # 检查是否检测到期望类别
        expected_classes = set(target_counts.keys())
        detected_classes = set(analysis['class_counts'].keys())
        correct_classes = expected_classes.intersection(detected_classes)
        
        if len(correct_classes) > 0:
            print(f"     ✅ 检测到正确类别: {correct_classes}")
            species_accuracy = len(correct_classes) / len(expected_classes)
            print(f"     种类准确率: {species_accuracy*100:.1f}%")
            
            if species_accuracy >= 0.8:
                print(f"\n🎉 种类识别成功！可以开始200轮完整训练！")
                return True
        else:
            print(f"     ❌ 未检测到正确类别，期望: {expected_classes}")
    
    return False

def main():
    print("🔥 可视化检测结果脚本")
    print("=" * 70)
    print("功能：可视化检测结果，分析置信度和数量问题")
    print("对齐：完全对齐PyTorch版本的可视化实现")
    print("=" * 70)
    
    success = test_current_model_visualization()
    
    if success:
        print(f"\n🎉🎉🎉 可视化测试成功！🎉🎉🎉")
        print(f"✅ 检测结果可视化正常")
        print(f"✅ 置信度分析完整")
        print(f"✅ 可以开始200轮完整训练")
    else:
        print(f"\n⚠️ 需要进一步分析置信度问题")

if __name__ == "__main__":
    main()
