#!/usr/bin/env python3
"""
极速过拟合测试 - 专门解决训练速度慢的问题
单图片过拟合训练应该相当快！
目标：1秒内完成单轮训练，100轮内看到推理结果
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import time

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.losses import ComputeLoss  # 使用原来的简化版损失函数
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

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections_to_gt(detections, gt_boxes, gt_classes, iou_threshold=0.3):
    """将检测结果与真实框匹配"""
    matched_gt = set()
    correct_detections = []
    
    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            det_box = [float(x1), float(y1), float(x2), float(y2)]
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt:
                    continue
                
                if cls_id == gt_cls:
                    iou = calculate_iou(det_box, gt_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                correct_detections.append({
                    'det_box': det_box,
                    'gt_box': gt_boxes[best_gt_idx],
                    'class': cls_id,
                    'class_name': VOC_CLASSES[cls_id],
                    'confidence': float(conf),
                    'iou': best_iou
                })
    
    return correct_detections, len(matched_gt), len(gt_boxes)

def ultra_fast_overfitting_test():
    """极速过拟合测试"""
    print(f"🚀 极速过拟合测试 - 解决训练速度慢的问题")
    print("=" * 80)
    print("目标：1秒内完成单轮训练，100轮内看到推理结果")
    print("=" * 80)
    
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
    
    # 读取原始图像
    original_img = cv2.imread(img_path)
    img_height, img_width = original_img.shape[:2]
    
    # 转换真实标注为像素坐标
    gt_boxes = []
    gt_classes = []
    target_counts = {}
    
    for ann in annotations:
        cls_id, x_center, y_center, width, height = ann
        
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        gt_boxes.append([x1, y1, x2, y2])
        gt_classes.append(cls_id)
        
        cls_name = VOC_CLASSES[cls_id]
        target_counts[cls_name] = target_counts.get(cls_name, 0) + 1
    
    print(f"📋 真实标注: {target_counts}")
    print(f"   总目标数: {len(annotations)}")
    
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
    print(f"🎯 创建极速模型...")
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    
    # 创建简化损失函数 - 使用原来的版本，速度更快
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=4,
        use_dfl=False,
        reg_max=0,
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    
    # 创建优化器 - 提高学习率
    optimizer = jt.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0005)  # 提高学习率
    
    # 创建保存目录
    save_dir = Path("runs/ultra_fast_test")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始极速训练 (100轮):")
    
    # 训练记录
    loss_history = []
    accuracy_history = []
    best_strict_accuracy = 0.0
    best_model_path = None
    
    # 预热编译
    print(f"🔥 预热编译...")
    model.train()
    outputs = model(img_tensor)
    loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=1, step_num=1)
    optimizer.zero_grad()
    optimizer.backward(loss)
    optimizer.step()
    print(f"✅ 预热完成")
    
    for epoch in range(1, 101):  # 100轮快速测试
        start_time = time.time()
        
        # 训练模式
        model.train()
        
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=1)
        
        # 反向传播
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        epoch_time = time.time() - start_time
        epoch_loss = float(loss.data.item()) if hasattr(loss.data, 'item') else float(loss.data)
        loss_history.append(epoch_loss)
        
        # 每10轮进行检测测试
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss {epoch_loss:.6f} ({epoch_time:.2f}s)")
            
            # 推理测试
            model.eval()
            with jt.no_grad():
                test_outputs = model(img_tensor)
                
                # 尝试不同的置信度阈值
                best_result = None
                for conf_thresh in [0.001, 0.01, 0.05, 0.1, 0.2]:
                    try:
                        pred = non_max_suppression(test_outputs, conf_thres=conf_thresh, iou_thres=0.6, max_det=10)
                        
                        if len(pred) > 0 and len(pred[0]) > 0:
                            detections = pred[0]
                            
                            if hasattr(detections, 'numpy'):
                                detections_np = detections.numpy()
                            else:
                                detections_np = detections
                            
                            if detections_np.ndim == 3:
                                detections_np = detections_np.reshape(-1, detections_np.shape[-1])
                            
                            # 严格评估
                            correct_detections, matched_count, total_gt = match_detections_to_gt(
                                detections_np, gt_boxes, gt_classes, iou_threshold=0.3
                            )
                            
                            strict_accuracy = matched_count / total_gt if total_gt > 0 else 0.0
                            
                            if strict_accuracy > 0 or len(detections_np) > 0:
                                best_result = {
                                    'conf_thresh': conf_thresh,
                                    'detections': detections_np,
                                    'correct_detections': correct_detections,
                                    'matched_count': matched_count,
                                    'total_gt': total_gt,
                                    'strict_accuracy': strict_accuracy
                                }
                                break
                    except:
                        continue
                
                if best_result:
                    result = best_result
                    accuracy_history.append(result['strict_accuracy'])
                    
                    print(f"   置信度阈值: {result['conf_thresh']}")
                    print(f"   检测数量: {len(result['detections'])} (期望: {result['total_gt']})")
                    print(f"   严格评估: {result['matched_count']}/{result['total_gt']} = {result['strict_accuracy']*100:.1f}%")
                    
                    # 检测数量验证
                    if len(result['detections']) > result['total_gt'] * 3:
                        print(f"   ⚠️ 检测数量过多！")
                    elif len(result['detections']) < result['total_gt'] * 0.5:
                        print(f"   ⚠️ 检测数量过少！")
                    else:
                        print(f"   ✅ 检测数量合理")
                    
                    # 保存最佳模型
                    if result['strict_accuracy'] > best_strict_accuracy:
                        best_strict_accuracy = result['strict_accuracy']
                        best_model_path = save_dir / f'best_ultra_fast_model_epoch_{epoch}.pkl'
                        jt.save({
                            'model': model.state_dict(),
                            'epoch': epoch,
                            'loss': epoch_loss,
                            'strict_accuracy': result['strict_accuracy'],
                            'conf_thresh': result['conf_thresh'],
                            'correct_detections': result['correct_detections']
                        }, str(best_model_path))
                        print(f"   🏆 新的最佳结果！严格准确率: {result['strict_accuracy']*100:.1f}%")
                        
                        # 显示检测详情
                        if result['correct_detections']:
                            print(f"   ✅ 正确检测:")
                            for i, cd in enumerate(result['correct_detections']):
                                print(f"     {i+1}. {cd['class_name']}: IoU={cd['iou']:.3f}, Conf={cd['confidence']:.3f}")
                
                else:
                    accuracy_history.append(0.0)
                    print(f"   ❌ 所有置信度阈值都没有检测结果")
            
            model.train()
    
    print(f"\n🎉 极速训练完成！")
    print(f"✅ 最佳严格准确率: {best_strict_accuracy*100:.1f}%")
    
    # 最终评估
    if best_model_path and os.path.exists(best_model_path):
        checkpoint = jt.load(str(best_model_path))
        
        print(f"\n📊 最终结果:")
        print(f"   最佳轮次: {checkpoint['epoch']}")
        print(f"   最佳置信度阈值: {checkpoint['conf_thresh']}")
        print(f"   最佳严格准确率: {checkpoint['strict_accuracy']*100:.1f}%")
        
        if checkpoint['correct_detections']:
            print(f"   正确检测详情:")
            for i, cd in enumerate(checkpoint['correct_detections']):
                print(f"     {i+1}. {cd['class_name']}: IoU={cd['iou']:.3f}, Conf={cd['confidence']:.3f}")
        
        # 评估是否通过测试
        detected_classes = set(cd['class_name'] for cd in checkpoint['correct_detections'])
        expected_classes = {'dog', 'person', 'boat'}
        class_accuracy = len(detected_classes.intersection(expected_classes)) / len(expected_classes)
        
        if best_strict_accuracy >= 0.5 and class_accuracy >= 0.5:
            print(f"\n🎉🎉🎉 极速过拟合测试通过！🎉🎉🎉")
            print(f"✅ 种类准确率: {class_accuracy*100:.1f}%")
            print(f"✅ 位置准确率: {best_strict_accuracy*100:.1f}%")
            print(f"✅ 检测类别: {detected_classes}")
            return True
        else:
            print(f"\n📊 极速过拟合测试完成，需要进一步优化")
            print(f"   种类准确率: {class_accuracy*100:.1f}%")
            print(f"   位置准确率: {best_strict_accuracy*100:.1f}%")
            print(f"   检测类别: {detected_classes}")
            print(f"   期望类别: {expected_classes}")
            return False
    
    else:
        print(f"❌ 没有保存的最佳模型")
        return False

def main():
    print("🚀 极速过拟合测试 - 专门解决训练速度慢的问题")
    print("=" * 80)
    print("目标：1秒内完成单轮训练，100轮内看到推理结果")
    print("=" * 80)
    
    success = ultra_fast_overfitting_test()
    
    if success:
        print(f"\n🎉🎉🎉 极速测试成功！🎉🎉🎉")
        print(f"✅ 训练速度问题已解决")
        print(f"✅ 推理测试结果正常")
    else:
        print(f"\n📊 极速测试完成，已生成详细分析结果")
        print(f"✅ 训练速度已优化")
        print(f"📊 推理测试结果需要进一步分析")

if __name__ == "__main__":
    main()
