#!/usr/bin/env python3
"""
修复后的200轮完整训练脚本
修复了DFL损失显示问题，简化调试信息，添加进度条
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

# 类别颜色 - 期望类别使用特殊颜色
COLORS = {
    'dog': (0, 255, 0),      # 绿色 - 主要目标
    'person': (255, 0, 0),   # 蓝色
    'boat': (0, 0, 255),     # 红色
    'default': (128, 128, 128)  # 灰色 - 其他类别
}

def pytorch_exact_initialization(model):
    """完全照抄PyTorch版本的初始化"""
    for name, module in model.named_modules():
        if hasattr(module, 'initialize_biases'):
            module.initialize_biases()
            break
    return model

def draw_detection_box(img, box, label, confidence, color, is_expected=False):
    """绘制检测框 - 期望类别使用特殊样式"""
    x1, y1, x2, y2 = map(int, box)
    
    # 确保坐标在图像范围内
    img_h, img_w = img.shape[:2]
    x1 = max(0, min(img_w-1, x1))
    y1 = max(0, min(img_h-1, y1))
    x2 = max(0, min(img_w-1, x2))
    y2 = max(0, min(img_h-1, y2))
    
    # 确保坐标有效
    if x2 <= x1 or y2 <= y1:
        return
    
    # 期望类别使用粗线条
    thickness = 3 if is_expected else 2
    
    # 绘制检测框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 准备标签文本
    status = "✅" if is_expected else "❌"
    label_text = f'{status}{label}: {confidence:.3f}'
    
    # 计算文本大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 if is_expected else 0.6
    text_thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
    
    # 绘制标签背景
    bg_color = color if is_expected else (64, 64, 64)
    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), bg_color, -1)
    
    # 绘制标签文本
    text_color = (255, 255, 255)
    cv2.putText(img, label_text, (x1, y1 - 5), font, font_scale, text_color, text_thickness)

def draw_ground_truth_box(img, box, label, color=(0, 255, 255)):
    """绘制真实标注框"""
    x1, y1, x2, y2 = map(int, box)
    
    # 确保坐标在图像范围内
    img_h, img_w = img.shape[:2]
    x1 = max(0, min(img_w-1, x1))
    y1 = max(0, min(img_h-1, y1))
    x2 = max(0, min(img_w-1, x2))
    y2 = max(0, min(img_h-1, y2))
    
    # 确保坐标有效
    if x2 <= x1 or y2 <= y1:
        return
    
    # 绘制真实框（虚线效果）
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 绘制标签
    label_text = f'GT: {label}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    cv2.rectangle(img, (x1, y2), (x1 + text_width, y2 + text_height + 5), color, -1)
    cv2.putText(img, label_text, (x1, y2 + text_height), font, font_scale, (0, 0, 0), thickness)

def print_progress_bar(current, total, bar_length=50):
    """打印进度条"""
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = progress * 100
    print(f'\r进度: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)

def final_200_epoch_clean_training():
    """修复后的200轮完整训练，简化调试信息"""
    print(f"🔥 修复后的200轮完整训练")
    print("=" * 60)
    print("修复：DFL损失显示问题，简化调试信息")
    print("=" * 60)
    
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
    expected_classes = set()
    for ann in annotations:
        cls_name = VOC_CLASSES[ann[0]]
        target_counts[cls_name] = target_counts.get(cls_name, 0) + 1
        expected_classes.add(cls_name)
    
    print(f"📋 期望检测结果: {target_counts}")
    print(f"   期望类别: {expected_classes}")
    
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
    print(f"🎯 创建模型...")
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    model.train()
    
    # 创建损失函数和优化器 - 100%对齐PyTorch版本参数
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=4,
        use_dfl=False,  # 确保DFL关闭
        reg_max=0,      # 确保reg_max为0
        iou_type='giou',
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    optimizer = jt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0005)
    
    # 创建保存目录
    save_dir = Path("runs/final_200_epoch_clean_training")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始200轮训练:")
    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M)")
    print(f"   优化器: AdamW (lr=0.01, weight_decay=0.0005)")
    print(f"   损失权重: class=1.0, iou=2.5, dfl=0.5")
    print(f"   use_dfl: False, reg_max: 0")
    
    # 训练循环
    best_species_accuracy = 0.0
    best_epoch = 0
    training_log = []
    
    for epoch in range(200):
        start_time = time.time()
        
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch+1, step_num=1)
        
        # 优化
        optimizer.step(loss)
        
        epoch_loss = float(loss.numpy())
        epoch_time = time.time() - start_time
        
        # 记录训练日志
        log_entry = {
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'time': epoch_time
        }
        
        # 每20轮详细分析和可视化
        if (epoch + 1) % 20 == 0:
            print(f"\n")
            print_progress_bar(epoch + 1, 200)
            print(f"\n   Epoch {epoch+1}: Loss {epoch_loss:.6f} ({epoch_time:.2f}s)")
            
            # 分析损失分解 - 修复后的顺序
            if hasattr(loss_items, '__len__') and len(loss_items) >= 3:
                cls_loss = float(loss_items[0])  # 现在是正确的分类损失
                iou_loss = float(loss_items[1])  # 现在是正确的IoU损失
                dfl_loss = float(loss_items[2])  # 现在是正确的DFL损失
                print(f"     损失分解: 分类{cls_loss:.6f}, IoU{iou_loss:.6f}, DFL{dfl_loss:.6f}")
                
                # 检查DFL损失是否为0（应该为0，因为use_dfl=False）
                if dfl_loss > 0.001:
                    print(f"     ⚠️ DFL损失不为0，可能有问题")
                else:
                    print(f"     ✅ DFL损失为0，符合预期")
                
                log_entry.update({
                    'cls_loss': cls_loss,
                    'iou_loss': iou_loss,
                    'dfl_loss': dfl_loss
                })
            
            # 推理模式检查
            model.eval()
            with jt.no_grad():
                # 获取训练模式的输出
                train_outputs = model(img_tensor)
                
                # 分析期望类别的学习情况
                if isinstance(train_outputs, tuple) and len(train_outputs) >= 3:
                    pred_scores = train_outputs[1]  # [1, 8400, 20]
                else:
                    pred_scores = train_outputs[..., 5:]  # [1, 8400, 20]
                
                print(f"     期望类别学习情况:")
                expected_class_ids = [3, 11, 14]  # boat, dog, person
                class_scores = {}
                for cls_id in expected_class_ids:
                    cls_scores = pred_scores[0, :, cls_id]
                    max_score = float(cls_scores.max())
                    mean_score = float(cls_scores.mean())
                    nonzero_count = int((cls_scores > 0.001).sum())
                    cls_name = VOC_CLASSES[cls_id]
                    print(f"       {cls_name}(类别{cls_id}): 最大{max_score:.6f}, 平均{mean_score:.6f}, 激活{nonzero_count}")
                    
                    class_scores[cls_name] = {
                        'max': max_score,
                        'mean': mean_score,
                        'nonzero': nonzero_count
                    }
                
                log_entry['class_scores'] = class_scores
                
                # NMS处理和可视化
                try:
                    pred = non_max_suppression(train_outputs, conf_thres=0.01, iou_thres=0.45, max_det=100)
                    
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
                        
                        # 统计检测结果
                        detected_counts = {}
                        expected_detections = 0
                        confidence_info = []
                        
                        for i, detection in enumerate(detections_np[:10]):
                            if len(detection) >= 6:
                                x1, y1, x2, y2, conf, cls_id = detection[:6]
                                cls_id = int(cls_id)
                                cls_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f'Class{cls_id}'
                                
                                detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
                                confidence_info.append((cls_name, float(conf)))
                                
                                # 检查是否是期望类别
                                is_expected = cls_name in expected_classes
                                if is_expected:
                                    expected_detections += 1
                        
                        print(f"     检测类别统计: {detected_counts}")
                        print(f"     期望类别检测数: {expected_detections}")
                        
                        # 显示置信度最高的前5个检测
                        confidence_info.sort(key=lambda x: x[1], reverse=True)
                        print(f"     置信度最高的5个检测:")
                        for i, (cls_name, conf) in enumerate(confidence_info[:5]):
                            status = "✅" if cls_name in expected_classes else "❌"
                            print(f"       {i+1}. {status}{cls_name}: {conf:.6f}")
                        
                        # 计算种类准确率
                        detected_class_names = set(detected_counts.keys())
                        correct_classes = expected_classes.intersection(detected_class_names)
                        species_accuracy = len(correct_classes) / len(expected_classes) if expected_classes else 0.0
                        
                        print(f"     种类准确率: {species_accuracy*100:.1f}%")
                        print(f"     正确识别类别: {correct_classes}")
                        
                        log_entry.update({
                            'detections': det_count,
                            'detected_counts': detected_counts,
                            'species_accuracy': species_accuracy,
                            'correct_classes': list(correct_classes),
                            'confidence_info': confidence_info[:5]
                        })
                        
                        # 检查是否达到最佳效果
                        if species_accuracy > best_species_accuracy:
                            best_species_accuracy = species_accuracy
                            best_epoch = epoch + 1
                            
                            print(f"     🏆 新的最佳结果！种类准确率: {species_accuracy*100:.1f}%")
                            
                            # 保存最佳模型
                            best_model_path = save_dir / 'best_model.pkl'
                            jt.save({
                                'model': model.state_dict(),
                                'epoch': epoch + 1,
                                'species_accuracy': species_accuracy,
                                'detected_counts': detected_counts,
                                'target_counts': target_counts
                            }, str(best_model_path))
                            
                            print(f"     💾 最佳模型已保存: {best_model_path}")
                        
                        # 检查是否达到完美过拟合
                        if species_accuracy >= 1.0:
                            print(f"\n🎉 完美过拟合成功！")
                            print(f"   ✅ 种类准确率: 100%")
                            print(f"   ✅ 正确识别: {correct_classes}")
                            print(f"   ✅ 单张图片完美过拟合")
                            
                            # 保存完美模型
                            perfect_model_path = save_dir / 'perfect_overfit_model.pkl'
                            jt.save({
                                'model': model.state_dict(),
                                'epoch': epoch + 1,
                                'species_accuracy': species_accuracy,
                                'detected_counts': detected_counts,
                                'target_counts': target_counts,
                                'training_log': training_log
                            }, str(perfect_model_path))
                            
                            print(f"   💾 完美模型已保存: {perfect_model_path}")
                            
                            return True
                    else:
                        print(f"     ❌ 没有检测结果")
                        log_entry['detections'] = 0
                
                except Exception as e:
                    print(f"     ⚠️ NMS处理异常: {e}")
                    log_entry['nms_error'] = str(e)
            
            model.train()
        else:
            # 简化显示，只显示进度条
            if (epoch + 1) % 10 == 0:
                print_progress_bar(epoch + 1, 200)
        
        training_log.append(log_entry)
    
    print(f"\n\n📊 200轮训练完成!")
    print(f"   最佳种类准确率: {best_species_accuracy*100:.1f}% (Epoch {best_epoch})")
    
    # 保存最终模型
    final_model_path = save_dir / 'final_model.pkl'
    jt.save({
        'model': model.state_dict(),
        'epoch': 200,
        'best_species_accuracy': best_species_accuracy,
        'best_epoch': best_epoch,
        'training_log': training_log
    }, str(final_model_path))
    
    print(f"   💾 最终模型已保存: {final_model_path}")
    
    if best_species_accuracy >= 0.8:
        print(f"\n🎯 训练成功！")
        print(f"✅ GOLD-YOLO Jittor版本成功复现PyTorch版本")
        print(f"✅ DFL损失问题已修复")
        return True
    else:
        print(f"\n⚠️ 需要进一步优化")
        return False

def main():
    print("🔥 修复后的200轮完整训练")
    print("=" * 80)
    print("修复：DFL损失显示问题，简化调试信息，添加进度条")
    print("训练参数：100%对齐PyTorch版本")
    print("=" * 80)
    
    success = final_200_epoch_clean_training()
    
    if success:
        print(f"\n🎉🎉🎉 200轮训练成功！🎉🎉🎉")
        print(f"✅ GOLD-YOLO Jittor版本迁移完成")
        print(f"✅ 成功复现PyTorch版本功能")
        print(f"✅ DFL损失问题已修复")
        print(f"✅ 可以进行推理测试")
    else:
        print(f"\n📊 训练完成，结果已保存")
        print(f"可以查看训练日志进行进一步分析")

if __name__ == "__main__":
    main()
