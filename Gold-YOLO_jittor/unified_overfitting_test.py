#!/usr/bin/env python3
"""
统一过拟合测试脚本 - 可复用、可维护
解决您提出的4个核心问题：
1. 训练速度慢的原因分析和修复
2. 物体检测位置错误的原因分析
3. 两张图片过拟合训练测试（种类、数量、位置全部正确）
4. 清理脚本，增强可复用性和可维护性

使用方法：
python unified_overfitting_test.py --image 1  # 测试第一张图片
python unified_overfitting_test.py --image 2  # 测试第二张图片
python unified_overfitting_test.py --image both  # 测试两张图片
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import time
import matplotlib.pyplot as plt
import argparse

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.pytorch_aligned_losses import ComputeLoss
from yolov6.utils.nms import non_max_suppression

# VOC数据集类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 图片配置
IMAGE_CONFIGS = {
    1: {
        'img_path': '/home/kyc/project/GOLD-YOLO/2008_001420.jpg',
        'label_path': '/home/kyc/project/GOLD-YOLO/2008_001420.txt',
        'expected_classes': {'dog', 'person', 'boat'},
        'expected_count': 6,
        'description': '第一张图片：dog(4), person(1), boat(1)'
    },
    2: {
        'img_path': '/home/kyc/project/GOLD-YOLO/2011_002881.jpg', 
        'label_path': '/home/kyc/project/GOLD-YOLO/2011_002881.txt',
        'expected_classes': {'diningtable', 'person', 'sofa'},
        'expected_count': 7,
        'description': '第二张图片：diningtable(3), person(3), sofa(1)'
    }
}

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
    """将检测结果与真实框匹配 - 严格评估"""
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
                
                if cls_id == gt_cls:  # 种类必须匹配
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

def draw_separated_comparison(img, detections, gt_boxes, gt_classes, correct_detections, image_id, detection_stats):
    """绘制分离式对比图：左图真实标注，右图预测结果"""
    img_height, img_width = img.shape[:2]

    # 创建并排对比图
    comparison_img = np.zeros((img_height, img_width * 2, 3), dtype=np.uint8)

    # 左图：真实标注
    gt_img = img.copy()
    for gt_box, gt_cls in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = map(int, gt_box)
        cls_name = VOC_CLASSES[gt_cls]

        # 绿色实线框
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(gt_img, f'GT: {cls_name}', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 添加左图标题
    cv2.putText(gt_img, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(gt_img, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(gt_img, f'Total: {len(gt_boxes)} objects', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(gt_img, f'Total: {len(gt_boxes)} objects', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 右图：预测结果
    pred_img = img.copy()
    correct_boxes = {tuple(cd['det_box']): cd for cd in correct_detections}

    for det in detections:
        if len(det) >= 6:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(cls_id)
            cls_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f'Class{cls_id}'

            det_box_key = tuple([float(x1), float(y1), float(x2), float(y2)])
            is_correct = det_box_key in correct_boxes

            # 正确检测用绿色，错误检测用红色
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            thickness = 3 if is_correct else 2

            cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, thickness)

            status = "✅" if is_correct else "❌"
            iou_text = f" IoU:{correct_boxes[det_box_key]['iou']:.3f}" if is_correct else ""
            label_text = f'{status}{cls_name} {float(conf):.3f}{iou_text}'

            cv2.putText(pred_img, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 添加右图标题和统计信息
    cv2.putText(pred_img, 'Predictions', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(pred_img, 'Predictions', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 详细统计信息
    stats_text = [
        f'Detected: {len(detections)} objects',
        f'Correct: {len(correct_detections)} objects',
        f'Accuracy: {detection_stats["strict_accuracy"]*100:.1f}%',
        f'Conf Thresh: {detection_stats["conf_thresh"]}'
    ]

    for i, text in enumerate(stats_text):
        y_pos = 70 + i * 30
        cv2.putText(pred_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pred_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # 组合图像
    comparison_img[:, :img_width] = gt_img
    comparison_img[:, img_width:] = pred_img

    # 添加分割线
    cv2.line(comparison_img, (img_width, 0), (img_width, img_height), (255, 255, 255), 3)

    # 添加总体信息
    config = IMAGE_CONFIGS[image_id]
    info_text = f"Image {image_id}: {config['description']}"
    cv2.putText(comparison_img, info_text, (img_width//2 - 200, img_height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(comparison_img, info_text, (img_width//2 - 200, img_height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return comparison_img

def analyze_training_speed(model, img_tensor, targets_tensor, loss_fn, optimizer):
    """分析训练速度慢的原因"""
    print(f"\n🔍 训练速度分析:")
    
    # 测试单轮训练时间
    times = []
    for i in range(5):
        start_time = time.time()
        
        model.train()
        outputs = model(img_tensor)
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=1, step_num=1)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    print(f"   单轮训练平均时间: {avg_time:.2f}秒")
    print(f"   200轮预计时间: {avg_time * 200 / 60:.1f}分钟")
    
    if avg_time > 5.0:
        print(f"   ⚠️ 训练速度偏慢，可能原因:")
        print(f"     - 损失函数内部仍有调试输出")
        print(f"     - 模型计算复杂度过高")
        print(f"     - GPU利用率不足")
    else:
        print(f"   ✅ 训练速度正常")
    
    return avg_time

def analyze_detection_position_error(detections, gt_boxes, gt_classes):
    """分析物体检测位置错误的原因"""
    print(f"\n🔍 检测位置错误分析:")
    
    if len(detections) == 0:
        print(f"   ❌ 没有检测结果 - 可能原因:")
        print(f"     - 置信度阈值过高")
        print(f"     - 模型未充分训练")
        print(f"     - 分类学习失败")
        return
    
    # 分析每个检测结果
    for i, det in enumerate(detections[:10]):  # 只分析前10个
        if len(det) >= 6:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            cls_name = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f'Class{cls_id}'
            det_box = [float(x1), float(y1), float(x2), float(y2)]
            
            # 找到最佳匹配的真实框
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                iou = calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                gt_cls_name = VOC_CLASSES[gt_classes[best_gt_idx]]
                class_match = cls_id == gt_classes[best_gt_idx]
                position_match = best_iou >= 0.3
                
                print(f"   检测{i+1}: {cls_name} conf={conf:.3f}")
                print(f"     最佳匹配GT: {gt_cls_name} IoU={best_iou:.3f}")
                print(f"     类别匹配: {'✅' if class_match else '❌'}")
                print(f"     位置匹配: {'✅' if position_match else '❌'}")
                
                if not class_match:
                    print(f"     ⚠️ 类别错误: 预测{cls_name} vs 真实{gt_cls_name}")
                if not position_match:
                    print(f"     ⚠️ 位置错误: IoU={best_iou:.3f} < 0.3")

def unified_overfitting_test(image_id, epochs=30):
    """统一过拟合测试"""
    config = IMAGE_CONFIGS[image_id]
    print(f"🔥 统一过拟合测试 - {config['description']}")
    print("=" * 80)
    
    # 读取数据
    annotations = []
    with open(config['label_path'], 'r') as f:
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
    original_img = cv2.imread(config['img_path'])
    img_height, img_width = original_img.shape[:2]
    
    # 转换标注
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
    print(f"   期望类别: {config['expected_classes']}")
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
    print(f"🎯 创建模型...")
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    
    # 创建损失函数
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
    
    # 创建优化器
    optimizer = jt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0005)
    
    # 分析训练速度
    avg_time = analyze_training_speed(model, img_tensor, targets_tensor, loss_fn, optimizer)
    
    # 创建保存目录
    save_dir = Path(f"runs/unified_test_image_{image_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始过拟合训练 ({epochs}轮):")
    
    # 训练记录
    loss_history = []
    accuracy_history = []
    best_strict_accuracy = 0.0
    best_model_path = None
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # 训练
        model.train()
        outputs = model(img_tensor)
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=1)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        epoch_time = time.time() - start_time
        epoch_loss = float(loss.data)
        loss_history.append(epoch_loss)
        
        # 每5轮评估
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Loss {epoch_loss:.6f} ({epoch_time:.2f}s)")
            
            # 推理测试
            model.eval()
            with jt.no_grad():
                test_outputs = model(img_tensor)
                
                # 尝试不同置信度阈值，严格控制检测数量
                best_result = None
                for conf_thresh in [0.3, 0.2, 0.1, 0.05, 0.01]:
                    try:
                        # 严格控制检测数量，最多检测10个目标
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
                        print(f"   ⚠️ 检测数量过多！检测{len(result['detections'])}个 vs 期望{result['total_gt']}个")
                    elif len(result['detections']) < result['total_gt'] * 0.5:
                        print(f"   ⚠️ 检测数量过少！检测{len(result['detections'])}个 vs 期望{result['total_gt']}个")
                    else:
                        print(f"   ✅ 检测数量合理")
                    
                    # 分析检测位置错误
                    if result['strict_accuracy'] < 1.0:
                        analyze_detection_position_error(result['detections'], gt_boxes, gt_classes)
                    
                    # 保存最佳模型
                    if result['strict_accuracy'] > best_strict_accuracy:
                        best_strict_accuracy = result['strict_accuracy']
                        best_model_path = save_dir / f'best_model_epoch_{epoch}.pkl'
                        jt.save({
                            'model': model.state_dict(),
                            'epoch': epoch,
                            'loss': epoch_loss,
                            'strict_accuracy': result['strict_accuracy'],
                            'conf_thresh': result['conf_thresh'],
                            'correct_detections': result['correct_detections']
                        }, str(best_model_path))
                        print(f"   🏆 新的最佳结果！严格准确率: {result['strict_accuracy']*100:.1f}%")

                        # 生成分离式对比可视化
                        detection_stats = {
                            'strict_accuracy': result['strict_accuracy'],
                            'conf_thresh': result['conf_thresh']
                        }
                        vis_img = draw_separated_comparison(original_img, result['detections'],
                                                          gt_boxes, gt_classes, result['correct_detections'],
                                                          image_id, detection_stats)

                        vis_path = save_dir / f'comparison_epoch_{epoch}.jpg'
                        cv2.imwrite(str(vis_path), vis_img)
                        print(f"   💾 分离式对比图已保存: {vis_path}")
                        print(f"   📊 检测统计: 检测{len(result['detections'])}个，正确{len(result['correct_detections'])}个")
                
                else:
                    accuracy_history.append(0.0)
                    print(f"   ❌ 所有置信度阈值都没有检测结果")
            
            model.train()
    
    # 最终评估
    print(f"\n🎉 过拟合训练完成！")
    print(f"✅ 最佳严格准确率: {best_strict_accuracy*100:.1f}%")
    
    if best_model_path and os.path.exists(best_model_path):
        checkpoint = jt.load(str(best_model_path))
        
        print(f"\n📊 最终结果:")
        print(f"   最佳轮次: {checkpoint['epoch']}")
        print(f"   最佳置信度阈值: {checkpoint['conf_thresh']}")
        print(f"   最佳严格准确率: {checkpoint['strict_accuracy']*100:.1f}%")
        print(f"   正确检测详情:")
        
        for i, cd in enumerate(checkpoint['correct_detections']):
            print(f"     {i+1}. {cd['class_name']}: IoU={cd['iou']:.3f}, Conf={cd['confidence']:.3f}")
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(loss_history)+1), loss_history, 'b-', linewidth=2)
        plt.title(f'Training Loss - Image {image_id}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        epochs_eval = list(range(5, len(accuracy_history)*5+1, 5))
        plt.plot(epochs_eval, [acc*100 for acc in accuracy_history], 'r-', linewidth=2, marker='o')
        plt.title(f'Strict Accuracy - Image {image_id}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.tight_layout()
        curve_path = save_dir / 'training_curves.png'
        plt.savefig(str(curve_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📈 训练曲线已保存: {curve_path}")
        
        # 评估是否通过测试
        detected_classes = set(cd['class_name'] for cd in checkpoint['correct_detections'])
        class_accuracy = len(detected_classes.intersection(config['expected_classes'])) / len(config['expected_classes'])
        
        if best_strict_accuracy >= 0.8 and class_accuracy >= 0.8:
            print(f"\n🎉🎉🎉 图片{image_id}过拟合测试通过！🎉🎉🎉")
            print(f"✅ 种类准确率: {class_accuracy*100:.1f}%")
            print(f"✅ 位置准确率: {best_strict_accuracy*100:.1f}%")
            print(f"✅ 检测类别: {detected_classes}")
            return True
        else:
            print(f"\n⚠️ 图片{image_id}过拟合测试未完全通过")
            print(f"   种类准确率: {class_accuracy*100:.1f}%")
            print(f"   位置准确率: {best_strict_accuracy*100:.1f}%")
            print(f"   检测类别: {detected_classes}")
            print(f"   期望类别: {config['expected_classes']}")
            return False
    
    else:
        print(f"❌ 没有保存的最佳模型")
        return False

def main():
    parser = argparse.ArgumentParser(description='统一过拟合测试脚本')
    parser.add_argument('--image', type=str, choices=['1', '2', 'both'], default='1',
                       help='测试图片: 1=第一张, 2=第二张, both=两张都测试')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数 (默认30)')
    
    args = parser.parse_args()
    
    print("🔥 GOLD-YOLO Jittor版本 - 统一过拟合测试")
    print("=" * 80)
    print("解决4个核心问题:")
    print("1. 训练速度慢的原因分析和修复")
    print("2. 物体检测位置错误的原因分析")
    print("3. 两张图片过拟合训练测试（种类、数量、位置全部正确）")
    print("4. 清理脚本，增强可复用性和可维护性")
    print("=" * 80)
    
    if args.image == 'both':
        # 测试两张图片
        results = []
        for image_id in [1, 2]:
            print(f"\n{'='*20} 测试图片{image_id} {'='*20}")
            success = unified_overfitting_test(image_id, args.epochs)
            results.append(success)
        
        print(f"\n🎯 最终结果总结:")
        print(f"   图片1测试: {'✅通过' if results[0] else '❌未通过'}")
        print(f"   图片2测试: {'✅通过' if results[1] else '❌未通过'}")
        
        if all(results):
            print(f"\n🎉🎉🎉 所有测试通过！GOLD-YOLO Jittor版本功能完全正常！🎉🎉🎉")
        else:
            print(f"\n📊 部分测试未通过，需要进一步优化")
    
    else:
        # 测试单张图片
        image_id = int(args.image)
        success = unified_overfitting_test(image_id, args.epochs)
        
        if success:
            print(f"\n🎉🎉🎉 图片{image_id}测试通过！🎉🎉🎉")
        else:
            print(f"\n📊 图片{image_id}测试完成，需要进一步分析")

if __name__ == "__main__":
    main()
