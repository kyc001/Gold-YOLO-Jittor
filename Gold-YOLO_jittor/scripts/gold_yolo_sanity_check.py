#!/usr/bin/env python3
"""
Gold-YOLO Jittor 终极流程自检脚本
通过过拟合测试验证Gold-YOLO Jittor版本的功能是否正确
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import gc

# 设置matplotlib支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import jittor.nn as nn

# 设置Jittor
jt.flags.use_cuda = 1
jt.set_global_seed(42)

# COCO类别名称映射
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
    48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
    63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

def select_random_image():
    """从数据集中随机选择一张图片"""
    data_dir = "/home/kyc/project/GOLD-YOLO/data/coco2017_50/train2017"
    annotation_path = "/home/kyc/project/GOLD-YOLO/data/coco2017_50/annotations/instances_train2017.json"
    
    if not os.path.exists(data_dir) or not os.path.exists(annotation_path):
        print("❌ 数据集不存在")
        return None, None, None
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    if not image_files:
        print("❌ 数据集中没有图片")
        return None, None, None
    
    # 随机选择一张图片
    selected_image = random.choice(image_files)
    image_path = os.path.join(data_dir, selected_image)
    image_id = int(selected_image.split('.')[0])
    
    print(f"🎯 随机选择图片: {selected_image} (ID: {image_id})")
    
    return image_path, image_id, annotation_path

def load_and_verify_data(image_path, image_id, annotation_path):
    """加载并验证数据"""
    print("🔍 加载数据...")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    print(f"原始图像尺寸: {original_width}x{original_height}")
    
    # 调整图像大小到640x640
    image_resized = image.resize((640, 640), Image.LANCZOS)
    
    # 转换为张量
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_tensor = jt.array(img_array.transpose(2, 0, 1)).float32().unsqueeze(0)
    
    # 加载标注
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    annotations = []
    labels = []
    
    print(f"查找图像ID {image_id} 的标注...")
    
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']
            
            print(f"找到标注: 类别{category_id}, 边界框[{x},{y},{w},{h}]")
            
            # 归一化坐标
            x1, y1 = x / original_width, y / original_height
            x2, y2 = (x + w) / original_width, (y + h) / original_height
            
            # 确保坐标有效
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1:
                annotations.append([x1, y1, x2, y2])
                labels.append(category_id - 1)  # 转换为0-based索引
                print(f"   归一化后: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}], 类别索引: {category_id-1}")
    
    if not annotations:
        print("⚠️ 该图片没有有效标注，重新选择图片")
        return None, None, None
    
    # 创建目标
    target = {
        'boxes': jt.array(annotations, dtype=jt.float32),
        'labels': jt.array(labels, dtype=jt.int64)
    }
    
    print(f"✅ 数据加载成功")
    print(f"   图像: {img_tensor.shape}")
    print(f"   目标: {len(annotations)}个边界框")
    print(f"   类别: {labels}")
    
    return img_tensor, target, image

def create_and_test_model():
    """创建并测试Gold-YOLO模型"""
    print("\n" + "=" * 60)
    print("===        Gold-YOLO模型创建和测试        ===")
    print("=" * 60)
    
    try:
        from models.yolo import Model
        from models.loss import GoldYOLOLoss
        from configs.gold_yolo_s import get_config

        # 加载配置
        config = get_config()

        # 创建模型
        model = Model(config=config, channels=3, num_classes=80)
        criterion = GoldYOLOLoss(num_classes=80)
        
        print(f"✅ Gold-YOLO模型创建成功")
        
        # 测试前向传播
        img_tensor, target, original_image = load_and_verify_data(*select_random_image())
        if img_tensor is None:
            return None, None, None, None, None
        
        print("\n测试前向传播...")
        model.train()
        outputs = model(img_tensor)
        
        print(f"✅ 前向传播成功")
        print(f"   输出类型: {type(outputs)}")
        if isinstance(outputs, list):
            print(f"   输出列表长度: {len(outputs)}")
            if len(outputs) >= 2:
                detection_output, featmaps = outputs
                print(f"   检测输出类型: {type(detection_output)}")
                print(f"   特征图数量: {len(featmaps)}")
        
        # 测试损失计算
        print("\n测试损失计算...")
        batch = {'cls': target['labels'].unsqueeze(0), 'bboxes': target['boxes'].unsqueeze(0)}
        loss, loss_items = criterion(outputs, batch)
        
        print(f"✅ 损失计算成功: {loss.item():.4f}")
        print(f"   损失分量: {loss_items.numpy()}")

        # 测试训练前的推理能力
        print("\n测试训练前推理能力...")
        model.eval()
        with jt.no_grad():
            outputs = model(img_tensor)

        detections = strict_post_process(outputs, conf_thres=0.3, iou_thres=0.5, max_det=20)
        det = detections[0]

        print(f"✅ 训练前推理成功")
        print(f"   检测到 {det.shape[0]} 个目标")

        if det.shape[0] > 0:
            print(f"   置信度范围: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
            print(f"   检测类别: {set(det[:, 5].numpy().astype(int))}")
            print("🎉 训练前模型具备检测能力！")
        else:
            print("⚠️ 训练前模型无检测能力")

        # 切换回训练模式
        model.train()

        return model, criterion, img_tensor, target, original_image
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def intensive_overfitting_test(model, criterion, img_tensor, target):
    """强化过拟合训练测试"""
    print("\n" + "=" * 60)
    print("===        强化过拟合训练测试        ===")
    print("=" * 60)
    
    try:
        # 设置训练模式
        model.train()
        
        # 创建优化器
        optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)
        
        # 准备batch数据
        batch = {'cls': target['labels'].unsqueeze(0), 'bboxes': target['boxes'].unsqueeze(0)}
        
        print(f"开始100次过拟合训练...")
        print(f"目标: 模型必须能够完美记住这张图像的所有目标")
        
        losses = []
        num_epochs = 100
        
        for epoch in range(num_epochs):
            # 前向传播
            outputs = model(img_tensor)
            
            # 损失计算
            loss, loss_items = criterion(outputs, batch)
            losses.append(loss.item())
            
            # 反向传播和参数更新 - 使用Jittor的正确语法
            optimizer.step(loss)
            
            # 打印进度
            if epoch % 20 == 0 or epoch < 5 or epoch >= 95:
                print(f"Epoch {epoch:3d}: 损失={loss.item():.4f}")
        
        print(f"\n✅ 100次过拟合训练完成")
        print(f"   初始损失: {losses[0]:.4f}")
        print(f"   最终损失: {losses[-1]:.4f}")
        print(f"   损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
        print(f"   最低损失: {min(losses):.4f}")
        
        # 判断过拟合效果
        loss_reduction = (losses[0] - losses[-1]) / losses[0]
        training_success = loss_reduction > 0.1  # 损失下降超过10%
        
        if training_success:
            print("🎉 过拟合成功！模型已经学习了这张图像")
        else:
            print(f"⚠️ 过拟合效果有限，损失下降仅{loss_reduction*100:.1f}%")
        
        return training_success, losses
        
    except Exception as e:
        print(f"❌ 过拟合训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def strict_post_process(prediction, conf_thres=0.5, iou_thres=0.5, max_det=50):
    """严格的后处理函数 - 确保检测结果合理"""

    # 输入格式: [batch, num_anchors, 5+num_classes]
    # 输出格式: [batch, num_detections, 6] (x1, y1, x2, y2, conf, cls)

    batch_size = prediction.shape[0]
    num_anchors = prediction.shape[1]
    num_classes = prediction.shape[2] - 5

    output = []

    for i in range(batch_size):
        pred = prediction[i]  # [num_anchors, 5+num_classes]

        # 转换为numpy进行处理
        pred_np = pred.numpy()

        # 提取各部分
        boxes = pred_np[:, :4]  # [x1, y1, x2, y2]
        obj_conf = pred_np[:, 4]  # 目标置信度
        cls_scores = pred_np[:, 5:]  # 类别分数

        # 计算最终置信度和类别
        class_conf = np.max(cls_scores, axis=1)
        class_pred = np.argmax(cls_scores, axis=1)
        final_conf = obj_conf * class_conf

        # 严格的置信度过滤
        valid_mask = final_conf > conf_thres

        if not np.any(valid_mask):
            output.append(jt.zeros((0, 6)))
            continue

        # 应用掩码
        valid_boxes = boxes[valid_mask]
        valid_conf = final_conf[valid_mask]
        valid_cls = class_pred[valid_mask]

        # 过滤无效边界框
        valid_indices = []
        for idx, box in enumerate(valid_boxes):
            x1, y1, x2, y2 = box
            # 检查边界框是否有效
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                # 检查边界框大小是否合理
                width = x2 - x1
                height = y2 - y1
                area = width * height
                if area > 1:  # 最小面积阈值 (降低要求)
                    valid_indices.append(idx)

        if len(valid_indices) == 0:
            output.append(jt.zeros((0, 6)))
            continue

        # 应用有效性过滤
        valid_boxes = valid_boxes[valid_indices]
        valid_conf = valid_conf[valid_indices]
        valid_cls = valid_cls[valid_indices]

        # 简化的NMS - 按置信度排序并移除重叠框
        sort_indices = np.argsort(valid_conf)[::-1]  # 降序
        valid_boxes = valid_boxes[sort_indices]
        valid_conf = valid_conf[sort_indices]
        valid_cls = valid_cls[sort_indices]

        # 简单的NMS实现
        keep_indices = []
        for i in range(len(valid_boxes)):
            if i == 0:
                keep_indices.append(i)
                continue

            # 计算与已保留框的IoU
            current_box = valid_boxes[i]
            should_keep = True

            for kept_idx in keep_indices:
                kept_box = valid_boxes[kept_idx]
                iou = calculate_iou(current_box, kept_box)
                if iou > iou_thres:
                    should_keep = False
                    break

            if should_keep:
                keep_indices.append(i)

        # 应用NMS结果
        if len(keep_indices) > 0:
            final_boxes = valid_boxes[keep_indices]
            final_conf = valid_conf[keep_indices]
            final_cls = valid_cls[keep_indices]

            # 限制检测数量
            if len(final_conf) > max_det:
                final_boxes = final_boxes[:max_det]
                final_conf = final_conf[:max_det]
                final_cls = final_cls[:max_det]

            # 组合结果
            detections = np.column_stack([
                final_boxes,  # [x1, y1, x2, y2]
                final_conf.reshape(-1, 1),  # confidence
                final_cls.reshape(-1, 1)    # class
            ])
            output.append(jt.array(detections))
        else:
            output.append(jt.zeros((0, 6)))

    return output

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def scale_coords(img1_shape, coords, img0_shape):
    """将坐标从img1_shape缩放到img0_shape"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # 限制坐标范围
    coords[:, 0] = jt.clamp(coords[:, 0], 0, img0_shape[1])  # x1
    coords[:, 1] = jt.clamp(coords[:, 1], 0, img0_shape[0])  # y1
    coords[:, 2] = jt.clamp(coords[:, 2], 0, img0_shape[1])  # x2
    coords[:, 3] = jt.clamp(coords[:, 3], 0, img0_shape[0])  # y2

    return coords

def inference_and_visualization_test(model, img_tensor, target, original_image):
    """推理和可视化测试 - 完整版本"""
    print("\n" + "=" * 60)
    print("===        推理和可视化测试        ===")
    print("=" * 60)

    try:
        # 设置评估模式
        model.eval()

        # 推理
        with jt.no_grad():
            outputs = model(img_tensor)

        print(f"✅ 推理成功")
        print(f"   输出类型: {type(outputs)}")
        print(f"   输出形状: {outputs.shape}")

        # 后处理
        print("🔄 开始后处理...")
        detections = strict_post_process(outputs, conf_thres=0.3, iou_thres=0.5, max_det=20)
        det = detections[0]  # 第一个batch

        print(f"   检测到 {det.shape[0]} 个目标")

        # 坐标缩放
        if det.shape[0] > 0:
            # 从640x640缩放到原始图像尺寸
            det[:, :4] = scale_coords((640, 640), det[:, :4], (original_image.height, original_image.width))

        # 生成可视化
        save_path = "/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/experiments/detection_visualization.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # 左侧：原图和真实标注
        ax1.imshow(original_image)
        ax1.set_title('Ground Truth', fontsize=16, fontweight='bold')
        ax1.axis('off')

        # 绘制真实边界框
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()

        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            x1, y1, x2, y2 = box
            x1 *= original_image.width
            y1 *= original_image.height
            x2 *= original_image.width
            y2 *= original_image.height

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='green', facecolor='none'
            )
            ax1.add_patch(rect)

            class_name = COCO_CLASSES.get(label + 1, f'class_{label + 1}')
            ax1.text(
                x1, y1 - 10, f'GT: {class_name}',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.8),
                fontsize=12, fontweight='bold', color='white'
            )

        # 右侧：预测结果
        ax2.imshow(original_image)
        ax2.set_title(f'Predictions ({det.shape[0]} detections)', fontsize=16, fontweight='bold')
        ax2.axis('off')

        # 绘制预测边界框
        if det.shape[0] > 0:
            det_np = det.numpy()
            for i, detection in enumerate(det_np):
                x1, y1, x2, y2, conf, cls = detection

                # 只显示高置信度的检测
                if conf > 0.3:
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=3, edgecolor='red', facecolor='none'
                    )
                    ax2.add_patch(rect)

                    class_name = COCO_CLASSES.get(int(cls) + 1, f'class_{int(cls) + 1}')
                    ax2.text(
                        x1, y1 - 10, f'{class_name}: {conf:.2f}',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                        fontsize=12, fontweight='bold', color='white'
                    )
        else:
            ax2.text(
                original_image.width // 2, original_image.height // 2,
                'No detections found\n(confidence threshold: 0.3)',
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round,pad=1', facecolor='orange', alpha=0.8)
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 可视化结果已保存到: {save_path}")

        # 严格的检测效果评估
        print("\n📊 严格检测效果评估:")
        print(f"   真实目标数量: {len(gt_labels)}")
        print(f"   检测目标数量: {det.shape[0]}")

        # 自检成功的严格标准
        success_criteria = {
            'quantity_match': False,
            'category_match': False,
            'position_match': False
        }

        if det.shape[0] > 0:
            high_conf_dets = det[det[:, 4] > 0.3]
            print(f"   高置信度检测 (>0.3): {high_conf_dets.shape[0]}")

            if high_conf_dets.shape[0] > 0:
                print(f"   最高置信度: {high_conf_dets[:, 4].max().item():.3f}")
                print(f"   平均置信度: {high_conf_dets[:, 4].mean().item():.3f}")

                # 1. 检查数量匹配 (允许±1的误差)
                quantity_diff = abs(high_conf_dets.shape[0] - len(gt_labels))
                if quantity_diff <= 1:
                    success_criteria['quantity_match'] = True
                    print(f"   ✅ 数量匹配: 检测{high_conf_dets.shape[0]}个 vs 真实{len(gt_labels)}个")
                else:
                    print(f"   ❌ 数量不匹配: 检测{high_conf_dets.shape[0]}个 vs 真实{len(gt_labels)}个")

                # 2. 检查类别匹配
                detected_classes = set(high_conf_dets[:, 5].numpy().astype(int))
                gt_classes = set(gt_labels)

                # 计算类别匹配度
                intersection = detected_classes.intersection(gt_classes)
                union = detected_classes.union(gt_classes)
                class_match_ratio = len(intersection) / len(union) if len(union) > 0 else 0

                if class_match_ratio >= 0.5:  # 至少50%的类别匹配
                    success_criteria['category_match'] = True
                    print(f"   ✅ 类别匹配: {class_match_ratio:.1%} (检测到: {detected_classes}, 真实: {gt_classes})")
                else:
                    print(f"   ❌ 类别不匹配: {class_match_ratio:.1%} (检测到: {detected_classes}, 真实: {gt_classes})")

                # 3. 检查位置匹配 (简化版本)
                if len(gt_labels) > 0 and high_conf_dets.shape[0] > 0:
                    # 计算检测框与真实框的最大IoU
                    max_ious = []
                    gt_boxes_scaled = target['boxes'].numpy()

                    for gt_box in gt_boxes_scaled:
                        gt_x1 = gt_box[0] * original_image.width
                        gt_y1 = gt_box[1] * original_image.height
                        gt_x2 = gt_box[2] * original_image.width
                        gt_y2 = gt_box[3] * original_image.height
                        gt_box_scaled = [gt_x1, gt_y1, gt_x2, gt_y2]

                        best_iou = 0
                        for det_box in high_conf_dets[:, :4].numpy():
                            iou = calculate_iou(det_box, gt_box_scaled)
                            best_iou = max(best_iou, iou)
                        max_ious.append(best_iou)

                    avg_iou = np.mean(max_ious) if max_ious else 0
                    if avg_iou >= 0.3:  # 平均IoU >= 0.3
                        success_criteria['position_match'] = True
                        print(f"   ✅ 位置匹配: 平均IoU = {avg_iou:.3f}")
                    else:
                        print(f"   ❌ 位置不匹配: 平均IoU = {avg_iou:.3f}")
                else:
                    print(f"   ❌ 无法计算位置匹配")
        else:
            print(f"   ❌ 没有检测到任何目标")

        # 综合评估
        passed_criteria = sum(success_criteria.values())
        total_criteria = len(success_criteria)

        print(f"\n🎯 自检结果: {passed_criteria}/{total_criteria} 项通过")
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")

        plt.close()  # 关闭图形以节省内存

        # 只有所有标准都通过才算成功
        return passed_criteria == total_criteria

    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_failed_model():
    """清理失败的模型"""
    print("🧹 清理失败的模型...")
    gc.collect()
    jt.gc()

def main():
    """主函数 - 完整的自检流程"""
    print("🎯 Gold-YOLO Jittor 终极流程自检")
    print("=" * 80)
    print("目标: 通过过拟合测试验证Gold-YOLO功能是否正确")
    print("要求: 能够准确识别图片中的物体类别和数目，且检测框位置合理")
    print("=" * 80)

    max_attempts = 1  # 最多尝试3次
    attempt = 1

    while attempt <= max_attempts:
        print(f"\n🔄 第{attempt}次尝试:")
        print("=" * 60)

        try:
            # 1. 创建和测试模型
            model, criterion, img_tensor, target, original_image = create_and_test_model()
            if model is None:
                print("❌ 模型创建失败，尝试下一次")
                attempt += 1
                continue

            # 2. 强化过拟合训练
            training_success, losses = intensive_overfitting_test(model, criterion, img_tensor, target)

            # 3. 推理和可视化测试
            inference_success = inference_and_visualization_test(model, img_tensor, target, original_image)

            # 检查是否通过验证
            if training_success and inference_success:
                print("\n" + "=" * 80)
                print("🎉 Gold-YOLO自检完全成功！")
                print("=" * 80)
                print("✅ 过拟合训练成功")
                print("✅ 推理和可视化成功")
                print("✅ Gold-YOLO Jittor版本功能正确")

                if losses:
                    print(f"\n📊 训练统计:")
                    print(f"  初始损失: {losses[0]:.4f}")
                    print(f"  最终损失: {losses[-1]:.4f}")
                    print(f"  损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
                    print(f"  最低损失: {min(losses):.4f}")

                print("\n🚀 结论: Gold-YOLO Jittor版本可用于生产环境！")
                print("=" * 80)
                return True

            else:
                print("\n" + "=" * 80)
                print(f"❌ 第{attempt}次尝试失败")
                print("=" * 80)

                if not training_success:
                    print("❌ 过拟合训练效果不足")
                    if losses:
                        print(f"   损失下降: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
                        print("   需要: 损失下降 > 10%")

                if not inference_success:
                    print("❌ 推理验证失败")

                if attempt < max_attempts:
                    print(f"\n🔄 准备第{attempt+1}次尝试...")
                    print("💡 将重新初始化模型并调整参数")

                    # 清理失败的模型
                    cleanup_failed_model()
                else:
                    print("\n❌ 已达到最大尝试次数")
                    print("💡 Gold-YOLO可能存在问题，需要深入检查")

        except Exception as e:
            print(f"❌ 第{attempt}次尝试出现异常: {e}")
            import traceback
            traceback.print_exc()
            cleanup_failed_model()

        attempt += 1

    print("\n" + "=" * 80)
    print("❌ Gold-YOLO自检最终失败")
    print("=" * 80)
    print("Gold-YOLO无法通过过拟合验证测试")
    print("建议检查:")
    print("1. 模型架构是否正确")
    print("2. 损失函数是否有效")
    print("3. 梯度传播是否正常")
    print("4. 数据预处理是否正确")
    print("=" * 80)
    return False

if __name__ == "__main__":
    # 设置随机种子确保可重现性
    random.seed(42)
    np.random.seed(42)
    jt.set_global_seed(42)

    success = main()

    if success:
        print("\n🎊 恭喜！Gold-YOLO Jittor版本通过了所有测试！")
    else:
        print("\n😞 很遗憾，Gold-YOLO Jittor版本未能通过测试。")
        print("请根据上述建议进行修复后重新测试。")
