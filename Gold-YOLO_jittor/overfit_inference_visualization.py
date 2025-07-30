#!/usr/bin/env python3
"""
过拟合模型推理可视化
训练模型并展示推理结果
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import time
import math

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.pytorch_aligned_losses import ComputeLoss
from yolov6.utils.nms import non_max_suppression

# VOC数据集类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def cosine_lr_scheduler(epoch, total_epochs, lr0, lrf):
    """Cosine学习率调度器"""
    return lrf + (lr0 - lrf) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

def draw_bbox(img, bbox, label, conf, color=(0, 255, 0)):
    """绘制边界框"""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 绘制标签
    label_text = f"{VOC_CLASSES[int(label)]}: {conf:.2f}"
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def train_and_visualize():
    """训练模型并可视化推理结果"""
    print(f"🎯 过拟合训练并可视化推理结果")
    print("=" * 80)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    # 读取标注
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
    
    # 读取图像
    original_img = cv2.imread(img_path)
    img_height, img_width = original_img.shape[:2]
    
    # 预处理图像
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
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
    
    print(f"📊 数据准备:")
    print(f"   原始图像尺寸: {img_width}x{img_height}")
    print(f"   预处理后尺寸: {img.shape}")
    print(f"   目标数量: {len(annotations)}个")
    for i, ann in enumerate(annotations):
        cls_id, x_center, y_center, width, height = ann
        print(f"     目标{i+1}: {VOC_CLASSES[cls_id]} ({x_center:.3f},{y_center:.3f}) {width:.3f}x{height:.3f}")
    
    # 创建模型
    print(f"\n🎯 创建模型:")
    model = create_perfect_gold_yolo_model()
    model.train()
    
    # 创建损失函数
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=500,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    
    # 使用PyTorch版本的超参数
    lr0 = 0.02
    lrf = 0.01
    momentum = 0.937
    weight_decay = 0.0005
    total_epochs = 100  # 减少轮次以便快速看到结果
    
    optimizer = jt.optim.SGD(
        model.parameters(), 
        lr=lr0, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    
    print(f"\n🔧 开始过拟合训练:")
    print(f"   训练轮次: {total_epochs}")
    print(f"   学习率: {lr0} -> {lrf} (Cosine)")
    print("-" * 60)
    
    # 训练过程
    loss_history = []
    
    for epoch in range(total_epochs):
        # 更新学习率
        current_lr = cosine_lr_scheduler(epoch, total_epochs, lr0, lrf)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
        
        # 反向传播
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        loss_value = float(loss.data.item())
        loss_history.append(loss_value)
        
        # 每20轮打印一次
        if (epoch + 1) % 20 == 0:
            loss_items_values = [float(item.data.item()) for item in loss_items]
            print(f"   轮次 {epoch+1:3d}: 总损失={loss_value:.6f}, IoU={loss_items_values[0]:.4f}, 分类={loss_items_values[2]:.4f}")
    
    # 训练完成，计算损失下降
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\n📊 训练完成:")
    print(f"   初始损失: {initial_loss:.6f}")
    print(f"   最终损失: {final_loss:.6f}")
    print(f"   损失下降: {loss_reduction:.1f}%")
    
    # 切换到推理模式
    print(f"\n🔍 开始推理可视化:")
    model.eval()
    
    with jt.no_grad():
        # 推理
        outputs = model(img_tensor)

        # 输出格式: [1, 5249, 25] (x, y, w, h, conf, cls0, cls1, ..., cls19)
        predictions = outputs

        # 确保是3维张量 [batch, anchors, features]
        if len(predictions.shape) == 2:
            predictions = predictions.unsqueeze(0)  # [5249, 25] -> [1, 5249, 25]
        
        print(f"   推理输出形状: {predictions.shape}")
        print(f"   预测范围: [{float(predictions.min()):.6f}, {float(predictions.max()):.6f}]")
        
        # 应用NMS
        pred_results = non_max_suppression(
            predictions,
            conf_thres=0.01,  # 进一步降低置信度阈值
            iou_thres=0.5,
            max_det=100
        )
        
        print(f"   NMS后检测数量: {len(pred_results[0]) if pred_results[0] is not None else 0}")
        
        # 可视化结果
        vis_img = original_img.copy()
        
        # 绘制GT框 (绿色)
        print(f"\n📋 绘制GT框 (绿色):")
        for i, ann in enumerate(annotations):
            cls_id, x_center, y_center, width, height = ann
            
            # 转换为像素坐标
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height
            
            x1 = x_center_px - width_px / 2
            y1 = y_center_px - height_px / 2
            x2 = x_center_px + width_px / 2
            y2 = y_center_px + height_px / 2
            
            draw_bbox(vis_img, [x1, y1, x2, y2], cls_id, 1.0, color=(0, 255, 0))
            print(f"   GT{i+1}: {VOC_CLASSES[cls_id]} ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")
        
        # 绘制预测框 (红色)
        print(f"\n🎯 绘制预测框 (红色):")
        if pred_results[0] is not None and len(pred_results[0]) > 0:
            detections = pred_results[0].data  # [N, 6] (x1, y1, x2, y2, conf, cls)
            
            # 缩放到原始图像尺寸
            scale_x = img_width / 500
            scale_y = img_height / 500
            
            for i, det in enumerate(detections):
                # 检查det的形状
                if len(det) == 6:
                    x1, y1, x2, y2, conf, cls_id = det
                elif len(det) == 1:
                    # 如果det是单个元素，可能需要进一步解包
                    det_data = det[0] if hasattr(det[0], '__len__') and len(det[0]) == 6 else det
                    if len(det_data) == 6:
                        x1, y1, x2, y2, conf, cls_id = det_data
                    else:
                        print(f"     预测{i+1}: 格式错误，跳过")
                        continue
                else:
                    print(f"     预测{i+1}: 未知格式 (长度={len(det)})，跳过")
                    continue
                
                # 缩放坐标
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                
                draw_bbox(vis_img, [x1, y1, x2, y2], cls_id, conf, color=(0, 0, 255))
                print(f"   预测{i+1}: {VOC_CLASSES[int(cls_id)]} {conf:.3f} ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")
        else:
            print(f"   ❌ 没有检测到任何目标")
        
        # 保存可视化结果
        output_path = "overfit_visualization_result.jpg"
        cv2.imwrite(output_path, vis_img)
        print(f"\n💾 可视化结果已保存: {output_path}")
        
        # 添加图例 - 修复尺寸匹配问题
        legend_height = 100
        legend_img = np.zeros((legend_height, img_width, 3), dtype=np.uint8)  # 使用原始图像宽度
        cv2.putText(legend_img, "Green: Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(legend_img, "Red: Predictions", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(legend_img, f"Loss Reduction: {loss_reduction:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 合并图像和图例
        combined_img = np.vstack([vis_img, legend_img])
        combined_path = "overfit_visualization_with_legend.jpg"
        cv2.imwrite(combined_path, combined_img)
        print(f"💾 带图例的结果已保存: {combined_path}")
        
        return loss_reduction, len(pred_results[0]) if pred_results[0] is not None else 0

def main():
    print("🎯 过拟合模型推理可视化")
    print("=" * 80)
    
    try:
        loss_reduction, num_detections = train_and_visualize()
        
        print(f"\n" + "=" * 80)
        print(f"📊 过拟合推理可视化结果:")
        print(f"=" * 80)
        print(f"   损失下降: {loss_reduction:.1f}%")
        print(f"   检测数量: {num_detections}")
        
        if loss_reduction > 50:
            print(f"   ✅ 模型训练成功")
        else:
            print(f"   ⚠️ 模型训练效果一般")
        
        if num_detections > 0:
            print(f"   ✅ 模型能够检测目标")
        else:
            print(f"   ❌ 模型未检测到目标")
        
        print(f"\n🎯 请查看生成的可视化图像:")
        print(f"   - overfit_visualization_result.jpg")
        print(f"   - overfit_visualization_with_legend.jpg")
        
    except Exception as e:
        print(f"\n❌ 可视化异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
