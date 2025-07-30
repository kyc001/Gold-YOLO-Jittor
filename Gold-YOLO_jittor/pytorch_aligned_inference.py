#!/usr/bin/env python3
"""
完全对齐PyTorch版本的推理脚本
训练模型并展示预测结果
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

def train_and_infer():
    """训练模型并进行推理"""
    print(f"🎯 PyTorch对齐版本：训练并推理")
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
    
    # 预处理图像 - 对齐PyTorch版本
    img_size = 640  # 使用PyTorch版本的默认尺寸
    img = letterbox(original_img, new_shape=img_size, stride=32, auto=False)[0]
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
    
    # 创建损失函数 - 完全对齐PyTorch版本
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=img_size,
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
    total_epochs = 50  # 减少轮次以便快速看到结果
    
    optimizer = jt.optim.SGD(
        model.parameters(), 
        lr=lr0, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    
    print(f"\n🔧 开始训练:")
    print(f"   训练轮次: {total_epochs}")
    print(f"   学习率: {lr0} -> {lrf} (Cosine)")
    print(f"   图像尺寸: {img_size}x{img_size}")
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
        
        # 每10轮打印一次
        if (epoch + 1) % 10 == 0:
            loss_items_values = [float(item.data.item()) for item in loss_items]
            print(f"   轮次 {epoch+1:2d}: 总损失={loss_value:.6f}, IoU={loss_items_values[0]:.4f}, 分类={loss_items_values[2]:.4f}")
    
    # 训练完成，计算损失下降
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\n📊 训练完成:")
    print(f"   初始损失: {initial_loss:.6f}")
    print(f"   最终损失: {final_loss:.6f}")
    print(f"   损失下降: {loss_reduction:.1f}%")
    
    # 切换到推理模式 - 对齐PyTorch版本
    print(f"\n🔍 开始推理 (对齐PyTorch版本):")
    model.eval()
    
    with jt.no_grad():
        # 推理
        outputs = model(img_tensor)
        
        # 确保输出格式正确
        if len(outputs.shape) == 2:
            outputs = outputs.unsqueeze(0)  # [anchors, features] -> [1, anchors, features]
        
        print(f"   推理输出形状: {outputs.shape}")
        print(f"   预测范围: [{float(outputs.min()):.6f}, {float(outputs.max()):.6f}]")
        
        # 应用NMS - 对齐PyTorch版本参数
        pred_results = non_max_suppression(
            outputs, 
            conf_thres=0.4,   # 对齐PyTorch版本默认值
            iou_thres=0.45,   # 对齐PyTorch版本默认值
            max_det=1000      # 对齐PyTorch版本默认值
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
            detections = pred_results[0]
            
            # 缩放到原始图像尺寸
            scale_x = img_width / img_size
            scale_y = img_height / img_size
            
            print(f"   缩放因子: x={scale_x:.3f}, y={scale_y:.3f}")
            
            # 检查detections的格式
            print(f"   检测结果格式: {type(detections)}, 形状: {detections.shape if hasattr(detections, 'shape') else 'N/A'}")
            
            # 修复NMS输出格式问题
            try:
                detections_data = detections.data if hasattr(detections, 'data') else detections

                # 检查并修复形状 [240,1,6] -> [240,6]
                if len(detections_data.shape) == 3 and detections_data.shape[1] == 1:
                    detections_data = detections_data.squeeze(1)  # [240,1,6] -> [240,6]

                print(f"   修复后检测结果形状: {detections_data.shape}")

                for i in range(min(10, len(detections_data))):  # 最多显示10个检测
                    det = detections_data[i]

                    # 检查det的格式
                    if hasattr(det, '__len__') and len(det) >= 6:
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                        
                        # 缩放坐标
                        x1 = float(x1) * scale_x
                        y1 = float(y1) * scale_y
                        x2 = float(x2) * scale_x
                        y2 = float(y2) * scale_y
                        conf = float(conf)
                        cls_id = int(cls_id)
                        
                        # 检查坐标和类别是否合理
                        if 0 <= cls_id < len(VOC_CLASSES) and conf > 0.1:
                            draw_bbox(vis_img, [x1, y1, x2, y2], cls_id, conf, color=(0, 0, 255))
                            print(f"   预测{i+1}: {VOC_CLASSES[cls_id]} {conf:.3f} ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")
                        else:
                            print(f"   预测{i+1}: 无效检测 (cls={cls_id}, conf={conf:.3f})")
                    else:
                        print(f"   预测{i+1}: 格式错误 (长度={len(det) if hasattr(det, '__len__') else 'N/A'})")
                        
            except Exception as e:
                print(f"   ❌ 解析检测结果失败: {e}")
                print(f"   检测结果类型: {type(detections)}")
                if hasattr(detections, 'shape'):
                    print(f"   检测结果形状: {detections.shape}")
        else:
            print(f"   ❌ 没有检测到任何目标")
            print(f"   可能原因: 1) 置信度阈值过高 2) 模型训练不足 3) NMS参数不当")
        
        # 保存可视化结果
        output_path = "pytorch_aligned_inference_result.jpg"
        cv2.imwrite(output_path, vis_img)
        print(f"\n💾 推理结果已保存: {output_path}")
        
        # 添加图例
        legend_height = 120
        legend_img = np.zeros((legend_height, img_width, 3), dtype=np.uint8)
        cv2.putText(legend_img, "Green: Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(legend_img, "Red: Predictions", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(legend_img, f"Loss Reduction: {loss_reduction:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(legend_img, f"PyTorch Aligned Inference", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 合并图像和图例
        combined_img = np.vstack([vis_img, legend_img])
        combined_path = "pytorch_aligned_inference_with_legend.jpg"
        cv2.imwrite(combined_path, combined_img)
        print(f"💾 带图例的结果已保存: {combined_path}")
        
        return loss_reduction, len(pred_results[0]) if pred_results[0] is not None else 0

def main():
    print("🎯 PyTorch对齐版本推理测试")
    print("=" * 80)
    
    try:
        loss_reduction, num_detections = train_and_infer()
        
        print(f"\n" + "=" * 80)
        print(f"📊 PyTorch对齐版本推理结果:")
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
            print(f"   ❌ 模型未检测到目标，可能需要:")
            print(f"     1. 降低置信度阈值")
            print(f"     2. 增加训练轮次")
            print(f"     3. 调整损失权重")
        
        print(f"\n🎯 请查看生成的推理图像:")
        print(f"   - pytorch_aligned_inference_result.jpg")
        print(f"   - pytorch_aligned_inference_with_legend.jpg")
        
    except Exception as e:
        print(f"\n❌ 推理异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
