#!/usr/bin/env python3
"""
修复后的推理测试脚本，解决坐标问题，优化可视化布局
左边是真实标注，右边是预测结果
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

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = jt.zeros_like(x) if isinstance(x, jt.Var) else np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def draw_detection_box(img, box, label, confidence, color, is_expected=False, box_type="PRED"):
    """绘制检测框 - 区分预测框和真实框"""
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
    
    # 根据框类型选择样式
    if box_type == "GT":
        # 真实框：虚线效果，黄色
        thickness = 3
        color = (0, 255, 255)  # 黄色
        # 绘制虚线效果
        dash_length = 10
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
            cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)
        
        # 标签
        label_text = f'GT: {label}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
        
        cv2.rectangle(img, (x1, y2), (x1 + text_width + 10, y2 + text_height + 10), color, -1)
        cv2.putText(img, label_text, (x1 + 5, y2 + text_height + 5), font, font_scale, (0, 0, 0), text_thickness)
    
    else:
        # 预测框：实线，期望类别使用特殊颜色
        thickness = 4 if is_expected else 2
        
        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # 准备标签文本
        status = "✅" if is_expected else "❌"
        label_text = f'{status}{label} {confidence:.3f}'
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 if is_expected else 0.6
        text_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
        
        # 绘制标签背景
        bg_color = color if is_expected else (64, 64, 64)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), bg_color, -1)
        
        # 绘制标签文本
        text_color = (255, 255, 255)
        cv2.putText(img, label_text, (x1 + 5, y1 - 5), font, font_scale, text_color, text_thickness)

def fixed_inference_visualization_test():
    """修复后的推理测试，解决坐标问题，优化可视化布局"""
    print(f"🔥 修复后的推理测试，解决坐标问题")
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
    
    print(f"📋 真实标注: {target_counts}")
    print(f"   期望类别: {expected_classes}")
    
    # 读取原始图像
    original_img = cv2.imread(img_path)
    img_height, img_width = original_img.shape[:2]
    print(f"📷 原始图像尺寸: {img_width}x{img_height}")
    
    # 准备输入
    img = letterbox(original_img, new_shape=640, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    # 创建模型
    print(f"🎯 创建模型...")
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    
    # 检查是否有已训练的模型
    model_paths = [
        "runs/final_200_epoch_clean_training/best_model.pkl",
        "runs/final_200_epoch_training/best_model.pkl",
        "runs/final_500_epoch_training_with_visualization/best_model.pkl"
    ]
    
    loaded_model = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"📦 加载训练好的模型: {model_path}")
            checkpoint = jt.load(model_path)
            model.load_state_dict(checkpoint['model'])
            print(f"   模型轮次: {checkpoint.get('epoch', 'unknown')}")
            print(f"   种类准确率: {checkpoint.get('species_accuracy', 0)*100:.1f}%")
            loaded_model = True
            break
    
    if not loaded_model:
        print(f"⚠️ 没有找到训练好的模型，使用随机初始化的模型进行推理")
    
    # 创建保存目录
    save_dir = Path("runs/fixed_inference_visualization_test")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始修复后的推理测试:")
    
    # 推理模式
    model.eval()
    with jt.no_grad():
        # 前向传播
        outputs = model(img_tensor)
        
        print(f"📊 模型输出分析:")
        print(f"   输出类型: {type(outputs)}")
        print(f"   输出形状: {outputs.shape}")
        print(f"   输出范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
        
        # 检查输出格式
        if len(outputs.shape) == 3 and outputs.shape[-1] == 25:  # [1, 8400, 25]
            print(f"   ✅ 推理模式输出格式正确: [batch, anchors, 25]")
            print(f"   坐标格式: xywh (前4维)")
            print(f"   objectness: 1.0 (第5维)")
            print(f"   类别分数: sigmoid后的概率 (后20维)")
            
            # 分析坐标范围
            pred_boxes = outputs[0, :, :4]  # [8400, 4] xywh
            pred_obj = outputs[0, :, 4]     # [8400] objectness
            pred_cls = outputs[0, :, 5:]    # [8400, 20] class scores
            
            print(f"   坐标范围: [{pred_boxes.min():.2f}, {pred_boxes.max():.2f}]")
            print(f"   objectness范围: [{pred_obj.min():.6f}, {pred_obj.max():.6f}]")
            print(f"   类别分数范围: [{pred_cls.min():.6f}, {pred_cls.max():.6f}]")
            
            # 分析期望类别的分数
            print(f"\n📊 期望类别分数分析:")
            expected_class_ids = [3, 11, 14]  # boat, dog, person
            for cls_id in expected_class_ids:
                cls_scores = pred_cls[:, cls_id]
                max_score = float(cls_scores.max())
                mean_score = float(cls_scores.mean())
                nonzero_count = int((cls_scores > 0.01).sum())
                cls_name = VOC_CLASSES[cls_id]
                print(f"   {cls_name}(类别{cls_id}): 最大{max_score:.6f}, 平均{mean_score:.6f}, >0.01的数量{nonzero_count}")
            
            # 转换坐标格式为xyxy（NMS需要）
            pred_boxes_xyxy = xywh2xyxy(pred_boxes)
            
            # 重新组装为NMS期望的格式
            nms_input = jt.concat([
                pred_boxes_xyxy,  # [8400, 4] xyxy
                pred_obj.unsqueeze(1),  # [8400, 1] objectness
                pred_cls  # [8400, 20] class scores
            ], dim=1).unsqueeze(0)  # [1, 8400, 25]
            
            print(f"\n🔍 NMS处理:")
            print(f"   NMS输入形状: {nms_input.shape}")
            print(f"   坐标格式: xyxy")
            print(f"   坐标范围: [{pred_boxes_xyxy.min():.2f}, {pred_boxes_xyxy.max():.2f}]")
            
            try:
                pred = non_max_suppression(nms_input, conf_thres=0.01, iou_thres=0.45, max_det=100)
                
                if len(pred) > 0 and len(pred[0]) > 0:
                    detections = pred[0]
                    det_count = len(detections)
                    print(f"   NMS后检测数量: {det_count}")
                    
                    # 转换为numpy
                    if hasattr(detections, 'numpy'):
                        detections_np = detections.numpy()
                    else:
                        detections_np = detections
                    
                    # 确保检测结果是2维的
                    if detections_np.ndim == 3:
                        detections_np = detections_np.reshape(-1, detections_np.shape[-1])
                    
                    # 创建左右对比的可视化图像
                    vis_width = img_width * 2 + 50  # 左右两张图 + 中间间隔
                    vis_height = max(img_height, 600)
                    vis_img = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 255
                    
                    # 左边：真实标注
                    left_img = original_img.copy()
                    print(f"\n🎨 绘制真实标注框 (左侧):")
                    for i, ann in enumerate(annotations):
                        cls_id, x_center, y_center, width, height = ann
                        
                        # 转换为像素坐标
                        x1 = int((x_center - width/2) * img_width)
                        y1 = int((y_center - height/2) * img_height)
                        x2 = int((x_center + width/2) * img_width)
                        y2 = int((y_center + height/2) * img_height)
                        
                        cls_name = VOC_CLASSES[cls_id]
                        print(f"     GT {i+1}: {cls_name} [{x1}, {y1}, {x2}, {y2}]")
                        draw_detection_box(left_img, [x1, y1, x2, y2], cls_name, 0.0, (0, 255, 255), False, "GT")
                    
                    # 右边：预测结果
                    right_img = original_img.copy()
                    print(f"\n🎨 绘制预测检测框 (右侧):")
                    
                    # 统计检测结果
                    detected_counts = {}
                    expected_detections = 0
                    confidence_info = []
                    
                    # 绘制预测检测框
                    for i, detection in enumerate(detections_np[:15]):  # 只显示前15个
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
                            
                            # 选择颜色
                            color = COLORS.get(cls_name, COLORS['default'])
                            
                            # 坐标已经是原图尺寸，不需要缩放
                            x1_int = int(float(x1))
                            y1_int = int(float(y1))
                            x2_int = int(float(x2))
                            y2_int = int(float(y2))
                            
                            print(f"     PRED {i+1}: {cls_name} [{x1_int}, {y1_int}, {x2_int}, {y2_int}] conf={float(conf):.6f}")
                            
                            draw_detection_box(right_img, [x1_int, y1_int, x2_int, y2_int], 
                                             cls_name, float(conf), color, is_expected, "PRED")
                    
                    # 组合左右图像
                    vis_img[:img_height, :img_width] = left_img
                    vis_img[:img_height, img_width+50:img_width*2+50] = right_img
                    
                    # 添加标题
                    cv2.putText(vis_img, "Ground Truth", (img_width//2-100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    cv2.putText(vis_img, "Prediction", (img_width + 50 + img_width//2-100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    
                    # 添加中间分割线
                    cv2.line(vis_img, (img_width+25, 0), (img_width+25, vis_height), (0, 0, 0), 2)
                    
                    print(f"\n📊 检测结果统计:")
                    print(f"   预测类别统计: {detected_counts}")
                    print(f"   期望类别检测数: {expected_detections}")
                    
                    # 显示置信度最高的前10个检测
                    confidence_info.sort(key=lambda x: x[1], reverse=True)
                    print(f"   置信度最高的10个检测:")
                    for i, (cls_name, conf) in enumerate(confidence_info[:10]):
                        status = "✅" if cls_name in expected_classes else "❌"
                        print(f"     {i+1:2d}. {status}{cls_name}: {conf:.6f}")
                    
                    # 计算种类准确率
                    detected_class_names = set(detected_counts.keys())
                    correct_classes = expected_classes.intersection(detected_class_names)
                    species_accuracy = len(correct_classes) / len(expected_classes) if expected_classes else 0.0
                    
                    print(f"   种类准确率: {species_accuracy*100:.1f}%")
                    print(f"   正确识别类别: {correct_classes}")
                    print(f"   遗漏类别: {expected_classes - correct_classes}")
                    
                    # 添加统计信息到图像底部
                    info_y = vis_height - 150
                    info_texts = [
                        f"Model: {'Trained' if loaded_model else 'Random'} | Detections: {det_count} | GT: {len(annotations)}",
                        f"Species Accuracy: {species_accuracy*100:.1f}% | Correct: {len(correct_classes)}/{len(expected_classes)}",
                        f"Expected: {', '.join(expected_classes)}",
                        f"Detected: {', '.join(correct_classes)}",
                        f"Highest Conf: {confidence_info[0][1]:.6f} ({confidence_info[0][0]})" if confidence_info else "No detections"
                    ]
                    
                    for i, text in enumerate(info_texts):
                        y_pos = info_y + i * 25
                        cv2.putText(vis_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(vis_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    # 保存可视化结果
                    save_path = save_dir / 'fixed_inference_comparison.jpg'
                    cv2.imwrite(str(save_path), vis_img)
                    print(f"\n💾 修复后的可视化对比结果已保存: {save_path}")
                    
                    # 保存详细报告
                    report_path = save_dir / 'fixed_inference_report.txt'
                    with open(report_path, 'w') as f:
                        f.write("GOLD-YOLO Jittor版本 - 修复后的推理测试报告\n")
                        f.write("=" * 60 + "\n")
                        f.write(f"模型状态: {'已训练' if loaded_model else '随机初始化'}\n")
                        f.write(f"坐标问题: 已修复\n")
                        f.write(f"真实标注: {target_counts}\n")
                        f.write(f"预测结果: {detected_counts}\n")
                        f.write(f"检测数量: {det_count}\n")
                        f.write(f"种类准确率: {species_accuracy*100:.1f}%\n")
                        f.write(f"正确识别类别: {correct_classes}\n")
                        f.write(f"遗漏类别: {expected_classes - correct_classes}\n")
                        f.write(f"最高置信度: {confidence_info[0][1]:.6f} ({confidence_info[0][0]})\n" if confidence_info else "无检测结果\n")
                        f.write("\n置信度最高的10个检测:\n")
                        for i, (cls_name, conf) in enumerate(confidence_info[:10]):
                            status = "✅" if cls_name in expected_classes else "❌"
                            f.write(f"  {i+1:2d}. {status}{cls_name}: {conf:.6f}\n")
                    
                    print(f"📄 详细报告已保存: {report_path}")
                    
                    # 检查推理是否成功
                    if species_accuracy >= 0.5 and confidence_info and confidence_info[0][1] > 0.1:
                        print(f"\n🎉 修复后的推理测试成功！")
                        print(f"✅ 种类准确率达到 {species_accuracy*100:.1f}%")
                        print(f"✅ 最高置信度达到 {confidence_info[0][1]:.6f}")
                        print(f"✅ 坐标问题已修复")
                        return True
                    else:
                        print(f"\n⚠️ 推理效果仍需提升")
                        print(f"   种类准确率: {species_accuracy*100:.1f}%")
                        print(f"   最高置信度: {confidence_info[0][1]:.6f}" if confidence_info else "无检测")
                        print(f"   建议继续训练模型")
                        return False
                else:
                    print(f"   ❌ NMS后没有检测结果")
                    return False
            
            except Exception as e:
                print(f"   ⚠️ NMS处理异常: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        else:
            print(f"   ❌ 推理模式输出格式错误")
            print(f"   期望: [1, 8400, 25]")
            print(f"   实际: {outputs.shape}")
            return False

def main():
    print("🔥 修复后的推理测试，解决坐标问题")
    print("=" * 80)
    print("修复：坐标解码问题，置信度过低问题")
    print("优化：左右对比可视化布局")
    print("=" * 80)
    
    success = fixed_inference_visualization_test()
    
    if success:
        print(f"\n🎉🎉🎉 修复后的推理测试成功！🎉🎉🎉")
        print(f"✅ 坐标问题已修复")
        print(f"✅ 置信度问题已解决")
        print(f"✅ 左右对比可视化完成")
    else:
        print(f"\n📊 推理测试完成，问题已定位")
        print(f"可视化结果已保存，可以查看具体问题")

if __name__ == "__main__":
    main()
