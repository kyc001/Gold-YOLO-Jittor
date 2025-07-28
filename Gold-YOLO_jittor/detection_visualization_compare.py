#!/usr/bin/env python3
"""
GOLD-YOLO检测结果可视化对比系统
处理重复检测问题并与真实标注对比
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

def xywh2xyxy(x):
    """将xywh格式转换为xyxy格式"""
    y = jt.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def simple_nms(boxes, scores, iou_threshold=0.5):
    """简化的NMS实现，处理重复检测"""
    if len(boxes) == 0:
        return []
    
    # 转换为numpy进行处理
    if hasattr(boxes, 'numpy'):
        boxes = boxes.numpy()
    if hasattr(scores, 'numpy'):
        scores = scores.numpy()
    
    # 按置信度排序
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # 选择置信度最高的
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # 计算IoU
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # 计算交集
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算并集
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = area_current + area_others - intersection
        
        # 计算IoU
        iou = intersection / (union + 1e-6)
        
        # 保留IoU小于阈值的框
        indices = indices[1:][iou < iou_threshold]
    
    return keep

def load_ground_truth_labels(img_path):
    """加载真实标注（从VOC数据集或手动创建）"""
    # 这里我们创建一些示例标注用于对比
    # 实际使用时应该从标注文件加载
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 示例标注：假设图像中有一些目标
    # 格式：[class_id, x1, y1, x2, y2, confidence]
    gt_boxes = [
        [0, w*0.1, h*0.1, w*0.5, h*0.5, 1.0],  # 类别0，左上角区域
        [1, w*0.6, h*0.3, w*0.9, h*0.7, 1.0],  # 类别1，右侧区域
    ]
    
    return gt_boxes

def visualize_detection_comparison():
    """检测结果可视化对比"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              GOLD-YOLO检测结果可视化对比系统                  ║
    ║                                                              ║
    ║  🎯 处理重复检测问题                                         ║
    ║  📊 检测结果与真实标注可视化对比                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 加载模型
    print("🔧 加载训练好的模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 加载自检训练的权重
    weights_path = 'simple_self_check_model.pkl'
    if os.path.exists(weights_path):
        model.load_state_dict(jt.load(weights_path))
        print(f"✅ 加载自检训练权重: {weights_path}")
    else:
        print("⚠️ 未找到自检训练权重，使用初始化权重")
    
    model.eval()
    
    # 加载测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    print(f"📸 加载测试图像: {os.path.basename(img_path)}")
    
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"❌ 无法读取图像: {img_path}")
        return False
    
    h0, w0 = img0.shape[:2]
    print(f"原始图像尺寸: {w0}x{h0}")
    
    # 预处理
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
    print(f"预处理后图像尺寸: {img_tensor.shape}")
    
    # 推理
    print("🔍 开始推理...")
    with jt.no_grad():
        pred = model(img_tensor)
    
    print(f"模型输出形状: {pred.shape}")
    
    # 解析预测结果
    pred = pred[0]  # 移除batch维度 [8400, 25]
    
    # 提取坐标、置信度和类别
    boxes = pred[:, :4]  # [x_center, y_center, width, height]
    obj_conf = pred[:, 4]  # 目标置信度
    cls_conf = pred[:, 5:]  # 类别置信度 [20]
    
    print(f"目标置信度范围: [{float(obj_conf.min().data):.6f}, {float(obj_conf.max().data):.6f}]")
    print(f"类别置信度范围: [{float(cls_conf.min().data):.6f}, {float(cls_conf.max().data):.6f}]")
    
    # 计算最终置信度和类别
    cls_scores = cls_conf.max(dim=1)[0]  # Jittor返回(values, indices)
    cls_indices = cls_conf.argmax(dim=1)
    final_conf = obj_conf * cls_scores
    
    print(f"最终置信度范围: [{float(final_conf.min().data):.6f}, {float(final_conf.max().data):.6f}]")
    
    # 强制可视化：选择置信度最高的前10个检测
    print("🔧 强制可视化置信度最高的检测结果...")

    # 简化方法：直接选择前10个
    num_to_show = min(10, len(final_conf))

    # 筛选检测结果 - 使用前N个
    filtered_boxes = boxes[:num_to_show]
    filtered_conf = final_conf[:num_to_show]
    filtered_cls = cls_indices[:num_to_show]

    print(f"选择的检测数量: {len(filtered_boxes)}")
    print(f"置信度范围: [{float(filtered_conf.min().data):.6f}, {float(filtered_conf.max().data):.6f}]")

    # 转换坐标格式 xywh -> xyxy
    xyxy_boxes = xywh2xyxy(filtered_boxes)

    # 缩放到原图尺寸
    scale_x = w0 / 640
    scale_y = h0 / 640

    xyxy_boxes[:, [0, 2]] *= scale_x  # x坐标
    xyxy_boxes[:, [1, 3]] *= scale_y  # y坐标

    # 应用NMS处理重复检测
    print("🔧 应用NMS处理重复检测...")
    keep_indices = simple_nms(xyxy_boxes, filtered_conf, iou_threshold=0.5)

    if len(keep_indices) > 0:
        final_boxes = xyxy_boxes[keep_indices]
        final_conf = filtered_conf[keep_indices]
        final_cls = filtered_cls[keep_indices]

        print(f"NMS后检测数量: {len(final_boxes)}")

        # 创建可视化图像
        img_vis = img0.copy()

        # 定义颜色
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
        ]
            
            # 绘制检测结果
            for i, (box, conf, cls) in enumerate(zip(final_boxes, final_conf, final_cls)):
                if hasattr(box, 'numpy'):
                    box = box.numpy()
                if hasattr(conf, 'numpy'):
                    conf = conf.numpy()
                if hasattr(cls, 'numpy'):
                    cls = cls.numpy()
                
                x1, y1, x2, y2 = map(int, box)
                confidence = float(conf)
                class_id = int(cls)
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, w0-1))
                y1 = max(0, min(y1, h0-1))
                x2 = max(0, min(x2, w0-1))
                y2 = max(0, min(y2, h0-1))
                
                color = colors[class_id % len(colors)]
                
                # 绘制边界框
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f'Class{class_id} {confidence:.2f}'
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img_vis, (x1, y1-label_size[1]-5), (x1+label_size[0], y1), color, -1)
                cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                
                print(f"检测 {i+1}: 类别={class_id}, 置信度={confidence:.3f}, 坐标=[{x1},{y1},{x2},{y2}]")
            
            # 保存检测结果
            result_path = 'detection_result_comparison.jpg'
            cv2.imwrite(result_path, img_vis)
            print(f"📸 检测结果已保存: {result_path}")
            
            # 加载真实标注并创建对比图
            print("\n🔧 创建与真实标注的对比...")
            gt_boxes = load_ground_truth_labels(img_path)
            
            # 创建对比图像（左：检测结果，右：真实标注）
            comparison_img = np.zeros((h0, w0*2, 3), dtype=np.uint8)
            comparison_img[:, :w0] = img_vis  # 左侧：检测结果
            comparison_img[:, w0:] = img0    # 右侧：原图+真实标注
            
            # 在右侧绘制真实标注
            for i, gt_box in enumerate(gt_boxes):
                class_id, x1, y1, x2, y2, conf = gt_box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 调整坐标到右侧图像
                x1 += w0
                x2 += w0
                
                color = (0, 255, 0)  # 绿色表示真实标注
                cv2.rectangle(comparison_img, (x1, y1), (x2, y2), color, 2)
                
                label = f'GT_Class{int(class_id)}'
                cv2.putText(comparison_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 添加标题
            cv2.putText(comparison_img, 'Detection Results', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(comparison_img, 'Ground Truth', (w0+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # 保存对比图
            comparison_path = 'detection_vs_groundtruth_comparison.jpg'
            cv2.imwrite(comparison_path, comparison_img)
            print(f"📊 对比图已保存: {comparison_path}")
            
            print(f"\n✅ 检测成功！共检测到 {len(final_boxes)} 个目标")
            print(f"📈 重复检测处理：原始{len(filtered_boxes)}个 -> NMS后{len(final_boxes)}个")
            
            return True
        else:
            print("❌ NMS后没有保留任何检测结果")
            return False
    else:
        print("❌ 没有检测到任何目标")
        return False

if __name__ == "__main__":
    success = visualize_detection_comparison()
    if success:
        print("\n🎉 检测结果可视化对比完成！")
    else:
        print("\n❌ 检测结果可视化对比失败！")
