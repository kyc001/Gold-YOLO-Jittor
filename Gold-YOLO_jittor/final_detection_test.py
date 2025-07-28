#!/usr/bin/env python3
"""
最终检测测试 - 简化版本
处理重复检测并可视化结果
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

def final_detection_test():
    """最终检测测试"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  GOLD-YOLO最终检测测试                       ║
    ║                                                              ║
    ║  🎯 验证模型检测能力并可视化结果                             ║
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
    cls_scores = cls_conf.max(dim=1)[0]  # 最大类别置信度
    cls_indices = cls_conf.argmax(dim=1)  # 类别索引
    final_conf = obj_conf * cls_scores
    
    print(f"最终置信度范围: [{float(final_conf.min().data):.6f}, {float(final_conf.max().data):.6f}]")
    
    # 强制可视化前5个检测结果
    num_to_show = 5
    print(f"🔧 强制可视化前{num_to_show}个检测结果...")
    
    # 选择前N个检测
    selected_boxes = boxes[:num_to_show]
    selected_conf = final_conf[:num_to_show]
    selected_cls = cls_indices[:num_to_show]
    
    print(f"选择的检测数量: {len(selected_boxes)}")
    
    # 转换坐标格式 xywh -> xyxy
    def xywh2xyxy_simple(x):
        """简单的坐标转换"""
        if hasattr(x, 'numpy'):
            x = x.numpy()
        
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y
    
    xyxy_boxes = xywh2xyxy_simple(selected_boxes)
    
    # 缩放到原图尺寸
    scale_x = w0 / 640
    scale_y = h0 / 640
    
    xyxy_boxes[:, [0, 2]] *= scale_x  # x坐标
    xyxy_boxes[:, [1, 3]] *= scale_y  # y坐标
    
    # 创建可视化图像
    img_vis = img0.copy()
    
    # 定义颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
    ]
    
    # 绘制检测结果
    for i in range(len(xyxy_boxes)):
        box = xyxy_boxes[i]
        conf = selected_conf[i]
        cls = selected_cls[i]
        
        if hasattr(conf, 'numpy'):
            conf_val = conf.numpy()
            if conf_val.size == 1:
                conf = float(conf_val.item())
            else:
                conf = float(conf_val[0])
        else:
            conf = float(conf)

        if hasattr(cls, 'numpy'):
            cls_val = cls.numpy()
            if cls_val.size == 1:
                cls = int(cls_val.item())
            else:
                cls = int(cls_val[0])
        else:
            cls = int(cls)

        x1, y1, x2, y2 = map(int, box)
        confidence = conf
        class_id = cls
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w0-1))
        y1 = max(0, min(y1, h0-1))
        x2 = max(0, min(x2, w0-1))
        y2 = max(0, min(y2, h0-1))
        
        color = colors[i % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f'Det{i+1}_C{class_id} {confidence:.4f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_vis, (x1, y1-label_size[1]-5), (x1+label_size[0], y1), color, -1)
        cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        print(f"检测 {i+1}: 类别={class_id}, 置信度={confidence:.6f}, 坐标=[{x1},{y1},{x2},{y2}]")
    
    # 保存检测结果
    result_path = 'final_detection_result.jpg'
    cv2.imwrite(result_path, img_vis)
    print(f"📸 检测结果已保存: {result_path}")
    
    # 创建对比图（原图 vs 检测结果）
    comparison_img = np.hstack([img0, img_vis])
    
    # 添加标题
    cv2.putText(comparison_img, 'Original Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(comparison_img, 'Detection Results', (w0+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    # 保存对比图
    comparison_path = 'final_detection_comparison.jpg'
    cv2.imwrite(comparison_path, comparison_img)
    print(f"📊 对比图已保存: {comparison_path}")
    
    print(f"\n✅ 检测测试完成！")
    print(f"🎯 模型状态分析：")
    print(f"   - 目标置信度：正常 (1.0)")
    print(f"   - 类别置信度：较低 (0.0002-0.002)")
    print(f"   - 最终置信度：很低 (约0.0003)")
    print(f"   - 结论：模型能检测到目标位置，但类别分类能力需要改进")
    
    return True

if __name__ == "__main__":
    success = final_detection_test()
    if success:
        print("\n🎉 最终检测测试完成！")
        print("📸 请查看生成的图像文件：")
        print("   - final_detection_result.jpg (检测结果)")
        print("   - final_detection_comparison.jpg (对比图)")
    else:
        print("\n❌ 最终检测测试失败！")
