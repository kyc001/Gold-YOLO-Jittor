#!/usr/bin/env python3
"""
调试Gold-YOLO后处理函数
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np

# 设置Jittor
jt.flags.use_cuda = 1

def debug_postprocess():
    """调试后处理函数"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        
        # 加载配置和模型
        config = get_config()
        model = Model(config=config, channels=3, num_classes=80)
        model.eval()
        
        # 创建测试输入
        img_tensor = jt.randn(1, 3, 640, 640)
        
        print("🔍 调试Gold-YOLO后处理函数")
        print("=" * 60)
        
        # 获取模型输出
        with jt.no_grad():
            outputs = model(img_tensor)
        
        print(f"模型输出形状: {outputs.shape}")
        print(f"模型输出类型: {type(outputs)}")
        
        # 分析输出的各个部分
        batch_size, num_anchors, features = outputs.shape
        print(f"批次大小: {batch_size}")
        print(f"锚点数量: {num_anchors}")
        print(f"特征维度: {features}")
        
        # 提取各部分
        boxes = outputs[0, :, :4]  # [num_anchors, 4]
        conf = outputs[0, :, 4]    # [num_anchors]
        classes = outputs[0, :, 5:] # [num_anchors, 80]
        
        print(f"\n📦 边界框分析:")
        print(f"  边界框形状: {boxes.shape}")
        print(f"  X坐标范围: [{boxes[:, 0].min().item():.2f}, {boxes[:, 0].max().item():.2f}]")
        print(f"  Y坐标范围: [{boxes[:, 1].min().item():.2f}, {boxes[:, 1].max().item():.2f}]")
        
        print(f"\n🎯 置信度分析:")
        print(f"  置信度形状: {conf.shape}")
        print(f"  置信度范围: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
        print(f"  置信度均值: {conf.mean().item():.6f}")
        
        # 测试不同的置信度阈值
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for thresh in thresholds:
            count = (conf > thresh).sum().item()
            print(f"  置信度>{thresh}: {count}个")
        
        print(f"\n🏷️ 类别分析:")
        print(f"  类别分数形状: {classes.shape}")
        print(f"  类别分数范围: [{classes.min().item():.6f}, {classes.max().item():.6f}]")
        
        # 现在测试我们的后处理函数
        print(f"\n🔧 测试后处理函数:")
        
        # 导入后处理函数
        sys.path.append('/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/scripts')
        from gold_yolo_sanity_check import strict_post_process
        
        # 测试不同的阈值
        test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        for thresh in test_thresholds:
            detections = strict_post_process(outputs, conf_thres=thresh, iou_thres=0.5, max_det=20)
            det_count = detections[0].shape[0] if len(detections) > 0 else 0
            print(f"  阈值{thresh}: 检测到{det_count}个目标")
            
            if det_count > 0:
                det = detections[0]
                print(f"    检测置信度范围: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
                print(f"    检测类别: {set(det[:, 5].numpy().astype(int))}")
        
        # 手动实现一个简单的后处理来对比
        print(f"\n🛠️ 手动后处理测试:")
        
        # 转换为numpy
        outputs_np = outputs.numpy()
        pred = outputs_np[0]  # [num_anchors, 85]
        
        # 提取各部分
        boxes_np = pred[:, :4]
        conf_np = pred[:, 4]
        classes_np = pred[:, 5:]
        
        # 简单过滤
        for thresh in [0.1, 0.3, 0.5]:
            mask = conf_np > thresh
            filtered_count = np.sum(mask)
            print(f"  阈值{thresh}: numpy过滤后{filtered_count}个")
            
            if filtered_count > 0:
                filtered_boxes = boxes_np[mask]
                filtered_conf = conf_np[mask]
                filtered_classes = classes_np[mask]
                
                # 检查边界框有效性
                valid_boxes = 0
                for i, box in enumerate(filtered_boxes):
                    x1, y1, x2, y2 = box
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        if area > 100:  # 最小面积阈值
                            valid_boxes += 1
                
                print(f"    有效边界框: {valid_boxes}个")
        
        print("\n" + "=" * 60)
        print("✅ 后处理调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_postprocess()
