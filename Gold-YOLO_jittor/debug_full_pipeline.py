#!/usr/bin/env python3
"""
调试完整的推理流程
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np
from PIL import Image

# 设置Jittor
jt.flags.use_cuda = 1

def debug_full_pipeline():
    """调试完整的推理流程"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        
        # 导入自检脚本的函数
        sys.path.append('/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/scripts')
        from gold_yolo_sanity_check import strict_post_process, scale_coords
        
        print("🔍 调试完整推理流程")
        print("=" * 60)
        
        # 加载配置和模型
        config = get_config()
        model = Model(config=config, channels=3, num_classes=80)
        model.eval()
        
        # 创建测试输入
        img_tensor = jt.randn(1, 3, 640, 640)
        
        print("🚀 步骤1: 模型推理")
        with jt.no_grad():
            outputs = model(img_tensor)
        
        print(f"  模型输出形状: {outputs.shape}")
        print(f"  置信度范围: [{outputs[0, :, 4].min().item():.3f}, {outputs[0, :, 4].max().item():.3f}]")
        
        print("\n🔧 步骤2: 后处理")
        detections = strict_post_process(outputs, conf_thres=0.3, iou_thres=0.5, max_det=20)
        det = detections[0]  # 第一个batch
        
        print(f"  后处理检测数量: {det.shape[0]}")
        
        if det.shape[0] > 0:
            print(f"  检测置信度范围: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
            print(f"  检测类别: {set(det[:, 5].numpy().astype(int))}")
            print(f"  检测框范围:")
            print(f"    X: [{det[:, 0].min().item():.1f}, {det[:, 2].max().item():.1f}]")
            print(f"    Y: [{det[:, 1].min().item():.1f}, {det[:, 3].max().item():.1f}]")
        
        print("\n📏 步骤3: 坐标缩放")
        
        # 模拟原始图像尺寸
        original_height, original_width = 424, 640
        
        if det.shape[0] > 0:
            print(f"  缩放前检测框数量: {det.shape[0]}")
            print(f"  原始图像尺寸: {original_height}x{original_width}")
            
            # 复制检测结果以避免原地修改
            det_before_scale = det.clone()
            
            # 坐标缩放
            det = scale_coords((640, 640), det, (original_height, original_width))
            
            print(f"  缩放后检测框数量: {det.shape[0]}")
            
            if det.shape[0] > 0:
                print(f"  缩放后置信度范围: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
                print(f"  缩放后检测框范围:")
                print(f"    X: [{det[:, 0].min().item():.1f}, {det[:, 2].max().item():.1f}]")
                print(f"    Y: [{det[:, 1].min().item():.1f}, {det[:, 3].max().item():.1f}]")
                
                # 检查有效检测框
                valid_count = 0
                for i in range(det.shape[0]):
                    x1, y1, x2, y2, conf, cls = det[i].numpy()
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        if area > 0:  # 任何正面积都算有效
                            valid_count += 1
                            print(f"    检测框{i+1}: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] 置信度={conf:.3f} 类别={int(cls)} - 有效")
                        else:
                            print(f"    检测框{i+1}: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] 置信度={conf:.3f} 类别={int(cls)} - 面积为0")
                    else:
                        print(f"    检测框{i+1}: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] 置信度={conf:.3f} 类别={int(cls)} - 无效坐标")
                
                print(f"  最终有效检测框数量: {valid_count}")
                
                # 如果没有有效检测框，让我们看看是什么问题
                if valid_count == 0:
                    print("\n🔍 深入分析无效检测框:")
                    print("  缩放前后对比:")
                    for i in range(min(5, det.shape[0])):  # 只看前5个
                        before = det_before_scale[i, :4].numpy()
                        after = det[i, :4].numpy()
                        print(f"    检测框{i+1}:")
                        print(f"      缩放前: [{before[0]:.1f},{before[1]:.1f},{before[2]:.1f},{before[3]:.1f}]")
                        print(f"      缩放后: [{after[0]:.1f},{after[1]:.1f},{after[2]:.1f},{after[3]:.1f}]")
            else:
                print("  ❌ 缩放后没有检测框！")
        else:
            print("  ❌ 后处理后没有检测框！")
        
        print("\n📊 步骤4: 最终统计")
        final_count = det.shape[0] if det.shape[0] > 0 else 0
        print(f"  最终检测框数量: {final_count}")
        
        if final_count == 0:
            print("\n🚨 问题诊断:")
            print("  1. 模型推理: ✅ 正常")
            print("  2. 后处理: ✅ 正常 (检测到目标)")
            print("  3. 坐标缩放: ❌ 可能有问题")
            print("  建议: 检查scale_coords函数的实现")
        else:
            print("  ✅ 完整流程正常工作!")
        
        print("\n" + "=" * 60)
        print("✅ 完整流程调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_pipeline()
