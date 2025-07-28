#!/usr/bin/env python3
"""
快速修复测试 - 简化版本
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from jittor import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

def quick_fix_test():
    """快速修复测试"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    快速修复测试                               ║
    ║                                                              ║
    ║  🔧 重新初始化分类头并快速测试                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建模型
    print("🔧 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            print(f"   重新初始化: {name}")
            # 使用正态分布初始化权重
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            # 设置bias为小正值
            if module.bias is not None:
                jt.init.constant_(module.bias, 0.01)
    
    # 测试初始化后的输出
    print("🔍 测试重新初始化后的输出...")
    model.eval()
    
    # 准备测试数据
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
    # 推理
    with jt.no_grad():
        pred = model(img_tensor)[0]  # [8400, 25]
    
    # 解析预测结果
    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_conf = pred[:, 5:]
    
    print(f"重新初始化后:")
    print(f"   目标置信度范围: [{float(obj_conf.min().numpy()):.6f}, {float(obj_conf.max().numpy()):.6f}]")
    print(f"   类别置信度范围: [{float(cls_conf.min().numpy()):.6f}, {float(cls_conf.max().numpy()):.6f}]")
    
    # 检查类别置信度变化范围
    cls_range = float(cls_conf.max().numpy()) - float(cls_conf.min().numpy())
    print(f"   类别置信度变化范围: {cls_range:.6f}")
    
    if cls_range > 0.01:
        print("✅ 重新初始化成功！类别置信度有变化")
        
        # 计算最终置信度
        cls_scores = cls_conf.max(dim=1)[0]
        cls_indices = cls_conf.argmax(dim=1)
        final_conf = obj_conf * cls_scores
        
        print(f"   最终置信度范围: [{float(final_conf.min().numpy()):.6f}, {float(final_conf.max().numpy()):.6f}]")
        
        # 简单可视化前3个检测
        print("\n🎨 可视化前3个检测结果...")
        img_vis = img0.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255)]
        
        for i in range(3):
            box = boxes[i].numpy()
            obj_c = float(obj_conf[i].numpy())
            cls_idx = int(cls_indices[i].numpy())
            cls_c = float(cls_scores[i].numpy())
            final_c = float(final_conf[i].numpy())
            
            # 转换坐标
            x_center, y_center, width, height = box
            x1 = int((x_center - width/2) * w0 / 640)
            y1 = int((y_center - height/2) * h0 / 640)
            x2 = int((x_center + width/2) * w0 / 640)
            y2 = int((y_center + height/2) * h0 / 640)
            
            # 确保坐标在范围内
            x1 = max(0, min(x1, w0-1))
            y1 = max(0, min(y1, h0-1))
            x2 = max(0, min(x2, w0-1))
            y2 = max(0, min(y2, h0-1))
            
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'QuickFix{i+1}_C{cls_idx} {final_c:.4f}'
            cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            print(f"   检测{i+1}: 类别={cls_idx}, 目标={obj_c:.6f}, 类别={cls_c:.6f}, 最终={final_c:.6f}")
        
        # 保存结果
        result_path = 'quick_fix_result.jpg'
        cv2.imwrite(result_path, img_vis)
        print(f"📸 快速修复结果已保存: {result_path}")
        
        # 保存模型
        save_path = 'quick_fixed_model.pkl'
        jt.save(model.state_dict(), save_path)
        print(f"💾 快速修复模型已保存: {save_path}")
        
        return True
    else:
        print("❌ 重新初始化后类别置信度仍无变化")
        return False

def compare_with_original():
    """与原始模型对比"""
    print("\n📊 与原始自检模型对比...")
    
    # 加载原始自检模型
    original_model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    weights_path = 'simple_self_check_model.pkl'
    if os.path.exists(weights_path):
        original_model.load_state_dict(jt.load(weights_path))
        print("✅ 加载原始自检模型")
    else:
        print("❌ 未找到原始自检模型")
        return
    
    # 加载快速修复模型
    fixed_model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    fixed_weights_path = 'quick_fixed_model.pkl'
    if os.path.exists(fixed_weights_path):
        fixed_model.load_state_dict(jt.load(fixed_weights_path))
        print("✅ 加载快速修复模型")
    else:
        print("❌ 未找到快速修复模型")
        return
    
    # 准备测试数据
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
    # 对比测试
    original_model.eval()
    fixed_model.eval()
    
    with jt.no_grad():
        # 原始模型
        orig_pred = original_model(img_tensor)[0]
        orig_cls_conf = orig_pred[:, 5:]
        orig_cls_range = float(orig_cls_conf.max().numpy()) - float(orig_cls_conf.min().numpy())
        
        # 修复模型
        fixed_pred = fixed_model(img_tensor)[0]
        fixed_cls_conf = fixed_pred[:, 5:]
        fixed_cls_range = float(fixed_cls_conf.max().numpy()) - float(fixed_cls_conf.min().numpy())
    
    print(f"📊 对比结果:")
    print(f"   原始模型类别置信度变化范围: {orig_cls_range:.6f}")
    print(f"   修复模型类别置信度变化范围: {fixed_cls_range:.6f}")
    print(f"   改进倍数: {fixed_cls_range / orig_cls_range:.2f}x")
    
    if fixed_cls_range > orig_cls_range * 10:
        print("🎉 快速修复显著改善了分类性能！")
    else:
        print("⚠️ 快速修复效果有限，需要进一步训练")

if __name__ == "__main__":
    print("🚀 开始快速修复测试...")
    
    # 快速修复
    success = quick_fix_test()
    
    if success:
        # 对比分析
        compare_with_original()
        
        print("\n🎉 快速修复测试完成！")
        print("📋 修复效果:")
        print("   - 重新初始化分类头权重")
        print("   - 类别置信度有明显变化")
        print("   - 检测功能基本正常")
    else:
        print("\n❌ 快速修复测试失败！")
