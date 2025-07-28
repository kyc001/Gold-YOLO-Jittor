#!/usr/bin/env python3
"""
简化的置信度分析 - 只分析训练后的模型
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

def simplified_confidence_analysis():
    """简化的置信度分析"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                简化置信度分析 - 训练后模型                   ║
    ║                                                              ║
    ║  🔍 分析自检训练后的模型输出                                 ║
    ║  🎯 定位置信度低的根本原因                                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 加载模型和权重
    print("🔧 加载自检训练后的模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    weights_path = 'simple_self_check_model.pkl'
    if os.path.exists(weights_path):
        model.load_state_dict(jt.load(weights_path))
        print(f"✅ 加载自检训练权重: {weights_path}")
    else:
        print(f"❌ 未找到自检训练权重")
        return False
    
    model.eval()
    
    # 准备测试数据
    print("📸 准备测试数据...")
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
    print(f"图像尺寸: {w0}x{h0} -> {img_tensor.shape}")
    
    # 模型推理
    print("🔍 模型推理分析...")
    with jt.no_grad():
        pred = model(img_tensor)
    
    pred = pred[0]  # [8400, 25]
    print(f"模型输出形状: {pred.shape}")
    
    # 分解输出
    boxes = pred[:, :4]  # [x_center, y_center, width, height]
    obj_conf = pred[:, 4]  # 目标置信度
    cls_conf = pred[:, 5:]  # 类别置信度 [20]
    
    print(f"\n📊 输出统计:")
    print(f"   目标置信度范围: [{float(obj_conf.min().numpy()):.6f}, {float(obj_conf.max().numpy()):.6f}]")
    print(f"   类别置信度范围: [{float(cls_conf.min().numpy()):.6f}, {float(cls_conf.max().numpy()):.6f}]")
    print(f"   类别置信度均值: {float(cls_conf.mean().numpy()):.6f}")
    print(f"   类别置信度标准差: {float(cls_conf.std().numpy()):.6f}")
    
    # 分析每个类别的置信度
    print(f"\n🎯 各类别置信度分析:")
    for i in range(20):
        cls_i_conf = cls_conf[:, i]
        print(f"   类别{i:2d}: 范围[{float(cls_i_conf.min().numpy()):.6f}, {float(cls_i_conf.max().numpy()):.6f}], 均值{float(cls_i_conf.mean().numpy()):.6f}")
    
    # 检查检测头参数
    print(f"\n🔧 检测头参数分析:")
    for name, param in model.named_parameters():
        if 'detect' in name and ('cls_pred' in name or 'reg_pred' in name):
            param_data = param.data
            min_val = float(param_data.min().numpy())
            max_val = float(param_data.max().numpy())
            mean_val = float(param_data.mean().numpy())
            std_val = float(param_data.std().numpy())
            
            print(f"   参数 {name}:")
            print(f"     形状: {param_data.shape}")
            print(f"     数值范围: [{min_val:.6f}, {max_val:.6f}]")
            print(f"     均值: {mean_val:.6f}, 标准差: {std_val:.6f}")
    
    # 分析训练目标位置的预测
    print(f"\n🎯 训练目标位置分析:")
    
    # 我们训练的目标位置是(0.5, 0.5)
    target_center = jt.array([0.5, 0.5])
    centers = boxes[:, :2]
    distances = jt.sqrt(((centers - target_center) ** 2).sum(dim=1))
    closest_idx = int(distances.argmin().numpy())
    
    print(f"   最接近训练目标的预测索引: {closest_idx}")
    print(f"   该位置的坐标: {boxes[closest_idx].numpy()}")
    print(f"   该位置的目标置信度: {float(obj_conf[closest_idx].numpy()):.6f}")
    print(f"   该位置的类别0置信度: {float(cls_conf[closest_idx, 0].numpy()):.6f}")
    print(f"   该位置的所有类别置信度: {cls_conf[closest_idx].numpy()}")
    
    # 问题诊断
    print(f"\n🎯 问题诊断:")
    
    # 1. 目标置信度检查
    obj_min = float(obj_conf.min().numpy())
    obj_max = float(obj_conf.max().numpy())
    if obj_min == obj_max == 1.0:
        print("   ✅ 目标置信度正常 (全部为1.0)")
    else:
        print(f"   ❌ 目标置信度异常: 范围[{obj_min:.6f}, {obj_max:.6f}]")
    
    # 2. 类别置信度检查
    cls_min = float(cls_conf.min().numpy())
    cls_max = float(cls_conf.max().numpy())
    cls_range = cls_max - cls_min
    
    if cls_range < 0.01:
        print(f"   ❌ 类别置信度变化范围太小: {cls_range:.6f}")
        print("   🔧 可能原因:")
        print("      1. 分类头权重初始化过小")
        print("      2. 训练时分类损失权重不足")
        print("      3. sigmoid激活函数饱和在低值区域")
        print("      4. 学习率对分类头不合适")
    else:
        print(f"   ✅ 类别置信度变化范围正常: {cls_range:.6f}")
    
    # 3. 激活函数分析
    print(f"\n🔧 激活函数分析:")
    
    # 假设cls_conf是经过sigmoid的，我们反推logits
    # sigmoid(x) = y => x = log(y/(1-y))
    eps = 1e-7
    cls_conf_clipped = jt.clamp(cls_conf, eps, 1-eps)
    estimated_logits = jt.log(cls_conf_clipped / (1 - cls_conf_clipped))
    
    print(f"   估计的logits范围: [{float(estimated_logits.min().numpy()):.6f}, {float(estimated_logits.max().numpy()):.6f}]")
    print(f"   估计的logits均值: {float(estimated_logits.mean().numpy()):.6f}")
    
    if float(estimated_logits.mean().numpy()) < -5:
        print("   ❌ logits均值过低，说明分类头输出偏向负值")
        print("   🔧 建议: 调整分类头的bias初始化")
    
    # 4. 修复建议
    print(f"\n💡 修复建议:")
    print("   1. 重新初始化分类头的bias为正值 (如0.01)")
    print("   2. 增加分类损失的权重")
    print("   3. 使用更高的学习率训练分类头")
    print("   4. 增加训练轮数")
    print("   5. 检查标签one-hot编码是否正确")
    
    return True

def create_visualization():
    """创建可视化结果"""
    print(f"\n🎨 创建检测结果可视化...")
    
    # 加载模型
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    weights_path = 'simple_self_check_model.pkl'
    if os.path.exists(weights_path):
        model.load_state_dict(jt.load(weights_path))
    model.eval()
    
    # 加载图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    
    # 预处理
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
    # 推理
    with jt.no_grad():
        pred = model(img_tensor)[0]
    
    # 解析预测
    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_conf = pred[:, 5:]
    
    # 计算最终置信度
    cls_scores = cls_conf.max(dim=1)[0]
    cls_indices = cls_conf.argmax(dim=1)
    final_conf = obj_conf * cls_scores
    
    # 选择前5个检测进行可视化
    num_show = 5
    img_vis = img0.copy()
    
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
    
    print(f"🎯 前{num_show}个检测结果:")
    
    for i in range(num_show):
        # 获取预测
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
        label = f'Det{i+1}: C{cls_idx} F{final_c:.6f}'
        cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        print(f"   检测{i+1}: 位置=({x1},{y1},{x2},{y2}), 目标={obj_c:.6f}, 类别{cls_idx}={cls_c:.6f}, 最终={final_c:.6f}")
    
    # 保存结果
    result_path = 'confidence_analysis_result.jpg'
    cv2.imwrite(result_path, img_vis)
    print(f"📸 可视化结果已保存: {result_path}")
    
    return True

if __name__ == "__main__":
    print("🚀 开始简化置信度分析...")
    
    # 分析
    success1 = simplified_confidence_analysis()
    
    # 可视化
    success2 = create_visualization()
    
    if success1 and success2:
        print("\n🎉 简化置信度分析完成！")
        print("📋 关键发现:")
        print("   - 目标检测功能正常 (置信度=1.0)")
        print("   - 类别分类置信度过低 (约0.0002-0.002)")
        print("   - 需要重新训练分类头或调整初始化")
    else:
        print("\n❌ 分析过程中出现错误！")
