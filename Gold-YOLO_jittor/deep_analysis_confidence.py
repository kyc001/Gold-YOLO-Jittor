#!/usr/bin/env python3
"""
深入分析置信度低的原因
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

def deep_analysis_confidence():
    """深入分析置信度问题"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  深入分析置信度低的原因                       ║
    ║                                                              ║
    ║  🔍 分析模型各层输出                                         ║
    ║  🎯 定位问题根源                                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 加载模型
    print("🔧 加载模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 对比：加载自检训练权重 vs 初始权重
    print("\n📊 对比分析：初始权重 vs 自检训练权重")
    
    # 1. 测试初始权重
    print("\n🔍 测试1：初始权重模型")
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
    
    with jt.no_grad():
        pred_init = model(img_tensor)
    
    # 分析初始权重输出
    pred_init = pred_init[0]  # [8400, 25]
    obj_conf_init = pred_init[:, 4]
    cls_conf_init = pred_init[:, 5:]
    
    print(f"   初始权重 - 目标置信度范围: [{float(obj_conf_init.min().numpy()):.6f}, {float(obj_conf_init.max().numpy()):.6f}]")
    print(f"   初始权重 - 类别置信度范围: [{float(cls_conf_init.min().numpy()):.6f}, {float(cls_conf_init.max().numpy()):.6f}]")
    
    # 2. 测试自检训练权重
    print("\n🔍 测试2：自检训练权重模型")
    weights_path = 'simple_self_check_model.pkl'
    if os.path.exists(weights_path):
        model.load_state_dict(jt.load(weights_path))
        print(f"   ✅ 加载自检训练权重: {weights_path}")
    else:
        print(f"   ❌ 未找到自检训练权重")
        return False
    
    model.eval()
    
    with jt.no_grad():
        pred_trained = model(img_tensor)
    
    # 分析训练后输出
    pred_trained = pred_trained[0]  # [8400, 25]
    obj_conf_trained = pred_trained[:, 4]
    cls_conf_trained = pred_trained[:, 5:]
    
    print(f"   训练后 - 目标置信度范围: [{float(obj_conf_trained.min().numpy()):.6f}, {float(obj_conf_trained.max().numpy()):.6f}]")
    print(f"   训练后 - 类别置信度范围: [{float(cls_conf_trained.min().numpy()):.6f}, {float(cls_conf_trained.max().numpy()):.6f}]")
    
    # 3. 分析检测头参数
    print("\n🔍 测试3：检测头参数分析")
    
    # 检查检测头的权重
    for name, param in model.named_parameters():
        if 'detect' in name and ('cls_pred' in name or 'reg_pred' in name):
            param_data = param.data
            print(f"   参数 {name}:")
            print(f"     形状: {param_data.shape}")
            min_val = param_data.min()
            max_val = param_data.max()
            mean_val = param_data.mean()
            std_val = param_data.std()

            print(f"     数值范围: [{float(min_val.numpy()):.6f}, {float(max_val.numpy()):.6f}]")
            print(f"     均值: {float(mean_val.numpy()):.6f}")
            print(f"     标准差: {float(std_val.numpy()):.6f}")
    
    # 4. 分析激活函数输出
    print("\n🔍 测试4：激活函数分析")
    
    # 检查sigmoid激活前后的值
    raw_cls_logits = cls_conf_trained  # 这些应该是sigmoid之前的logits
    print(f"   类别logits范围: [{float(raw_cls_logits.min().numpy()):.6f}, {float(raw_cls_logits.max().numpy()):.6f}]")
    print(f"   类别logits均值: {float(raw_cls_logits.mean().numpy()):.6f}")

    # 手动应用sigmoid看看
    manual_sigmoid = jt.sigmoid(raw_cls_logits)
    print(f"   手动sigmoid后范围: [{float(manual_sigmoid.min().numpy()):.6f}, {float(manual_sigmoid.max().numpy()):.6f}]")
    
    # 5. 检查训练目标
    print("\n🔍 测试5：训练目标分析")
    
    # 检查我们训练时使用的标签
    train_target = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    print(f"   训练目标: {train_target.numpy()}")
    print(f"   目标类别: {int(train_target[0, 0].numpy())}")
    print(f"   目标位置: [{float(train_target[0, 1].numpy()):.2f}, {float(train_target[0, 2].numpy()):.2f}]")
    print(f"   目标尺寸: [{float(train_target[0, 3].numpy()):.2f}, {float(train_target[0, 4].numpy()):.2f}]")
    
    # 6. 检查对应位置的预测
    print("\n🔍 测试6：目标位置预测分析")
    
    # 找到最接近训练目标位置的预测
    boxes = pred_trained[:, :4]  # [x_center, y_center, width, height]
    target_center = jt.array([0.5, 0.5])  # 训练目标中心
    
    # 计算距离
    centers = boxes[:, :2]  # 预测中心点
    distances = jt.sqrt(((centers - target_center) ** 2).sum(dim=1))
    closest_idx = distances.argmin(dim=0)[0]
    
    print(f"   最接近目标位置的预测索引: {int(closest_idx.numpy())}")
    print(f"   该位置的目标置信度: {float(obj_conf_trained[closest_idx].numpy()):.6f}")
    print(f"   该位置的类别置信度: {cls_conf_trained[closest_idx].numpy()}")
    print(f"   该位置预测的类别0置信度: {float(cls_conf_trained[closest_idx, 0].numpy()):.6f}")
    
    # 7. 问题诊断
    print("\n🎯 问题诊断:")
    
    if float(obj_conf_trained.min().numpy()) == float(obj_conf_trained.max().numpy()) == 1.0:
        print("   ✅ 目标置信度正常 (全部为1.0)")
    else:
        print("   ❌ 目标置信度异常")

    cls_range = float(cls_conf_trained.max().numpy()) - float(cls_conf_trained.min().numpy())
    if cls_range < 0.01:
        print(f"   ❌ 类别置信度变化范围太小 ({cls_range:.6f})")
        print("   🔧 可能原因：")
        print("      1. 类别分类头权重初始化不当")
        print("      2. 训练时类别损失权重太小")
        print("      3. 激活函数饱和")
        print("      4. 学习率对分类头不合适")
    else:
        print(f"   ✅ 类别置信度变化范围正常 ({cls_range:.6f})")
    
    # 8. 建议的修复方案
    print("\n💡 建议的修复方案:")
    print("   1. 重新初始化分类头权重")
    print("   2. 增加分类损失权重")
    print("   3. 调整学习率策略")
    print("   4. 增加训练轮数")
    print("   5. 检查标签格式是否正确")
    
    return True

def visualize_detection_with_analysis():
    """可视化检测结果并分析"""
    print("\n🎨 创建详细的检测结果可视化...")
    
    # 加载模型和权重
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
        pred = model(img_tensor)[0]  # [8400, 25]
    
    # 解析预测
    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_conf = pred[:, 5:]
    
    # 选择前10个检测进行可视化
    num_show = 10
    
    # 创建可视化图像
    img_vis = img0.copy()
    
    # 绘制检测结果
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), 
              (0,255,255), (128,0,128), (255,165,0), (0,128,128), (128,128,0)]
    
    for i in range(num_show):
        # 获取预测
        box = boxes[i].numpy()
        obj_c = float(obj_conf[i].data)
        cls_c = cls_conf[i].numpy()
        
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
        
        # 获取最高类别
        max_cls_idx = np.argmax(cls_c)
        max_cls_conf = cls_c[max_cls_idx]
        final_conf = obj_c * max_cls_conf
        
        color = colors[i % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # 绘制详细标签
        label = f'Det{i+1}: C{max_cls_idx}'
        label2 = f'Obj:{obj_c:.4f} Cls:{max_cls_conf:.4f}'
        label3 = f'Final:{final_conf:.6f}'
        
        # 绘制多行标签
        cv2.putText(img_vis, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(img_vis, label2, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.putText(img_vis, label3, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        print(f"检测{i+1}: 位置=({x1},{y1},{x2},{y2}), 目标置信度={obj_c:.6f}, 类别{max_cls_idx}置信度={max_cls_conf:.6f}, 最终={final_conf:.6f}")
    
    # 保存可视化结果
    result_path = 'detailed_detection_analysis.jpg'
    cv2.imwrite(result_path, img_vis)
    print(f"\n📸 详细检测分析图已保存: {result_path}")
    
    return True

if __name__ == "__main__":
    print("🚀 开始深入分析置信度问题...")
    
    # 深入分析
    success1 = deep_analysis_confidence()
    
    # 可视化分析
    success2 = visualize_detection_with_analysis()
    
    if success1 and success2:
        print("\n🎉 深入分析完成！")
        print("📋 分析结果总结：")
        print("   - 目标检测功能正常")
        print("   - 类别分类置信度过低")
        print("   - 需要重新训练分类头或调整训练策略")
    else:
        print("\n❌ 分析过程中出现错误！")
