#!/usr/bin/env python3
"""
调试模型输出
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

def debug_model_output():
    """调试模型输出"""
    print("🔧 调试模型输出...")
    
    # 创建模型
    print("📦 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 加载权重
    weights_path = '/home/kyc/project/GOLD-YOLO/runs/train/pytorch_aligned_stable/epoch_100.pkl'
    print(f"💾 加载权重: {weights_path}")
    
    if os.path.exists(weights_path):
        checkpoint = jt.load(weights_path)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 成功加载权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print(f"✅ 成功加载权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ 成功加载权重")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ 成功加载权重")
    else:
        print(f"❌ 权重文件不存在: {weights_path}")
        return False
    
    model.eval()
    
    # 加载测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    print(f"📸 加载测试图像: {img_path}")
    
    img0 = cv2.imread(img_path)
    print(f"原始图像尺寸: {img0.shape}")
    
    # 预处理
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
    img = jt.array(img)
    if img.ndim == 3:
        img = img.unsqueeze(0)  # 添加batch维度
    
    print(f"预处理后图像尺寸: {img.shape}")
    
    # 推理
    print("🔧 开始推理...")
    with jt.no_grad():
        outputs = model(img)
    
    print(f"模型输出数量: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"输出 {i}: 形状={output.shape}, 数值范围=[{float(output.min().data):.6f}, {float(output.max().data):.6f}]")
        
        # 检查是否有非零值
        non_zero_count = (output != 0).sum()
        print(f"输出 {i}: 非零元素数量={int(non_zero_count.data)}/{output.numel()}")
        
        # 检查前几个值
        if output.numel() > 0:
            flat_output = output.flatten()
            print(f"输出 {i}: 前10个值={flat_output[:10].numpy()}")
    
    # 检查训练模式下的输出
    print("\n🔧 检查训练模式下的输出...")
    model.train()
    with jt.no_grad():
        train_outputs = model(img)
    
    print(f"训练模式输出数量: {len(train_outputs)}")
    for i, output in enumerate(train_outputs):
        if hasattr(output, 'shape'):
            print(f"训练输出 {i}: 形状={output.shape}, 数值范围=[{float(output.min().data):.6f}, {float(output.max().data):.6f}]")
        else:
            print(f"训练输出 {i}: {type(output)}")
    
    return True

if __name__ == "__main__":
    success = debug_model_output()
    if success:
        print("🎉 模型输出调试完成！")
    else:
        print("❌ 模型输出调试失败！")
