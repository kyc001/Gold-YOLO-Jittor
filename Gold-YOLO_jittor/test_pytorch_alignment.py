#!/usr/bin/env python3
"""
测试PyTorch对齐后的版本
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

def test_pytorch_alignment():
    """测试PyTorch对齐"""
    print(f"🔧 测试PyTorch对齐版本")
    print("=" * 60)
    
    # 准备数据
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    print(f"📊 输入数据:")
    print(f"   图像张量: {img_tensor.shape}")
    
    # 创建模型
    print(f"\n🎯 创建模型:")
    model = create_perfect_gold_yolo_model()
    
    # 统计参数
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    print(f"   总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 前向传播 - 使用eval模式对齐PyTorch
    print(f"\n🔄 前向传播:")
    model.eval()  # 使用eval模式对齐PyTorch版本
    outputs = model(img_tensor)
    
    print(f"   输出类型: {type(outputs)}")
    if isinstance(outputs, (list, tuple)):
        for i, output in enumerate(outputs):
            if hasattr(output, 'shape'):
                print(f"     输出{i}: {output.shape}")
            else:
                print(f"     输出{i}: {type(output)}")
    elif hasattr(outputs, 'shape'):
        print(f"   输出形状: {outputs.shape}")
    
    # 对比PyTorch版本
    print(f"\n📊 对比PyTorch版本:")
    print(f"   PyTorch参数: 5,617,930 (5.62M)")
    print(f"   Jittor参数:  {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   参数差异: {abs(total_params - 5617930):,}")
    
    print(f"\n   PyTorch输出: [1,5249,25]")
    if hasattr(outputs, 'shape'):
        jittor_output_shape = outputs.shape
        print(f"   Jittor输出:  {jittor_output_shape}")

        # 检查输出格式是否对齐
        if len(jittor_output_shape) == 3 and jittor_output_shape[-1] == 25:
            print(f"   ✅ 输出格式完全对齐！")
        else:
            print(f"   ❌ 输出格式不对齐！期望[1,5249,25]，实际{jittor_output_shape}")
    elif isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        if hasattr(outputs[1], 'shape'):
            jittor_output_shape = outputs[1].shape
            print(f"   Jittor输出:  {jittor_output_shape}")
            print(f"   ❌ 输出格式不对齐！期望单个tensor，实际多个tensor")
    
    return {
        'model': model,
        'outputs': outputs,
        'total_params': total_params
    }

def main():
    print("🔧 PyTorch对齐测试")
    print("=" * 60)
    
    try:
        result = test_pytorch_alignment()
        
        if result:
            print(f"\n✅ 测试完成!")
            print(f"   模型参数: {result['total_params']:,}")
            print(f"   输出类型: {type(result['outputs'])}")
        else:
            print(f"\n❌ 测试失败!")
            
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
