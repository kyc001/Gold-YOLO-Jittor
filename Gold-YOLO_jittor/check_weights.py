#!/usr/bin/env python3
"""
检查权重文件
"""

import os
import jittor as jt

def check_weights():
    """检查权重文件"""
    weights_path = '/home/kyc/project/GOLD-YOLO/runs/train/pytorch_aligned_stable/epoch_100.pkl'
    print(f"🔧 检查权重文件: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"❌ 权重文件不存在")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(weights_path)
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
    
    try:
        # 加载权重文件
        checkpoint = jt.load(weights_path)
        print(f"权重文件类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"权重文件键: {list(checkpoint.keys())}")
            
            # 检查每个键的内容
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    print(f"键 '{key}': 字典，包含 {len(value)} 个子键")
                    if key == 'model':
                        # 检查模型参数
                        param_count = 0
                        for param_name, param_value in value.items():
                            param_count += 1
                            if hasattr(param_value, 'shape'):
                                print(f"  参数 {param_name}: 形状={param_value.shape}")
                            if param_count >= 10:  # 只显示前10个
                                print(f"  ... 还有 {len(value) - 10} 个参数")
                                break
                else:
                    print(f"键 '{key}': {type(value)}, 值={value}")
        else:
            print(f"权重文件不是字典格式，而是: {type(checkpoint)}")
            
    except Exception as e:
        print(f"❌ 加载权重文件失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    check_weights()
