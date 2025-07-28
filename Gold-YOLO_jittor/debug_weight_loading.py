#!/usr/bin/env python3
"""
调试权重加载
"""

import os
import sys
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model

def debug_weight_loading():
    """调试权重加载"""
    print("🔧 调试权重加载...")
    
    # 创建模型
    print("📦 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 检查模型参数
    print("🔧 检查模型初始参数...")
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'detect' in name and ('cls_pred' in name or 'reg_pred' in name):
            print(f"检测头参数 {name}: 形状={param.shape}, 数值范围=[{float(param.min().data):.6f}, {float(param.max().data):.6f}]")
    
    print(f"模型总参数量: {total_params:,}")
    
    # 加载权重文件
    weights_path = '/home/kyc/project/GOLD-YOLO/runs/train/pytorch_aligned_stable/epoch_100.pkl'
    print(f"💾 检查权重文件: {weights_path}")
    
    if os.path.exists(weights_path):
        checkpoint = jt.load(weights_path)
        print(f"权重文件类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"权重文件键: {list(checkpoint.keys())}")
            
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                print(f"模型状态字典类型: {type(model_state)}")
                print(f"模型状态字典键数量: {len(model_state)}")
                
                # 检查几个关键参数
                for key in list(model_state.keys())[:10]:
                    param = model_state[key]
                    if hasattr(param, 'shape'):
                        print(f"权重参数 {key}: 形状={param.shape}")
                    else:
                        print(f"权重参数 {key}: 类型={type(param)}")
                
                # 尝试加载权重
                print("🔧 尝试加载权重...")
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                    print(f"✅ 权重加载完成")
                    print(f"缺失的键数量: {len(missing_keys)}")
                    print(f"意外的键数量: {len(unexpected_keys)}")
                    
                    if missing_keys:
                        print(f"缺失的键（前10个）: {missing_keys[:10]}")
                    if unexpected_keys:
                        print(f"意外的键（前10个）: {unexpected_keys[:10]}")
                        
                except Exception as e:
                    print(f"❌ 权重加载失败: {e}")
                    return False
            else:
                print("❌ 权重文件中没有'model'键")
                return False
        else:
            print("❌ 权重文件不是字典格式")
            return False
    else:
        print(f"❌ 权重文件不存在: {weights_path}")
        return False
    
    # 检查加载后的参数
    print("🔧 检查加载后的参数...")
    for name, param in model.named_parameters():
        if 'detect' in name and ('cls_pred' in name or 'reg_pred' in name):
            print(f"加载后参数 {name}: 形状={param.shape}, 数值范围=[{float(param.min().data):.6f}, {float(param.max().data):.6f}]")
    
    return True

if __name__ == "__main__":
    success = debug_weight_loading()
    if success:
        print("🎉 权重加载调试完成！")
    else:
        print("❌ 权重加载调试失败！")
