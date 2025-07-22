#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试Jittor Gold-YOLO Nano版本
新芽第二阶段：验证Nano版本参数量
"""

import os
import sys
import jittor as jt
from pathlib import Path

# 设置Jittor
jt.flags.use_cuda = 1

def test_jittor_nano():
    """测试Jittor Nano版本"""
    print("🔍 测试Jittor Gold-YOLO Nano版本")
    print("=" * 60)
    
    # 添加Jittor路径
    jittor_root = Path("/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor")
    if str(jittor_root) not in sys.path:
        sys.path.append(str(jittor_root))
    
    try:
        from gold_yolo.models.gold_yolo import GoldYOLO
        
        # 创建模型
        print("🚀 创建Jittor Nano模型...")
        model = GoldYOLO(num_classes=20)
        
        # 获取模型信息
        model_info = model.get_model_info()
        
        print(f"\n📋 Jittor Nano配置:")
        print(f"   depth_multiple: {model_info['depth_multiple']}")
        print(f"   width_multiple: {model_info['width_multiple']}")
        print(f"   通道数: {model_info['channels']}")
        print(f"   重复次数: {model_info['repeats']}")
        
        print(f"\n📊 Jittor Nano参数统计:")
        print(f"   总参数: {model_info['total_params']:,} ({model_info['total_params']/1e6:.2f}M)")
        print(f"   可训练参数: {model_info['trainable_params']:,} ({model_info['trainable_params']/1e6:.2f}M)")
        
        # 测试前向传播
        print(f"\n🧪 测试前向传播...")
        test_input = jt.randn(1, 3, 640, 640)
        output = model(test_input)
        
        print(f"   前向传播: ✅ 成功")
        print(f"   输出格式: {type(output)}")
        
        if isinstance(output, (list, tuple)):
            print(f"   输出长度: {len(output)}")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"   输出[{i}]: {out.shape}")
        
        # 保存结果到文件
        result = {
            'total_params': model_info['total_params'],
            'trainable_params': model_info['trainable_params'],
            'depth_multiple': model_info['depth_multiple'],
            'width_multiple': model_info['width_multiple'],
            'channels': model_info['channels'],
            'repeats': model_info['repeats'],
            'success': True
        }
        
        import json
        with open('/home/kyc/project/GOLD-YOLO/jittor_nano_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Jittor Nano测试成功！")
        print(f"📁 结果已保存到: jittor_nano_result.json")
        
        return result
        
    except Exception as e:
        print(f"❌ Jittor Nano测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        error_result = {'success': False, 'error': str(e)}
        import json
        with open('/home/kyc/project/GOLD-YOLO/jittor_nano_result.json', 'w') as f:
            json.dump(error_result, f, indent=2)
        
        return error_result

def main():
    """主函数"""
    print("🔄 Jittor Gold-YOLO Nano版本测试")
    print("新芽第二阶段：切换到Nano版本实现")
    print("=" * 60)
    
    # 测试Jittor版本
    result = test_jittor_nano()
    
    if result['success']:
        print(f"\n🎉 Jittor Nano版本测试成功！")
        print(f"   参数量: {result['total_params']/1e6:.2f}M")
        print(f"   width_multiple: {result['width_multiple']}")
        print(f"💡 现在可以测试PyTorch版本进行对比")
    else:
        print(f"\n❌ Jittor Nano版本测试失败！")
        print(f"💡 需要检查模型实现")
    
    return result['success']

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Jittor Nano准备完成！")
    else:
        print(f"\n⚠️ Jittor Nano需要修复")
