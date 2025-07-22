#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试PyTorch Gold-YOLO Nano版本
新芽第二阶段：验证Nano版本参数量
"""

import os
import sys
import torch
from pathlib import Path

def test_pytorch_nano():
    """测试PyTorch Nano版本"""
    print("🔍 测试PyTorch Gold-YOLO Nano版本")
    print("=" * 60)
    
    # 添加PyTorch路径
    pytorch_root = Path("/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch")
    if str(pytorch_root) not in sys.path:
        sys.path.append(str(pytorch_root))
    
    try:
        from yolov6.models.yolo import Model
        from yolov6.utils.config import Config
        
        # 加载Nano配置
        config_path = pytorch_root / "configs" / "gold_yolo-n.py"
        cfg = Config.fromfile(str(config_path))
        
        # 添加缺失的配置参数
        if not hasattr(cfg, 'training_mode'):
            cfg.training_mode = 'repvgg'
        if not hasattr(cfg, 'num_classes'):
            cfg.num_classes = 20
        
        print(f"📋 PyTorch Nano配置:")
        print(f"   模型类型: {cfg.model.type}")
        print(f"   depth_multiple: {cfg.model.depth_multiple}")
        print(f"   width_multiple: {cfg.model.width_multiple}")
        
        # 检查backbone配置
        backbone = cfg.model.backbone
        print(f"\n🏗️ Backbone配置:")
        print(f"   类型: {backbone.type}")
        print(f"   重复次数: {backbone.num_repeats}")
        print(f"   输出通道: {backbone.out_channels}")
        
        # 检查neck配置
        neck = cfg.model.neck
        print(f"\n🔗 Neck配置:")
        print(f"   类型: {neck.type}")
        print(f"   重复次数: {neck.num_repeats}")
        print(f"   输出通道: {neck.out_channels}")
        
        if hasattr(neck, 'extra_cfg'):
            extra = neck.extra_cfg
            print(f"   🔧 额外配置:")
            print(f"      fusion_in: {extra.get('fusion_in', 'N/A')}")
            print(f"      embed_dim_p: {extra.get('embed_dim_p', 'N/A')}")
            print(f"      embed_dim_n: {extra.get('embed_dim_n', 'N/A')}")
        
        # 创建模型
        print(f"\n🚀 创建PyTorch Nano模型...")
        model = Model(cfg, channels=3, num_classes=20)
        
        # 计算参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 PyTorch Nano参数统计:")
        print(f"   总参数: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 分析各模块参数
        print(f"\n🔍 各模块参数分布:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            percentage = module_params / total_params * 100
            print(f"   {name:15s}: {module_params:8,} ({percentage:5.1f}%)")
        
        # 测试前向传播
        print(f"\n🧪 测试前向传播...")
        test_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   前向传播: ✅ 成功")
        print(f"   输出格式: {type(output)}")
        
        if isinstance(output, (list, tuple)):
            print(f"   输出长度: {len(output)}")
            if len(output) >= 2:
                pred_tuple, featmaps = output
                if isinstance(pred_tuple, tuple) and len(pred_tuple) >= 3:
                    print(f"   分类预测: {pred_tuple[1].shape}")
                    print(f"   回归预测: {pred_tuple[2].shape}")
        
        # 保存结果
        result = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'depth_multiple': cfg.model.depth_multiple,
            'width_multiple': cfg.model.width_multiple,
            'backbone_repeats': backbone.num_repeats,
            'backbone_channels': backbone.out_channels,
            'neck_repeats': neck.num_repeats,
            'neck_channels': neck.out_channels,
            'success': True
        }
        
        import json
        with open('/home/kyc/project/GOLD-YOLO/pytorch_nano_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ PyTorch Nano测试成功！")
        print(f"📁 结果已保存到: pytorch_nano_result.json")
        
        return result
        
    except Exception as e:
        print(f"❌ PyTorch Nano测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        error_result = {'success': False, 'error': str(e)}
        import json
        with open('/home/kyc/project/GOLD-YOLO/pytorch_nano_result.json', 'w') as f:
            json.dump(error_result, f, indent=2)
        
        return error_result

def compare_with_jittor():
    """对比Jittor结果"""
    print(f"\n📊 PyTorch vs Jittor Nano版本对比")
    print("=" * 60)
    
    # 读取Jittor结果
    try:
        import json
        with open('/home/kyc/project/GOLD-YOLO/jittor_nano_result.json', 'r') as f:
            jittor_result = json.load(f)
        
        with open('/home/kyc/project/GOLD-YOLO/pytorch_nano_result.json', 'r') as f:
            pytorch_result = json.load(f)
        
        if not jittor_result['success'] or not pytorch_result['success']:
            print("❌ 无法进行对比，因为有版本测试失败")
            return False
        
        # 参数量对比
        pt_params = pytorch_result['total_params']
        jt_params = jittor_result['total_params']
        
        print(f"📈 参数量对比:")
        print(f"   PyTorch Nano: {pt_params:,} ({pt_params/1e6:.2f}M)")
        print(f"   Jittor Nano:  {jt_params:,} ({jt_params/1e6:.2f}M)")
        print(f"   差异: {abs(pt_params - jt_params):,}")
        print(f"   比例: {pt_params/jt_params:.3f}x" if jt_params > 0 else "   比例: N/A")
        
        # 判断对齐程度
        diff_percentage = abs(pt_params - jt_params) / pt_params * 100
        
        print(f"\n🎯 对齐评估:")
        print(f"   参数差异百分比: {diff_percentage:.1f}%")
        
        if diff_percentage < 5:
            print(f"   ✅ 优秀对齐 (差异<5%)")
            alignment = "优秀"
        elif diff_percentage < 15:
            print(f"   ✅ 良好对齐 (差异<15%)")
            alignment = "良好"
        elif diff_percentage < 30:
            print(f"   ⚠️ 一般对齐 (差异<30%)")
            alignment = "一般"
        else:
            print(f"   ❌ 对齐较差 (差异>30%)")
            alignment = "较差"
        
        # 配置对比
        print(f"\n🔧 配置对比:")
        print(f"   depth_multiple: PT={pytorch_result['depth_multiple']}, JT={jittor_result['depth_multiple']}")
        print(f"   width_multiple: PT={pytorch_result['width_multiple']}, JT={jittor_result['width_multiple']}")
        
        return alignment in ["优秀", "良好"]
        
    except Exception as e:
        print(f"❌ 对比失败: {e}")
        return False

def main():
    """主函数"""
    print("🔄 PyTorch Gold-YOLO Nano版本测试")
    print("新芽第二阶段：切换到Nano版本实现")
    print("=" * 60)
    
    # 测试PyTorch版本
    result = test_pytorch_nano()
    
    if result['success']:
        print(f"\n🎉 PyTorch Nano版本测试成功！")
        print(f"   参数量: {result['total_params']/1e6:.2f}M")
        print(f"   width_multiple: {result['width_multiple']}")
        
        # 对比Jittor版本
        comparison_success = compare_with_jittor()
        
        if comparison_success:
            print(f"\n🎉 Nano版本对齐成功！")
            print(f"💡 现在可以进行训练对比")
        else:
            print(f"\n⚠️ Nano版本对齐需要改进")
            print(f"💡 需要进一步调整实现")
    else:
        print(f"\n❌ PyTorch Nano版本测试失败！")
        print(f"💡 需要检查配置和环境")
    
    return result['success']

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ PyTorch Nano准备完成！")
    else:
        print(f"\n⚠️ PyTorch Nano需要修复")
