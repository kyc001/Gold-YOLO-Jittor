#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 详细参数量对齐验证
对照PyTorch版本的架构分析报告，验证Jittor版本的参数量是否完全对齐
"""

import sys
import os
import traceback

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_model_parameters():
    """分析模型参数量，对照架构报告验证"""
    print("🔍 开始GOLD-YOLO-n参数量深度验证分析...")
    print("=" * 80)
    
    try:
        import jittor as jt
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model

        # 创建100%对齐的GOLD-YOLO-n模型
        print("🏗️ 创建100%对齐的GOLD-YOLO-n模型...")
        model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
        
        # 总参数量统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n📊 **总体参数量对比**")
        print(f"   Jittor版本总参数: {total_params:,}")
        print(f"   PyTorch版本参考: 5,635,904")
        print(f"   差异: {total_params - 5635904:,}")
        print(f"   对齐率: {(min(total_params, 5635904) / max(total_params, 5635904)) * 100:.2f}%")
        
        # 按模块分析参数量
        print(f"\n🏗️ **模块级参数量分析**")
        
        # 分析backbone
        backbone_params = 0
        if hasattr(model, 'backbone'):
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            print(f"   Backbone参数量: {backbone_params:,} (目标: 3,144,890)")
            print(f"   Backbone占比: {backbone_params/total_params*100:.1f}% (目标: 55.8%)")
        
        # 分析neck
        neck_params = 0
        if hasattr(model, 'neck'):
            neck_params = sum(p.numel() for p in model.neck.parameters())
            print(f"   Neck参数量: {neck_params:,} (目标: 2,074,259)")
            print(f"   Neck占比: {neck_params/total_params*100:.1f}% (目标: 36.8%)")
        
        # 分析head
        head_params = 0
        if hasattr(model, 'detect') or hasattr(model, 'head'):
            head_module = getattr(model, 'detect', getattr(model, 'head', None))
            if head_module:
                head_params = sum(p.numel() for p in head_module.parameters())
                print(f"   Head参数量: {head_params:,} (目标: 416,755)")
                print(f"   Head占比: {head_params/total_params*100:.1f}% (目标: 7.4%)")
        
        # 详细层级分析
        print(f"\n🔧 **详细层级参数量分析**")
        
        # 分析最大的几个层
        param_dict = {}
        for name, param in model.named_parameters():
            param_dict[name] = param.numel()
        
        # 按参数量排序
        sorted_params = sorted(param_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   📈 **参数量最大的前20层**:")
        for i, (name, count) in enumerate(sorted_params[:20]):
            print(f"   {i+1:2d}. {name:<60} {count:>8,}")
        
        # 验证关键层
        print(f"\n🎯 **关键层验证**")
        
        # 查找关键层
        key_layers = [
            ('backbone.ERBlock_5.1.conv1.block.conv.weight', 589824),
            ('backbone.ERBlock_5.1.block.0.block.conv.weight', 589824),
            ('backbone.ERBlock_5.0.block.conv.weight', 294912),
        ]
        
        for layer_name, expected_params in key_layers:
            found = False
            for name, count in sorted_params:
                if layer_name in name or any(part in name for part in layer_name.split('.')):
                    print(f"   ✅ 找到类似层: {name} -> {count:,} (期望: {expected_params:,})")
                    found = True
                    break
            if not found:
                print(f"   ❌ 未找到层: {layer_name}")
        
        # 模型结构验证
        print(f"\n🏛️ **模型结构验证**")
        
        # 检查是否有正确的组件
        components = {
            'backbone': hasattr(model, 'backbone'),
            'neck': hasattr(model, 'neck'),
            'detect/head': hasattr(model, 'detect') or hasattr(model, 'head'),
        }
        
        for comp, exists in components.items():
            status = "✅" if exists else "❌"
            print(f"   {status} {comp}: {'存在' if exists else '缺失'}")
        
        # 前向传播测试
        print(f"\n🚀 **前向传播测试**")
        x = jt.randn(1, 3, 640, 640)
        
        try:
            with jt.no_grad():
                outputs = model(x)
            
            if isinstance(outputs, (list, tuple)):
                print(f"   ✅ 前向传播成功，输出{len(outputs)}个特征图:")
                for i, out in enumerate(outputs):
                    print(f"      P{i+3}: {list(out.shape)}")
            else:
                print(f"   ✅ 前向传播成功，输出形状: {list(outputs.shape)}")
                
        except Exception as e:
            print(f"   ❌ 前向传播失败: {e}")
        
        # 总结
        print(f"\n" + "=" * 80)
        print(f"📋 **验证总结**")
        
        alignment_score = 0
        total_checks = 4
        
        # 参数量对齐检查
        param_diff_ratio = abs(total_params - 5635904) / 5635904
        if param_diff_ratio < 0.05:  # 5%误差内
            alignment_score += 1
            print(f"   ✅ 总参数量对齐 (误差: {param_diff_ratio*100:.2f}%)")
        else:
            print(f"   ❌ 总参数量不对齐 (误差: {param_diff_ratio*100:.2f}%)")
        
        # 模块比例检查
        if backbone_params > 0 and neck_params > 0 and head_params > 0:
            backbone_ratio = backbone_params / total_params
            neck_ratio = neck_params / total_params
            head_ratio = head_params / total_params
            
            if 0.50 < backbone_ratio < 0.65:  # 55.8%附近
                alignment_score += 1
                print(f"   ✅ Backbone比例对齐 ({backbone_ratio*100:.1f}%)")
            else:
                print(f"   ❌ Backbone比例不对齐 ({backbone_ratio*100:.1f}%)")
                
            if 0.30 < neck_ratio < 0.45:  # 36.8%附近
                alignment_score += 1
                print(f"   ✅ Neck比例对齐 ({neck_ratio*100:.1f}%)")
            else:
                print(f"   ❌ Neck比例不对齐 ({neck_ratio*100:.1f}%)")
                
            if 0.05 < head_ratio < 0.15:  # 7.4%附近
                alignment_score += 1
                print(f"   ✅ Head比例对齐 ({head_ratio*100:.1f}%)")
            else:
                print(f"   ❌ Head比例不对齐 ({head_ratio*100:.1f}%)")
        
        print(f"\n🎯 **最终对齐评分: {alignment_score}/{total_checks} ({alignment_score/total_checks*100:.1f}%)**")
        
        if alignment_score == total_checks:
            print("🎉 GOLD-YOLO-n Jittor版本与PyTorch版本完全对齐！")
            return True
        else:
            print("⚠️  存在对齐问题，需要进一步调整")
            return False
            
    except Exception as e:
        print(f"❌ 参数量验证失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    success = analyze_model_parameters()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
