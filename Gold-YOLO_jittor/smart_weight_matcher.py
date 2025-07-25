#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
智能权重匹配器
提高PyTorch权重到Jittor模型的匹配率
"""

import os
import sys
import numpy as np
import jittor as jt
from pathlib import Path

# 设置Jittor
jt.flags.use_cuda = 1 if jt.has_cuda else 0

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from pytorch_aligned_model import PyTorchAlignedGoldYOLO


class SmartWeightMatcher:
    """智能权重匹配器"""
    
    def __init__(self):
        """初始化"""
        self.pytorch_weights_path = "weights/pytorch_original_weights.npz"
        self.smart_matched_weights_path = "weights/smart_matched_weights.npz"
        
        print("🧠 智能权重匹配器")
        print("   提高PyTorch权重到Jittor模型的匹配率")
        print("=" * 80)
    
    def analyze_weight_patterns(self):
        """分析权重命名模式"""
        print("\n🔍 分析权重命名模式")
        print("-" * 60)
        
        # 加载PyTorch权重
        pytorch_weights = np.load(self.pytorch_weights_path)
        
        # 创建Jittor模型
        jittor_model = PyTorchAlignedGoldYOLO(num_classes=20)
        jittor_params = dict(jittor_model.named_parameters())
        
        print(f"   PyTorch权重: {len(pytorch_weights)}")
        print(f"   Jittor参数: {len(jittor_params)}")
        
        # 分析命名模式
        pytorch_patterns = {}
        jittor_patterns = {}
        
        # PyTorch模式分析
        for name in pytorch_weights.keys():
            if 'num_batches_tracked' in name:
                continue
            parts = name.split('.')
            if len(parts) >= 3:
                pattern = '.'.join(parts[:3])
                if pattern not in pytorch_patterns:
                    pytorch_patterns[pattern] = []
                pytorch_patterns[pattern].append(name)
        
        # Jittor模式分析
        for name in jittor_params.keys():
            parts = name.split('.')
            if len(parts) >= 3:
                pattern = '.'.join(parts[:3])
                if pattern not in jittor_patterns:
                    jittor_patterns[pattern] = []
                jittor_patterns[pattern].append(name)
        
        print(f"\n   📊 PyTorch权重模式:")
        for pattern, names in sorted(pytorch_patterns.items()):
            print(f"      {pattern}: {len(names)}个参数")
        
        print(f"\n   📊 Jittor参数模式:")
        for pattern, names in sorted(jittor_patterns.items()):
            print(f"      {pattern}: {len(names)}个参数")
        
        return pytorch_patterns, jittor_patterns, pytorch_weights, jittor_params
    
    def create_smart_mapping_rules(self, pytorch_patterns, jittor_patterns, pytorch_weights, jittor_params):
        """创建智能映射规则"""
        print(f"\n🧠 创建智能映射规则")
        print("-" * 60)
        
        mapping_rules = {}
        
        # 1. 直接匹配
        direct_matches = 0
        for jt_name, jt_param in jittor_params.items():
            if jt_name in pytorch_weights:
                pt_weight = pytorch_weights[jt_name]
                if pt_weight.shape == tuple(jt_param.shape):
                    mapping_rules[jt_name] = jt_name
                    direct_matches += 1
        
        print(f"   ✅ 直接匹配: {direct_matches}个")
        
        # 2. 模式匹配
        pattern_matches = 0
        
        # 检测头映射规则
        detect_mapping = {
            'detect.stems': 'detect.stems',
            'detect.cls_convs': 'detect.cls_convs', 
            'detect.reg_convs': 'detect.reg_convs',
            'detect.cls_preds': 'detect.cls_preds',
            'detect.reg_preds': 'detect.reg_preds'
        }
        
        for jt_pattern, pt_pattern in detect_mapping.items():
            if jt_pattern in jittor_patterns and pt_pattern in pytorch_patterns:
                jt_names = jittor_patterns[jt_pattern]
                pt_names = pytorch_patterns[pt_pattern]
                
                # 按索引匹配
                for jt_name in jt_names:
                    jt_parts = jt_name.split('.')
                    if len(jt_parts) >= 4:
                        layer_idx = jt_parts[2]  # 0, 1, 2
                        param_type = jt_parts[-1]  # weight, bias
                        
                        # 寻找对应的PyTorch参数
                        for pt_name in pt_names:
                            pt_parts = pt_name.split('.')
                            if (len(pt_parts) >= 4 and 
                                pt_parts[2] == layer_idx and 
                                pt_parts[-1] == param_type):
                                
                                # 检查形状匹配
                                if (jt_name in jittor_params and 
                                    pt_name in pytorch_weights):
                                    jt_shape = tuple(jittor_params[jt_name].shape)
                                    pt_shape = pytorch_weights[pt_name].shape
                                    
                                    if jt_shape == pt_shape:
                                        mapping_rules[jt_name] = pt_name
                                        pattern_matches += 1
                                        print(f"      映射: {pt_name} -> {jt_name}")
                                        break
        
        print(f"   ✅ 模式匹配: {pattern_matches}个")
        
        # 3. 形状匹配
        shape_matches = 0
        
        for jt_name, jt_param in jittor_params.items():
            if jt_name not in mapping_rules:  # 避免重复映射
                jt_shape = tuple(jt_param.shape)
                
                # 寻找相同形状的PyTorch权重
                for pt_name, pt_weight in pytorch_weights.items():
                    if 'num_batches_tracked' in pt_name:
                        continue
                    
                    if pt_weight.shape == jt_shape:
                        # 检查语义相似性
                        if self.check_semantic_similarity(jt_name, pt_name):
                            mapping_rules[jt_name] = pt_name
                            shape_matches += 1
                            print(f"      形状匹配: {pt_name} -> {jt_name}")
                            break
        
        print(f"   ✅ 形状匹配: {shape_matches}个")
        
        total_mappings = len(mapping_rules)
        coverage = total_mappings / len(jittor_params) * 100
        
        print(f"\n   📊 映射统计:")
        print(f"      总映射规则: {total_mappings}")
        print(f"      权重覆盖率: {coverage:.1f}%")
        
        return mapping_rules
    
    def check_semantic_similarity(self, jt_name, pt_name):
        """检查语义相似性"""
        jt_parts = set(jt_name.split('.'))
        pt_parts = set(pt_name.split('.'))
        
        # 关键词重叠
        overlap = jt_parts & pt_parts
        
        # 至少要有2个关键词重叠，包括weight/bias
        return len(overlap) >= 2 and ('weight' in overlap or 'bias' in overlap)
    
    def apply_smart_mapping(self, mapping_rules, pytorch_weights, jittor_model):
        """应用智能映射"""
        print(f"\n🔧 应用智能映射")
        print("-" * 60)
        
        # 创建最终权重字典
        final_weights = {}
        
        for jt_name, pt_name in mapping_rules.items():
            if pt_name in pytorch_weights:
                weight = pytorch_weights[pt_name].astype(np.float32)
                final_weights[jt_name] = weight
        
        print(f"   ✅ 准备加载权重: {len(final_weights)}个")
        
        # 加载权重到模型
        try:
            jt_state_dict = {}
            for name, weight in final_weights.items():
                jt_state_dict[name] = jt.array(weight)
            
            jittor_model.load_state_dict(jt_state_dict)
            jittor_model.eval()
            
            print(f"   ✅ 权重加载成功")
            
            # 测试推理
            test_input = jt.randn(1, 3, 640, 640)
            with jt.no_grad():
                output = jittor_model(test_input)
            
            if isinstance(output, list):
                detections, featmaps = output
                print(f"   🚀 推理测试:")
                print(f"      输出格式: list[detections, featmaps] ✅")
                print(f"      检测形状: {detections.shape}")
                print(f"      特征图数: {len(featmaps)}")
                
                # 分析检测结果
                det = detections[0]  # [anchors, 25]
                obj_conf = det[:, 4]
                cls_probs = det[:, 5:]
                max_cls_probs = jt.max(cls_probs, dim=1)[0]
                total_conf = obj_conf * max_cls_probs
                
                print(f"      目标置信度范围: [{obj_conf.min():.6f}, {obj_conf.max():.6f}]")
                print(f"      目标置信度唯一值: {len(jt.unique(obj_conf))}")
                print(f"      最高总置信度: {total_conf.max():.6f}")
                print(f"      >0.1检测数: {(total_conf > 0.1).sum()}")
                
                # 保存智能匹配的权重
                np.savez(self.smart_matched_weights_path, **final_weights)
                print(f"   💾 智能匹配权重已保存: {self.smart_matched_weights_path}")
                
                # 评估效果
                coverage = len(final_weights) / len(dict(jittor_model.named_parameters())) * 100
                has_detections = (total_conf > 0.1).sum() > 0
                obj_conf_diversity = len(jt.unique(obj_conf)) > 100
                
                if coverage > 80 and has_detections and obj_conf_diversity:
                    print(f"   🎉 智能匹配成功!")
                    return True, coverage
                elif coverage > 60:
                    print(f"   ✅ 智能匹配良好")
                    return True, coverage
                else:
                    print(f"   ⚠️ 智能匹配一般")
                    return False, coverage
                    
        except Exception as e:
            print(f"   ❌ 权重加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0
    
    def run_smart_matching(self):
        """运行智能匹配"""
        print("🧠 运行智能权重匹配")
        print("=" * 80)
        
        # 1. 分析权重模式
        pytorch_patterns, jittor_patterns, pytorch_weights, jittor_params = self.analyze_weight_patterns()
        
        # 2. 创建智能映射规则
        mapping_rules = self.create_smart_mapping_rules(pytorch_patterns, jittor_patterns, pytorch_weights, jittor_params)
        
        # 3. 创建新的Jittor模型
        jittor_model = PyTorchAlignedGoldYOLO(num_classes=20)
        
        # 4. 应用智能映射
        success, coverage = self.apply_smart_mapping(mapping_rules, pytorch_weights, jittor_model)
        
        print(f"\n🎉 智能权重匹配完成!")
        print("=" * 80)
        
        print(f"📊 匹配结果:")
        print(f"   权重覆盖率: {coverage:.1f}%")
        print(f"   映射规则数: {len(mapping_rules)}")
        
        if success:
            print(f"   🎯 智能匹配成功!")
            print(f"   建议使用智能匹配权重: {self.smart_matched_weights_path}")
        else:
            print(f"   ⚠️ 仍需进一步优化")
        
        return success


def main():
    """主函数"""
    matcher = SmartWeightMatcher()
    success = matcher.run_smart_matching()
    
    if success:
        print(f"\n🏆 智能权重匹配成功!")
        print(f"   Gold-YOLO Jittor版本权重覆盖率大幅提升")
    else:
        print(f"\n🔧 继续优化中...")


if __name__ == '__main__':
    main()
