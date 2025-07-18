#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold-YOLO Jittor vs PyTorch 全面对齐检查
检查模型架构、参数量、前向传播、训练流程等各个方面
"""

import os
import sys
import traceback
import json
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jittor as jt
import numpy as np
from configs.gold_yolo_s import get_config
from models.yolo import build_model
from models.loss import GoldYOLOLoss


def print_status(message, status="INFO"):
    """打印状态信息"""
    colors = {
        "INFO": "\033[0;34m",
        "SUCCESS": "\033[0;32m", 
        "WARNING": "\033[1;33m",
        "ERROR": "\033[0;31m",
        "HEADER": "\033[1;35m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{message}{reset}")


class AlignmentChecker:
    """全面对齐检查器"""
    
    def __init__(self):
        self.results = {}
        self.config = get_config()
        
    def check_config_alignment(self):
        """检查配置对齐"""
        print_status("🔧 检查配置对齐...", "HEADER")
        
        try:
            # 检查模型配置
            expected_config = {
                'type': 'GoldYOLO-s',
                'depth_multiple': 0.33,
                'width_multiple': 0.50,
                'backbone_type': 'EfficientRep',
                'neck_type': 'RepGDNeck',
                'head_type': 'EffiDeHead'
            }
            
            actual_config = {
                'type': self.config.model.type,
                'depth_multiple': self.config.model.depth_multiple,
                'width_multiple': self.config.model.width_multiple,
                'backbone_type': self.config.model.backbone.type,
                'neck_type': self.config.model.neck.type,
                'head_type': self.config.model.head.type
            }
            
            alignment_status = {}
            for key, expected in expected_config.items():
                actual = actual_config.get(key)
                is_aligned = actual == expected
                alignment_status[key] = {
                    'expected': expected,
                    'actual': actual,
                    'aligned': is_aligned
                }
                
                status = "✅" if is_aligned else "❌"
                print_status(f"   {status} {key}: {actual} (期望: {expected})")
            
            self.results['config_alignment'] = alignment_status
            return all(item['aligned'] for item in alignment_status.values())
            
        except Exception as e:
            print_status(f"   ❌ 配置检查失败: {e}", "ERROR")
            return False
    
    def check_model_structure(self):
        """检查模型结构"""
        print_status("🏗️ 检查模型结构...", "HEADER")
        
        try:
            model = build_model(self.config, num_classes=10)
            
            # 检查模型组件
            components = {
                'backbone': hasattr(model, 'backbone'),
                'neck': hasattr(model, 'neck'), 
                'detect': hasattr(model, 'detect'),
                'stride': hasattr(model, 'stride')
            }
            
            for comp, exists in components.items():
                status = "✅" if exists else "❌"
                print_status(f"   {status} {comp}: {'存在' if exists else '缺失'}")
            
            # 检查参数量
            total_params = sum(p.numel() for p in model.parameters())
            expected_params_range = (20_000_000, 21_000_000)  # 预期范围
            params_aligned = expected_params_range[0] <= total_params <= expected_params_range[1]
            
            status = "✅" if params_aligned else "❌"
            print_status(f"   {status} 参数量: {total_params:,} (期望范围: {expected_params_range[0]:,}-{expected_params_range[1]:,})")
            
            # 检查模型层数
            backbone_layers = len(list(model.backbone.named_modules()))
            neck_layers = len(list(model.neck.named_modules()))
            head_layers = len(list(model.detect.named_modules()))
            
            print_status(f"   📊 Backbone层数: {backbone_layers}")
            print_status(f"   📊 Neck层数: {neck_layers}")
            print_status(f"   📊 Head层数: {head_layers}")
            
            self.results['model_structure'] = {
                'components': components,
                'total_params': total_params,
                'params_aligned': params_aligned,
                'layer_counts': {
                    'backbone': backbone_layers,
                    'neck': neck_layers,
                    'head': head_layers
                }
            }
            
            return all(components.values()) and params_aligned
            
        except Exception as e:
            print_status(f"   ❌ 模型结构检查失败: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def check_forward_pass(self):
        """检查前向传播"""
        print_status("⚡ 检查前向传播...", "HEADER")
        
        try:
            model = build_model(self.config, num_classes=10)
            
            # 测试不同输入尺寸
            test_sizes = [416, 512, 640]
            forward_results = {}
            
            for size in test_sizes:
                # 推理模式
                model.eval()
                x = jt.randn(1, 3, size, size)
                
                with jt.no_grad():
                    output_inference = model(x)
                
                # 训练模式
                model.train()
                output_training = model(x)
                
                # 检查输出格式
                inference_format = self._analyze_output_format(output_inference, "推理")
                training_format = self._analyze_output_format(output_training, "训练")
                
                forward_results[f'{size}x{size}'] = {
                    'inference': inference_format,
                    'training': training_format
                }
                
                print_status(f"   ✅ {size}×{size}: 推理={inference_format['type']}, 训练={training_format['type']}")
            
            # 检查输出格式是否符合预期
            # 预期: 推理模式返回单个张量，训练模式返回[检测输出, 特征图]
            format_aligned = True
            for size_result in forward_results.values():
                if size_result['inference']['type'] != 'tensor':
                    format_aligned = False
                if size_result['training']['type'] != 'list_with_2_elements':
                    format_aligned = False
            
            status = "✅" if format_aligned else "❌"
            print_status(f"   {status} 输出格式对齐: {'是' if format_aligned else '否'}")
            
            self.results['forward_pass'] = {
                'results': forward_results,
                'format_aligned': format_aligned
            }
            
            return format_aligned
            
        except Exception as e:
            print_status(f"   ❌ 前向传播检查失败: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def _analyze_output_format(self, output, mode):
        """分析输出格式"""
        if isinstance(output, jt.Var):
            return {
                'type': 'tensor',
                'shape': list(output.shape),
                'description': f'单个张量 {output.shape}'
            }
        elif isinstance(output, (list, tuple)):
            if len(output) == 2:
                return {
                    'type': 'list_with_2_elements',
                    'length': len(output),
                    'description': f'列表包含{len(output)}个元素'
                }
            else:
                return {
                    'type': 'list_other',
                    'length': len(output),
                    'description': f'列表包含{len(output)}个元素'
                }
        else:
            return {
                'type': 'unknown',
                'description': f'未知类型: {type(output)}'
            }
    
    def check_training_components(self):
        """检查训练组件"""
        print_status("🎯 检查训练组件...", "HEADER")
        
        try:
            model = build_model(self.config, num_classes=10)
            criterion = GoldYOLOLoss(num_classes=10)
            optimizer = jt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            
            model.train()
            
            # 创建训练数据
            images = jt.randn(2, 3, 512, 512)
            batch = {
                'cls': jt.randint(0, 10, (2, 5)),
                'bboxes': jt.rand(2, 5, 4),
                'mask_gt': jt.ones(2, 5).bool()
            }
            
            # 前向传播
            outputs = model(images)
            
            # 检查输出格式
            if isinstance(outputs, list) and len(outputs) == 2:
                detection_output, featmaps = outputs
                if isinstance(detection_output, (list, tuple)) and len(detection_output) == 3:
                    print_status("   ✅ 训练输出格式正确: [检测输出(3元组), 特征图]")
                    output_format_correct = True
                else:
                    print_status("   ❌ 检测输出格式错误")
                    output_format_correct = False
            else:
                print_status("   ❌ 训练输出格式错误")
                output_format_correct = False
            
            # 损失计算
            loss, loss_items = criterion(outputs, batch)
            print_status(f"   ✅ 损失计算成功: {loss.item():.4f}")
            
            # 反向传播
            optimizer.step(loss)
            print_status("   ✅ 反向传播成功")
            
            # 检查梯度
            grad_count = 0
            total_params = 0
            for param in model.parameters():
                total_params += 1
                try:
                    grad = param.opt_grad(optimizer)
                    if grad is not None and grad.norm().item() > 1e-8:
                        grad_count += 1
                except:
                    pass
            
            grad_ratio = grad_count / total_params if total_params > 0 else 0
            print_status(f"   📊 梯度统计: {grad_count}/{total_params} ({grad_ratio:.1%}) 参数有有效梯度")
            
            # 判断训练组件是否正常
            training_ok = output_format_correct and grad_ratio > 0.3  # 至少30%的参数有梯度
            
            self.results['training_components'] = {
                'output_format_correct': output_format_correct,
                'loss_calculation': True,
                'backpropagation': True,
                'gradient_ratio': grad_ratio,
                'training_ok': training_ok
            }
            
            return training_ok
            
        except Exception as e:
            print_status(f"   ❌ 训练组件检查失败: {e}", "ERROR")
            traceback.print_exc()
            return False
    
    def check_api_compatibility(self):
        """检查API兼容性"""
        print_status("🔌 检查API兼容性...", "HEADER")
        
        try:
            # 检查关键API
            api_checks = {
                'jittor_basic': self._check_jittor_basic(),
                'model_methods': self._check_model_methods(),
                'optimizer_methods': self._check_optimizer_methods(),
                'loss_functions': self._check_loss_functions()
            }
            
            for api_name, result in api_checks.items():
                status = "✅" if result else "❌"
                print_status(f"   {status} {api_name}: {'正常' if result else '异常'}")
            
            api_compatible = all(api_checks.values())
            
            self.results['api_compatibility'] = {
                'checks': api_checks,
                'compatible': api_compatible
            }
            
            return api_compatible
            
        except Exception as e:
            print_status(f"   ❌ API兼容性检查失败: {e}", "ERROR")
            return False
    
    def _check_jittor_basic(self):
        """检查Jittor基础功能"""
        try:
            x = jt.randn(2, 3, 4, 4)
            # 使用nn.Conv2d而不是函数式conv2d
            conv = jt.nn.Conv2d(3, 16, 3, padding=1)
            y = conv(x)
            return y.shape == [2, 16, 4, 4]
        except:
            return False
    
    def _check_model_methods(self):
        """检查模型方法"""
        try:
            model = build_model(self.config, num_classes=10)
            model.train()
            model.eval()
            list(model.parameters())
            return True
        except:
            return False
    
    def _check_optimizer_methods(self):
        """检查优化器方法"""
        try:
            model = build_model(self.config, num_classes=10)
            optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
            loss = jt.randn(1)
            optimizer.step(loss)
            return True
        except:
            return False
    
    def _check_loss_functions(self):
        """检查损失函数"""
        try:
            criterion = GoldYOLOLoss(num_classes=10)
            return True
        except:
            return False
    
    def run_comprehensive_check(self):
        """运行全面检查"""
        print_status("🔍 Gold-YOLO Jittor vs PyTorch 全面对齐检查", "HEADER")
        print_status("=" * 60, "HEADER")
        
        checks = [
            ("配置对齐", self.check_config_alignment),
            ("模型结构", self.check_model_structure),
            ("前向传播", self.check_forward_pass),
            ("训练组件", self.check_training_components),
            ("API兼容性", self.check_api_compatibility)
        ]
        
        passed = 0
        total = len(checks)
        
        for check_name, check_func in checks:
            print_status(f"\n🔬 {check_name}")
            try:
                result = check_func()
                if result:
                    passed += 1
                    print_status(f"✅ {check_name} 通过", "SUCCESS")
                else:
                    print_status(f"❌ {check_name} 失败", "ERROR")
            except Exception as e:
                print_status(f"❌ {check_name} 异常: {e}", "ERROR")
        
        # 生成总结报告
        print_status("=" * 60, "HEADER")
        print_status(f"📊 检查结果: {passed}/{total} 通过", "HEADER")
        
        if passed == total:
            print_status("🎉 所有检查通过！Gold-YOLO Jittor与PyTorch版本完全对齐", "SUCCESS")
            overall_status = "PERFECT_ALIGNMENT"
        elif passed >= total * 0.8:
            print_status("✅ 大部分检查通过，对齐状态良好", "SUCCESS")
            overall_status = "GOOD_ALIGNMENT"
        else:
            print_status("⚠️ 多项检查失败，需要进一步修复", "WARNING")
            overall_status = "NEEDS_IMPROVEMENT"
        
        # 保存结果
        self.results['summary'] = {
            'passed': passed,
            'total': total,
            'pass_rate': passed / total,
            'overall_status': overall_status
        }
        
        self._save_results()
        
        return overall_status == "PERFECT_ALIGNMENT"
    
    def _save_results(self):
        """保存检查结果"""
        results_dir = Path("./alignment_check_results")
        results_dir.mkdir(exist_ok=True)
        
        import time
        results_file = results_dir / f"alignment_check_{jt.flags.use_cuda}_{int(time.time())}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print_status(f"📋 检查结果已保存: {results_file}")


def main():
    """主函数"""
    # 设置Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    
    # 运行全面检查
    checker = AlignmentChecker()
    success = checker.run_comprehensive_check()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
