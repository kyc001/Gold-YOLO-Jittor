#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
训练前模型验证 - 防止8小时训练后发现模型无法识别目标
新芽第二阶段：完整的预训练验证流程
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

import jittor as jt
import jittor.nn as nn
from PIL import Image
import cv2

# 设置Jittor
jt.flags.use_cuda = 1

# 导入完整模型
from full_pytorch_small_model import FullPyTorchGoldYOLOSmall

class PreTrainingValidator:
    """训练前验证器 - 确保模型能够正常工作"""
    
    def __init__(self):
        self.model = None
        self.test_results = {}
        
        print("🔍 训练前模型验证器")
        print("目标：确保模型在训练前就能正常工作")
    
    def test_1_model_creation(self):
        """测试1: 模型创建和参数统计"""
        print("\n" + "="*60)
        print("🧪 测试1: 模型创建和参数统计")
        
        try:
            self.model = FullPyTorchGoldYOLOSmall(num_classes=80)
            info = self.model.get_model_info()
            
            print(f"✅ 模型创建成功")
            print(f"   总参数: {info['total_params']:,}")
            print(f"   可训练参数: {info['trainable_params']:,}")
            print(f"   depth_multiple: {info['depth_multiple']}")
            print(f"   width_multiple: {info['width_multiple']}")
            
            # 验证参数量是否合理 (Small模型应该在10-20M之间)
            if 5_000_000 < info['total_params'] < 25_000_000:
                print(f"✅ 参数量合理: {info['total_params']:,}")
                self.test_results['model_creation'] = True
            else:
                print(f"❌ 参数量异常: {info['total_params']:,}")
                self.test_results['model_creation'] = False
                
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            self.test_results['model_creation'] = False
            return False
        
        return True
    
    def test_2_forward_pass(self):
        """测试2: 前向传播测试"""
        print("\n" + "="*60)
        print("🧪 测试2: 前向传播测试")
        
        if self.model is None:
            print("❌ 模型未创建，跳过测试")
            self.test_results['forward_pass'] = False
            return False
        
        try:
            # 测试不同批次大小
            batch_sizes = [1, 2, 4, 8]
            input_size = (3, 640, 640)
            
            for batch_size in batch_sizes:
                test_input = jt.randn(batch_size, *input_size)
                
                start_time = time.time()
                with jt.no_grad():
                    features, cls_pred, reg_pred = self.model(test_input)
                inference_time = time.time() - start_time
                
                print(f"✅ Batch {batch_size}: {inference_time*1000:.2f}ms")
                print(f"   输入: {test_input.shape}")
                print(f"   特征: {len(features)}层")
                print(f"   分类: {cls_pred.shape}")
                print(f"   回归: {reg_pred.shape}")
                
                # 验证输出形状
                expected_cls_shape = (batch_size, 525, 80)
                expected_reg_shape = (batch_size, 525, 68)
                
                if cls_pred.shape == expected_cls_shape and reg_pred.shape == expected_reg_shape:
                    print(f"   ✅ 输出形状正确")
                else:
                    print(f"   ❌ 输出形状错误")
                    print(f"      期望分类: {expected_cls_shape}, 实际: {cls_pred.shape}")
                    print(f"      期望回归: {expected_reg_shape}, 实际: {reg_pred.shape}")
                    self.test_results['forward_pass'] = False
                    return False
            
            self.test_results['forward_pass'] = True
            print("✅ 前向传播测试通过")
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            self.test_results['forward_pass'] = False
            return False
        
        return True
    
    def test_3_loss_function(self):
        """测试3: 损失函数测试"""
        print("\n" + "="*60)
        print("🧪 测试3: 损失函数测试")
        
        if self.model is None:
            print("❌ 模型未创建，跳过测试")
            self.test_results['loss_function'] = False
            return False
        
        try:
            # 创建损失函数 (从训练脚本复制)
            class TestYOLOLoss(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mse_loss = nn.MSELoss()
                    self.bce_loss = nn.BCEWithLogitsLoss()
                    self.lambda_box = 15.0
                    self.lambda_cls = 2.0
                    self.lambda_obj = 3.0
                    self.lambda_dfl = 3.0

                def execute(self, pred, targets=None):
                    features, cls_pred, reg_pred = pred
                    batch_size = cls_pred.shape[0]
                    num_anchors = cls_pred.shape[1]
                    num_classes = cls_pred.shape[2]
                    
                    # 创建简单目标
                    cls_targets = jt.zeros_like(cls_pred)
                    reg_targets = jt.zeros_like(reg_pred)
                    obj_mask = jt.zeros((batch_size, num_anchors))
                    
                    # 设置一些正样本
                    for b in range(batch_size):
                        num_pos = min(10, num_anchors//10)
                        for i in range(num_pos):
                            obj_mask[b, i] = 1.0
                            cls_targets[b, i, i % num_classes] = 1.0
                            reg_targets[b, i, 0] = 0.5
                            reg_targets[b, i, 1] = 0.5
                            reg_targets[b, i, 2] = 0.3
                            reg_targets[b, i, 3] = 0.3
                    
                    # 计算损失
                    pos_mask_cls = obj_mask.unsqueeze(-1).expand_as(cls_pred)
                    pos_mask_reg = obj_mask.unsqueeze(-1).expand_as(reg_pred)
                    
                    cls_loss = self.bce_loss(cls_pred * pos_mask_cls, cls_targets * pos_mask_cls)
                    reg_loss = self.mse_loss(reg_pred * pos_mask_reg, reg_targets * pos_mask_reg)
                    
                    obj_pred = jt.max(cls_pred, dim=-1)
                    if isinstance(obj_pred, tuple):
                        obj_pred = obj_pred[0]
                    obj_loss = self.bce_loss(obj_pred, obj_mask)
                    
                    total_loss = (self.lambda_box * reg_loss + 
                                 self.lambda_cls * cls_loss + 
                                 self.lambda_obj * obj_loss)
                    
                    return total_loss
            
            loss_fn = TestYOLOLoss()
            
            # 测试损失计算
            test_input = jt.randn(4, 3, 640, 640)
            outputs = self.model(test_input)
            loss = loss_fn(outputs)
            
            print(f"✅ 损失计算成功: {loss.item():.3f}")
            
            # 验证损失值是否合理 (应该在1-1000之间)
            if 0.1 < loss.item() < 1000.0:
                print(f"✅ 损失值合理: {loss.item():.3f}")
                self.test_results['loss_function'] = True
            else:
                print(f"❌ 损失值异常: {loss.item():.3f}")
                self.test_results['loss_function'] = False
                return False
            
        except Exception as e:
            print(f"❌ 损失函数测试失败: {e}")
            self.test_results['loss_function'] = False
            return False
        
        return True
    
    def test_4_gradient_flow(self):
        """测试4: 梯度流动测试"""
        print("\n" + "="*60)
        print("🧪 测试4: 梯度流动测试")
        
        if self.model is None:
            print("❌ 模型未创建，跳过测试")
            self.test_results['gradient_flow'] = False
            return False
        
        try:
            # 创建优化器
            optimizer = jt.optim.SGD(self.model.parameters(), lr=0.01)
            
            # 创建简单损失函数
            loss_fn = nn.MSELoss()
            
            # 测试梯度流动
            test_input = jt.randn(2, 3, 640, 640)
            features, cls_pred, reg_pred = self.model(test_input)
            
            # 创建简单目标
            cls_target = jt.randn_like(cls_pred)
            reg_target = jt.randn_like(reg_pred)
            
            # 计算损失
            loss = loss_fn(cls_pred, cls_target) + loss_fn(reg_pred, reg_target)
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            
            # 检查梯度
            grad_count = 0
            total_grad_norm = 0.0
            
            for name, param in self.model.named_parameters():
                if param.opt_grad(optimizer) is not None:
                    grad_count += 1
                    grad_tensor = param.opt_grad(optimizer)
                    # 修复：计算梯度范数的正确方法
                    grad_norm = float(jt.sqrt(jt.sum(grad_tensor * grad_tensor)).item())
                    total_grad_norm += grad_norm
            
            avg_grad_norm = total_grad_norm / grad_count if grad_count > 0 else 0
            
            print(f"✅ 梯度流动测试通过")
            print(f"   有梯度的参数: {grad_count}")
            print(f"   平均梯度范数: {avg_grad_norm:.6f}")
            
            if grad_count > 0 and 1e-8 < avg_grad_norm < 100.0:
                print(f"✅ 梯度范数合理")
                self.test_results['gradient_flow'] = True
            else:
                print(f"❌ 梯度范数异常")
                self.test_results['gradient_flow'] = False
                return False
            
        except Exception as e:
            print(f"❌ 梯度流动测试失败: {e}")
            self.test_results['gradient_flow'] = False
            return False
        
        return True
    
    def test_5_memory_usage(self):
        """测试5: 内存使用测试"""
        print("\n" + "="*60)
        print("🧪 测试5: 内存使用测试")
        
        if self.model is None:
            print("❌ 模型未创建，跳过测试")
            self.test_results['memory_usage'] = False
            return False
        
        try:
            # 测试不同批次大小的内存使用
            batch_sizes = [1, 4, 8, 16]
            memory_usage = []
            
            for batch_size in batch_sizes:
                # 清理内存
                jt.gc()
                
                test_input = jt.randn(batch_size, 3, 640, 640)
                
                # 前向传播
                features, cls_pred, reg_pred = self.model(test_input)
                
                # 估算内存使用 (简化)
                input_memory = test_input.numel() * 4 / 1024 / 1024  # MB
                output_memory = (cls_pred.numel() + reg_pred.numel()) * 4 / 1024 / 1024  # MB
                total_memory = input_memory + output_memory
                
                memory_usage.append(total_memory)
                print(f"✅ Batch {batch_size}: {total_memory:.1f} MB")
            
            # 检查内存增长是否线性
            if len(memory_usage) >= 2:
                growth_ratio = memory_usage[-1] / memory_usage[0]
                expected_ratio = batch_sizes[-1] / batch_sizes[0]
                
                if 0.5 * expected_ratio < growth_ratio < 2.0 * expected_ratio:
                    print(f"✅ 内存增长合理: {growth_ratio:.1f}x")
                    self.test_results['memory_usage'] = True
                else:
                    print(f"❌ 内存增长异常: {growth_ratio:.1f}x (期望约{expected_ratio:.1f}x)")
                    self.test_results['memory_usage'] = False
                    return False
            
        except Exception as e:
            print(f"❌ 内存使用测试失败: {e}")
            self.test_results['memory_usage'] = False
            return False
        
        return True
    
    def test_6_output_analysis(self):
        """测试6: 输出分析测试"""
        print("\n" + "="*60)
        print("🧪 测试6: 输出分析测试")
        
        if self.model is None:
            print("❌ 模型未创建，跳过测试")
            self.test_results['output_analysis'] = False
            return False
        
        try:
            test_input = jt.randn(4, 3, 640, 640)
            
            with jt.no_grad():
                features, cls_pred, reg_pred = self.model(test_input)
            
            # 分析分类输出
            cls_sigmoid = jt.sigmoid(cls_pred)
            cls_max = jt.max(cls_sigmoid).item()
            cls_min = jt.min(cls_sigmoid).item()
            cls_mean = jt.mean(cls_sigmoid).item()
            
            print(f"✅ 分类输出分析:")
            print(f"   范围: {cls_min:.3f} - {cls_max:.3f}")
            print(f"   均值: {cls_mean:.3f}")
            
            # 分析回归输出
            reg_max = jt.max(reg_pred).item()
            reg_min = jt.min(reg_pred).item()
            reg_mean = jt.mean(reg_pred).item()
            
            print(f"✅ 回归输出分析:")
            print(f"   范围: {reg_min:.3f} - {reg_max:.3f}")
            print(f"   均值: {reg_mean:.3f}")
            
            # 验证输出是否合理
            if (0.0 <= cls_min <= cls_max <= 1.0 and 
                -100.0 <= reg_min <= reg_max <= 100.0):
                print(f"✅ 输出范围合理")
                self.test_results['output_analysis'] = True
            else:
                print(f"❌ 输出范围异常")
                self.test_results['output_analysis'] = False
                return False
            
        except Exception as e:
            print(f"❌ 输出分析测试失败: {e}")
            self.test_results['output_analysis'] = False
            return False
        
        return True
    
    def generate_report(self):
        """生成验证报告"""
        print("\n" + "="*60)
        print("📋 训练前验证报告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        print("\n详细结果:")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
        
        # 总体评估
        if passed_tests == total_tests:
            print("\n🎉 所有测试通过！模型可以开始训练")
            return True
        elif passed_tests >= total_tests * 0.8:
            print("\n⚠️ 大部分测试通过，建议修复失败项后再训练")
            return False
        else:
            print("\n❌ 多项测试失败，必须修复后才能训练")
            return False
    
    def run_all_tests(self):
        """运行所有验证测试"""
        print("🎯 开始训练前完整验证...")
        print("目标：确保模型在8小时训练前就能正常工作")
        
        # 按顺序运行所有测试
        tests = [
            self.test_1_model_creation,
            self.test_2_forward_pass,
            self.test_3_loss_function,
            self.test_4_gradient_flow,
            self.test_5_memory_usage,
            self.test_6_output_analysis
        ]
        
        for test_func in tests:
            if not test_func():
                print(f"\n❌ 测试失败，停止后续测试")
                break
        
        # 生成报告
        return self.generate_report()


def main():
    """主函数"""
    print("🎯 Gold-YOLO 训练前验证")
    print("新芽第二阶段：防止8小时训练后发现问题")
    print("=" * 60)
    
    validator = PreTrainingValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n🚀 验证通过！可以开始长时间训练")
        print("建议：保存当前模型状态，开始正式训练")
    else:
        print("\n🛑 验证失败！请修复问题后再训练")
        print("建议：检查模型定义、损失函数、数据流程")
    
    return success


if __name__ == "__main__":
    main()
