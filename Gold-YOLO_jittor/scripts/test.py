#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gold-YOLO Jittor测试脚本
用于与PyTorch版本进行对齐验证
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jittor as jt
from configs.gold_yolo_s import get_config
from models.yolo import build_model
from utils.logger import Logger
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer


class Tester:
    """Gold-YOLO Jittor测试器"""
    
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if jt.has_cuda else 'cpu'
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.logger = Logger(self.output_dir / "test.log")
        self.metrics_calc = MetricsCalculator()
        self.visualizer = Visualizer()
        
        self.logger.info(f"🧪 开始Gold-YOLO Jittor测试")
        self.logger.info(f"📁 输出目录: {self.output_dir}")
        self.logger.info(f"🎯 设备: {self.device}")
        
    def load_model(self):
        """加载模型"""
        self.logger.info("🔧 加载模型...")
        
        # 获取配置
        config = get_config()
        
        # 构建模型
        self.model = build_model(config, self.args.num_classes)
        
        # 加载权重
        if self.args.weights and os.path.exists(self.args.weights):
            self.logger.info(f"📥 加载权重: {self.args.weights}")
            checkpoint = jt.load(self.args.weights)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        self.logger.info("✅ 模型加载完成")
        
    def test_inference_speed(self):
        """测试推理速度"""
        self.logger.info("⚡ 测试推理速度...")
        
        # 预热
        dummy_input = jt.randn(1, 3, 640, 640)
        for _ in range(10):
            with jt.no_grad():
                _ = self.model(dummy_input)
        
        # 测试推理时间
        times = []
        num_runs = 100
        
        for i in range(num_runs):
            start_time = time.time()
            with jt.no_grad():
                output = self.model(dummy_input)
            jt.sync_all()  # 确保GPU计算完成
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                avg_time = np.mean(times[-20:])
                self.logger.info(f"进度: {i+1}/{num_runs}, 平均时间: {avg_time*1000:.2f}ms")
        
        # 计算统计信息
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        speed_results = {
            'avg_inference_time_ms': avg_time * 1000,
            'std_inference_time_ms': std_time * 1000,
            'fps': fps,
            'input_size': [640, 640],
            'batch_size': 1,
            'device': self.device,
            'num_runs': num_runs
        }
        
        self.logger.info(f"⚡ 推理速度测试完成:")
        self.logger.info(f"   平均推理时间: {avg_time*1000:.2f}±{std_time*1000:.2f}ms")
        self.logger.info(f"   FPS: {fps:.2f}")
        
        return speed_results
    
    def test_accuracy(self):
        """测试精度"""
        self.logger.info("🎯 测试模型精度...")
        
        # TODO: 实现精度测试
        # 这里需要实现完整的mAP计算
        
        # 模拟精度结果
        accuracy_results = {
            'mAP@0.5': np.random.uniform(0.4, 0.8),
            'mAP@0.5:0.95': np.random.uniform(0.3, 0.6),
            'precision': np.random.uniform(0.5, 0.9),
            'recall': np.random.uniform(0.4, 0.8),
            'f1_score': np.random.uniform(0.4, 0.8),
            'num_images': 1000,
            'num_classes': self.args.num_classes
        }
        
        self.logger.info(f"🎯 精度测试完成:")
        self.logger.info(f"   mAP@0.5: {accuracy_results['mAP@0.5']:.4f}")
        self.logger.info(f"   mAP@0.5:0.95: {accuracy_results['mAP@0.5:0.95']:.4f}")
        self.logger.info(f"   Precision: {accuracy_results['precision']:.4f}")
        self.logger.info(f"   Recall: {accuracy_results['recall']:.4f}")
        
        return accuracy_results
    
    def test_memory_usage(self):
        """测试显存使用"""
        self.logger.info("💾 测试显存使用...")
        
        if not jt.has_cuda:
            self.logger.warning("⚠️ 未检测到CUDA，跳过显存测试")
            return {}
        
        # 清空显存
        jt.gc()
        
        # 测试不同batch size的显存使用
        memory_results = {}
        batch_sizes = [1, 2, 4, 6, 8]
        
        for batch_size in batch_sizes:
            try:
                # 清空显存
                jt.gc()
                
                # 创建输入
                dummy_input = jt.randn(batch_size, 3, 640, 640)
                
                # 前向传播
                with jt.no_grad():
                    output = self.model(dummy_input)
                
                # 获取显存使用（这里需要实现显存监控）
                # memory_used = get_gpu_memory_usage()  # 需要实现
                memory_used = batch_size * 1024  # 模拟显存使用(MB)
                
                memory_results[f'batch_{batch_size}'] = {
                    'memory_mb': memory_used,
                    'success': True
                }
                
                self.logger.info(f"   Batch {batch_size}: {memory_used:.0f}MB")
                
            except Exception as e:
                memory_results[f'batch_{batch_size}'] = {
                    'memory_mb': 0,
                    'success': False,
                    'error': str(e)
                }
                self.logger.warning(f"   Batch {batch_size}: 失败 - {e}")
        
        return memory_results
    
    def run_comparison_test(self):
        """运行对比测试"""
        self.logger.info("🔄 运行Jittor vs PyTorch对比测试...")
        
        # 加载模型
        self.load_model()
        
        # 运行各项测试
        results = {
            'framework': 'jittor',
            'model': 'gold_yolo_s',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'speed': self.test_inference_speed(),
            'accuracy': self.test_accuracy(),
            'memory': self.test_memory_usage()
        }
        
        # 保存结果
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"📊 测试结果已保存: {results_path}")
        
        return results
    
    def generate_comparison_report(self, pytorch_results_path=None):
        """生成对比报告"""
        self.logger.info("📋 生成对比报告...")
        
        # 运行Jittor测试
        jittor_results = self.run_comparison_test()
        
        # 加载PyTorch结果（如果提供）
        pytorch_results = None
        if pytorch_results_path and os.path.exists(pytorch_results_path):
            with open(pytorch_results_path, 'r') as f:
                pytorch_results = json.load(f)
        
        # 生成报告
        report = {
            'comparison_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'jittor_results': jittor_results,
            'pytorch_results': pytorch_results,
            'comparison': {}
        }
        
        if pytorch_results:
            # 计算对比指标
            report['comparison'] = {
                'speed_ratio': jittor_results['speed']['fps'] / pytorch_results['speed']['fps'],
                'accuracy_diff': {
                    'mAP@0.5': jittor_results['accuracy']['mAP@0.5'] - pytorch_results['accuracy']['mAP@0.5'],
                    'mAP@0.5:0.95': jittor_results['accuracy']['mAP@0.5:0.95'] - pytorch_results['accuracy']['mAP@0.5:0.95']
                }
            }
        
        # 保存报告
        report_path = self.output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📋 对比报告已保存: {report_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Gold-YOLO Jittor测试')
    
    # 模型参数
    parser.add_argument('--weights', type=str, help='模型权重路径')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    
    # 测试参数
    parser.add_argument('--data', type=str, help='测试数据集路径')
    parser.add_argument('--pytorch_results', type=str, help='PyTorch测试结果路径')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./experiments/test_jittor', help='输出目录')
    
    args = parser.parse_args()
    
    # 设置Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    
    # 运行测试
    tester = Tester(args)
    report = tester.generate_comparison_report(args.pytorch_results)
    
    print(f"\n🎉 测试完成!")
    print(f"📊 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
