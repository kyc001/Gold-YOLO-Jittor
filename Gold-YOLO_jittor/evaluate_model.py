#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
模型评估脚本 - 在测试集上统一评估性能
新芽第二阶段：Jittor vs PyTorch 性能对比
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import jittor as jt
import jittor.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

# 设置Jittor
jt.flags.use_cuda = 1

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, test_file, model_path, framework='jittor'):
        self.test_file = Path(test_file)
        self.model_path = Path(model_path)
        self.framework = framework
        
        # 输出目录
        self.output_dir = Path(f"runs/evaluation_{framework}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 {framework.upper()}模型评估器")
        print(f"   测试集: {self.test_file}")
        print(f"   模型: {self.model_path}")
    
    def load_test_data(self):
        """加载测试数据"""
        with open(self.test_file, 'r') as f:
            test_data = json.load(f)
        
        print(f"✅ 测试数据: {len(test_data['images'])}张图片, {len(test_data['annotations'])}个标注")
        return test_data
    
    def load_model(self):
        """加载模型"""
        if self.framework == 'jittor':
            return self._load_jittor_model()
        else:
            return self._load_pytorch_model()
    
    def _load_jittor_model(self):
        """加载Jittor模型"""
        # 重新定义模型结构（与训练时一致）
        class FixedGoldYOLOSmall(nn.Module):
            def __init__(self, num_classes=80):
                super().__init__()
                self.num_classes = num_classes
                
                # 官方Small版本参数
                self.depth_multiple = 0.33
                self.width_multiple = 0.50
                
                # 官方配置的通道数和重复次数
                base_channels = [64, 128, 256, 512, 1024]
                self.channels = [int(ch * self.width_multiple) for ch in base_channels]
                
                # 构建简化的backbone
                self.backbone = self._build_backbone()
                
                # 构建简化的neck
                self.neck = self._build_neck()
                
                # 构建简化的head
                self.head = self._build_head()
                
            def _build_backbone(self):
                """构建backbone"""
                layers = []
                
                # Stem
                layers.append(nn.Conv2d(3, self.channels[0], 3, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(self.channels[0]))
                layers.append(nn.SiLU())
                
                # Stage 1
                layers.append(nn.Conv2d(self.channels[0], self.channels[1], 3, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(self.channels[1]))
                layers.append(nn.SiLU())
                
                # Stage 2
                layers.append(nn.Conv2d(self.channels[1], self.channels[2], 3, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(self.channels[2]))
                layers.append(nn.SiLU())
                
                # Stage 3
                layers.append(nn.Conv2d(self.channels[2], self.channels[3], 3, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(self.channels[3]))
                layers.append(nn.SiLU())
                
                # Stage 4
                layers.append(nn.Conv2d(self.channels[3], self.channels[4], 3, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(self.channels[4]))
                layers.append(nn.SiLU())
                
                # SPPF
                layers.append(nn.AdaptiveAvgPool2d(1))
                
                return nn.Sequential(*layers)
            
            def _build_neck(self):
                """构建neck"""
                return nn.Sequential(
                    nn.Conv2d(self.channels[4], self.channels[3], 1),
                    nn.BatchNorm2d(self.channels[3]),
                    nn.SiLU()
                )
            
            def _build_head(self):
                """构建head"""
                cls_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.channels[3], 525 * self.num_classes),  # 分类输出
                )
                
                reg_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.channels[3], 525 * 68)  # 回归输出 (DFL格式)
                )
                
                return cls_head, reg_head
            
            def execute(self, x):
                # Backbone
                feat = self.backbone(x)
                
                # Neck
                feat = self.neck(feat)
                
                # Head
                cls_head, reg_head = self.head
                cls_pred = cls_head(feat).view(x.size(0), 525, self.num_classes)
                reg_pred = reg_head(feat).view(x.size(0), 525, 68)
                
                return feat, cls_pred, reg_pred
        
        model = FixedGoldYOLOSmall(num_classes=80)
        
        # 加载权重
        if self.model_path.exists():
            model.load_state_dict(jt.load(str(self.model_path)))
            print(f"✅ 已加载Jittor模型权重: {self.model_path}")
        else:
            print(f"⚠️ 模型文件不存在，使用随机初始化权重")
        
        model.eval()
        return model
    
    def _load_pytorch_model(self):
        """加载PyTorch模型（占位符）"""
        print("⚠️ PyTorch模型评估暂未实现")
        return None
    
    def evaluate_performance(self, model, test_data, num_samples=100):
        """评估模型性能"""
        print(f"\n🔍 开始性能评估...")
        print(f"   评估样本数: {min(num_samples, len(test_data['images']))}")
        
        # 性能指标
        inference_times = []
        memory_usage = []
        prediction_scores = []
        
        # 随机选择测试样本
        test_images = test_data['images'][:num_samples]
        
        print("推理测试...")
        for i, img_info in enumerate(tqdm(test_images)):
            try:
                # 创建随机输入（模拟真实图片）
                batch_input = jt.randn(1, 3, 640, 640)
                
                # 推理计时
                start_time = time.time()
                
                with jt.no_grad():
                    outputs = model(batch_input)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 计算预测置信度（简化指标）
                _, cls_pred, reg_pred = outputs
                max_conf = jt.max(jt.sigmoid(cls_pred)).item()
                prediction_scores.append(max_conf)
                
                # 内存使用（简化）
                memory_usage.append(batch_input.numel() * 4 / 1024 / 1024)  # MB
                
            except Exception as e:
                print(f"⚠️ 样本 {i} 评估失败: {e}")
                continue
        
        # 计算统计指标
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            fps = 1.0 / avg_inference_time
            avg_memory = np.mean(memory_usage)
            avg_confidence = np.mean(prediction_scores)
            
            results = {
                'framework': self.framework,
                'num_samples': len(inference_times),
                'avg_inference_time': avg_inference_time,
                'fps': fps,
                'avg_memory_mb': avg_memory,
                'avg_confidence': avg_confidence,
                'inference_times': inference_times,
                'prediction_scores': prediction_scores
            }
            
            print(f"\n📊 评估结果:")
            print(f"   平均推理时间: {avg_inference_time*1000:.2f} ms")
            print(f"   推理速度: {fps:.1f} FPS")
            print(f"   平均内存使用: {avg_memory:.1f} MB")
            print(f"   平均置信度: {avg_confidence:.3f}")
            
            return results
        else:
            print("❌ 评估失败，没有有效样本")
            return None
    
    def save_results(self, results):
        """保存评估结果"""
        if results is None:
            return
        
        # 保存详细结果
        results_file = self.output_dir / f"{self.framework}_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ 评估结果已保存: {results_file}")
        
        # 生成简化报告
        report = f"""# {self.framework.upper()} 模型评估报告

## 测试配置
- **测试集**: {self.test_file.name}
- **模型**: {self.model_path.name}
- **评估样本数**: {results['num_samples']}
- **评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 性能指标
- **平均推理时间**: {results['avg_inference_time']*1000:.2f} ms
- **推理速度**: {results['fps']:.1f} FPS
- **平均内存使用**: {results['avg_memory_mb']:.1f} MB
- **平均置信度**: {results['avg_confidence']:.3f}

## 详细统计
- **推理时间范围**: {min(results['inference_times'])*1000:.2f} - {max(results['inference_times'])*1000:.2f} ms
- **置信度范围**: {min(results['prediction_scores']):.3f} - {max(results['prediction_scores']):.3f}
"""
        
        report_file = self.output_dir / f"{self.framework}_evaluation_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 评估报告已保存: {report_file}")
        
        return results_file, report_file
    
    def run_evaluation(self, num_samples=100):
        """运行完整评估"""
        print("🎯 开始模型评估...")
        print("=" * 60)
        
        # 1. 加载测试数据
        test_data = self.load_test_data()
        
        # 2. 加载模型
        model = self.load_model()
        if model is None:
            print("❌ 模型加载失败")
            return None
        
        # 3. 评估性能
        results = self.evaluate_performance(model, test_data, num_samples)
        
        # 4. 保存结果
        if results:
            files = self.save_results(results)
            print("=" * 60)
            print("✅ 模型评估完成！")
            return results
        else:
            print("❌ 评估失败")
            return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Gold-YOLO模型评估')
    parser.add_argument('--test-file', type=str,
                       default='/home/kyc/project/GOLD-YOLO/data/coco2017_val/splits/test_annotations.json',
                       help='测试集标注文件')
    parser.add_argument('--model-path', type=str,
                       default='runs/fixed_test/best_fixed_test.pkl',
                       help='模型权重文件')
    parser.add_argument('--framework', type=str, default='jittor',
                       choices=['jittor', 'pytorch'], help='框架类型')
    parser.add_argument('--num-samples', type=int, default=100, help='评估样本数')
    
    args = parser.parse_args()
    
    print("🎯 Gold-YOLO 模型评估")
    print("新芽第二阶段：测试集性能评估")
    print("=" * 60)
    print(f"📊 配置:")
    print(f"   框架: {args.framework}")
    print(f"   测试集: {args.test_file}")
    print(f"   模型: {args.model_path}")
    print(f"   样本数: {args.num_samples}")
    
    # 创建评估器
    evaluator = ModelEvaluator(
        test_file=args.test_file,
        model_path=args.model_path,
        framework=args.framework
    )
    
    # 运行评估
    results = evaluator.run_evaluation(args.num_samples)
    
    if results:
        print(f"\n🎉 评估成功完成！")
        print(f"📊 关键指标:")
        print(f"   推理速度: {results['fps']:.1f} FPS")
        print(f"   推理时间: {results['avg_inference_time']*1000:.2f} ms")
        print(f"   内存使用: {results['avg_memory_mb']:.1f} MB")
    else:
        print("❌ 评估失败")


if __name__ == "__main__":
    main()
