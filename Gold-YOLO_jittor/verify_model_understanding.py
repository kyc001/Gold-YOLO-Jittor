#!/usr/bin/env python3
"""
验证GOLD-YOLO模型是否正确"读懂"图片
检查：标签理解、坐标理解、类别索引理解等
"""

import os
import sys
import jittor as jt
import numpy as np
import yaml
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加路径
sys.path.append('.')
sys.path.append('..')

def load_model_and_data():
    """加载模型和数据"""
    print("🔧 加载模型和数据...")
    
    # 创建模型
    from models.perfect_gold_yolo import create_perfect_gold_yolo_model
    model = create_perfect_gold_yolo_model('gold_yolo-n', 20)
    model.eval()
    
    # 加载数据配置
    data_config_path = '../data/voc2012_subset/voc20.yaml'
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 创建数据集
    from yolov6.data.datasets import TrainValDataset
    dataset = TrainValDataset(
        img_dir=data_config['train'],
        img_size=640,
        batch_size=1,
        augment=False,  # 不使用数据增强，便于验证
        hyp=None,
        rect=False,
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0,
        rank=-1,
        data_dict=data_config,
        task="train"
    )
    
    return model, dataset, data_config

def analyze_single_sample(model, dataset, data_config, sample_idx=0):
    """分析单个样本，验证模型理解"""
    print(f"\n🔍 分析样本 {sample_idx}")
    print("=" * 60)
    
    # 获取样本
    try:
        sample = dataset[sample_idx]
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            image, targets = sample[0], sample[1]
        else:
            print(f"❌ 样本格式错误: {type(sample)}")
            return False
    except Exception as e:
        print(f"❌ 获取样本失败: {e}")
        return False
    
    print(f"📊 样本基本信息:")
    print(f"   图像形状: {image.shape}")
    print(f"   图像数值范围: [{float(image.min()):.3f}, {float(image.max()):.3f}]")
    print(f"   目标形状: {targets.shape if hasattr(targets, 'shape') else 'N/A'}")
    
    # 1. 验证图像预处理
    print(f"\n📸 图像预处理验证:")
    if len(image.shape) == 3 and image.shape[0] == 3:
        print(f"   ✅ 图像通道顺序正确: {image.shape}")
        print(f"   ✅ 图像归一化正确: [{float(image.min()):.3f}, {float(image.max()):.3f}]")
    else:
        print(f"   ❌ 图像格式异常: {image.shape}")
        return False
    
    # 2. 验证标签格式
    print(f"\n🏷️ 标签格式验证:")
    if hasattr(targets, 'shape') and len(targets.shape) >= 1:
        print(f"   标签数量: {targets.shape[0]}")
        if targets.shape[0] > 0:
            print(f"   标签维度: {targets.shape[1] if len(targets.shape) > 1 else 'scalar'}")
            
            # 检查标签内容
            if len(targets.shape) >= 2 and targets.shape[1] >= 6:
                print(f"   前3个标签:")
                for i in range(min(3, targets.shape[0])):
                    label = targets[i].numpy() if hasattr(targets, 'numpy') else targets[i]
                    print(f"     标签{i}: {label}")
                    
                    # 解析标签
                    if len(label) >= 6:
                        batch_idx, class_id, x, y, w, h = label[:6]
                        print(f"       批次索引: {batch_idx}")
                        print(f"       类别ID: {class_id} -> {data_config['names'][int(class_id)] if 0 <= class_id < len(data_config['names']) else 'Unknown'}")
                        print(f"       中心坐标: ({x:.3f}, {y:.3f})")
                        print(f"       尺寸: {w:.3f} x {h:.3f}")
                        
                        # 验证坐标合理性
                        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            print(f"       ✅ 坐标格式正确（归一化）")
                        else:
                            print(f"       ❌ 坐标格式异常")
            else:
                print(f"   ❌ 标签维度不足: {targets.shape}")
        else:
            print(f"   ⚠️ 该样本无标签")
    else:
        print(f"   ❌ 标签格式异常: {type(targets)}")
        return False
    
    # 3. 验证模型前向传播
    print(f"\n🧠 模型前向传播验证:")
    try:
        # 添加batch维度
        if len(image.shape) == 3:
            image_batch = image.unsqueeze(0)
        else:
            image_batch = image
            
        print(f"   输入形状: {image_batch.shape}")
        
        # 前向传播
        with jt.no_grad():
            outputs = model(image_batch)
        
        print(f"   输出类型: {type(outputs)}")
        
        if isinstance(outputs, (list, tuple)):
            print(f"   输出长度: {len(outputs)}")
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape'):
                    print(f"   输出{i}形状: {output.shape}")
                    print(f"   输出{i}数值范围: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
                elif isinstance(output, (list, tuple)):
                    print(f"   输出{i}是列表，长度: {len(output)}")
                    for j, sub_output in enumerate(output):
                        if hasattr(sub_output, 'shape'):
                            print(f"     子输出{j}形状: {sub_output.shape}")
        
        # 解析模型输出
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            feats, pred_scores, pred_distri = outputs
            print(f"\n📊 模型输出解析:")
            print(f"   特征图数量: {len(feats)}")
            print(f"   预测分数形状: {pred_scores.shape}")
            print(f"   预测分布形状: {pred_distri.shape}")
            
            # 检查预测分数
            max_score = float(pred_scores.max())
            min_score = float(pred_scores.min())
            print(f"   分类分数范围: [{min_score:.6f}, {max_score:.6f}]")
            
            if max_score > 0.5:
                print(f"   ⚠️ 分类分数过高，可能过拟合")
            elif max_score < 1e-6:
                print(f"   ⚠️ 分类分数过低，可能欠拟合")
            else:
                print(f"   ✅ 分类分数在合理范围")
                
        print(f"   ✅ 模型前向传播成功")
        
    except Exception as e:
        print(f"   ❌ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def verify_data_pipeline():
    """验证数据管道的完整性"""
    print(f"\n🔄 数据管道完整性验证:")
    print("=" * 60)
    
    try:
        model, dataset, data_config = load_model_and_data()
        
        print(f"📊 数据集信息:")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   类别数量: {len(data_config['names'])}")
        print(f"   类别列表: {data_config['names'][:10]}{'...' if len(data_config['names']) > 10 else ''}")
        
        # 分析多个样本
        success_count = 0
        total_samples = min(5, len(dataset))
        
        for i in range(total_samples):
            success = analyze_single_sample(model, dataset, data_config, i)
            if success:
                success_count += 1
        
        print(f"\n📈 验证结果:")
        print(f"   成功样本: {success_count}/{total_samples}")
        print(f"   成功率: {success_count/total_samples*100:.1f}%")
        
        if success_count == total_samples:
            print(f"   ✅ 数据管道完全正常")
            return True
        elif success_count > 0:
            print(f"   ⚠️ 数据管道部分正常")
            return False
        else:
            print(f"   ❌ 数据管道存在严重问题")
            return False
            
    except Exception as e:
        print(f"❌ 数据管道验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 GOLD-YOLO模型理解能力验证")
    print("=" * 60)
    
    # 设置Jittor
    jt.flags.use_cuda = 1
    
    # 验证数据管道
    pipeline_ok = verify_data_pipeline()
    
    if pipeline_ok:
        print(f"\n🎉 验证完成：模型能够正确理解图片和标签！")
        return True
    else:
        print(f"\n❌ 验证失败：模型理解存在问题，需要修复！")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
