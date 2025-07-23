#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def analyze_training_failure():
    """深度分析训练失效的原因"""
    print('🔍 深度分析Gold-YOLO-n训练失效原因')
    print('=' * 80)
    
    # 1. 训练配置分析
    print('📊 1. 训练配置分析:')
    model_dir = Path('runs/train/gold_yolo_n_voc_rtx40604')
    
    if model_dir.exists():
        print(f'   模型目录: {model_dir}')
        
        # 检查训练参数
        best_model = model_dir / 'weights/best_ckpt.pt'
        if best_model.exists():
            try:
                checkpoint = torch.load(best_model, map_location='cpu')
                print(f'   训练轮数: {checkpoint.get("epoch", "未知")}')
                print(f'   最佳fitness: {checkpoint.get("best_fitness", "未知")}')
                
                # 检查优化器状态
                if 'optimizer' in checkpoint:
                    optimizer_state = checkpoint['optimizer']
                    print(f'   优化器状态: 已保存')
                    if 'param_groups' in optimizer_state:
                        lr = optimizer_state['param_groups'][0].get('lr', '未知')
                        print(f'   学习率: {lr}')
                
            except Exception as e:
                print(f'   模型加载错误: {e}')
    
    # 2. 数据集配置分析
    print(f'\n📋 2. 数据集配置分析:')
    try:
        with open('data/voc_subset.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f'   类别数: {data_config["nc"]}')
        print(f'   训练路径: {data_config["train"]}')
        print(f'   验证路径: {data_config["val"]}')
        
        # 检查路径是否正确
        train_path = Path(data_config["train"])
        print(f'   训练路径存在: {train_path.exists()}')
        if train_path.exists():
            img_count = len(list(train_path.glob('*.jpg')))
            print(f'   训练图片数量: {img_count}')
        
    except Exception as e:
        print(f'   配置文件读取错误: {e}')
    
    # 3. 数据分布分析
    print(f'\n📈 3. 数据分布分析:')
    labels_dir = Path('/home/kyc/project/GOLD-YOLO/data/voc2012_subset/labels')
    
    if labels_dir.exists():
        class_counts = {}
        total_objects = 0
        bbox_sizes = []
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            bbox_sizes.append(width * height)
                            total_objects += 1
        
        print(f'   总目标数: {total_objects}')
        print(f'   平均目标大小: {np.mean(bbox_sizes):.4f}')
        print(f'   目标大小标准差: {np.std(bbox_sizes):.4f}')
        
        # 类别不平衡分析
        print(f'\n   类别分布 (前10个):')
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (class_id, count) in enumerate(sorted_classes[:10]):
            percentage = count / total_objects * 100
            try:
                class_name = data_config['names'][class_id]
            except:
                class_name = f'class_{class_id}'
            print(f'   {class_id:2d} ({class_name:12s}): {count:4d} ({percentage:5.1f}%)')
        
        # 数据不平衡程度
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count
        print(f'\n   数据不平衡比例: {imbalance_ratio:.1f}:1')
        
    # 4. 模型架构分析
    print(f'\n🏗️ 4. 模型架构分析:')
    try:
        from yolov6.models.yolo import build_model
        from yolov6.utils.config import Config
        
        # 加载配置
        cfg = Config.fromfile('configs/gold_yolo-n.py')
        print(f'   模型配置文件: configs/gold_yolo-n.py')
        print(f'   模型深度倍数: {cfg.model.depth_multiple}')
        print(f'   模型宽度倍数: {cfg.model.width_multiple}')
        
        # 检查类别数是否匹配
        if hasattr(cfg.model, 'head') and hasattr(cfg.model.head, 'nc'):
            model_nc = cfg.model.head.nc
            data_nc = data_config.get('nc', 20)
            print(f'   模型类别数: {model_nc}')
            print(f'   数据类别数: {data_nc}')
            print(f'   类别数匹配: {model_nc == data_nc}')
        
    except Exception as e:
        print(f'   模型配置分析错误: {e}')
    
    # 5. 训练超参数分析
    print(f'\n⚙️ 5. 训练超参数分析:')
    print('   批次大小: 8 (可能过小)')
    print('   训练轮数: 49 (明显不足)')
    print('   图像尺寸: 640x640')
    print('   工作进程: 2')
    print('   评估间隔: 5轮')
    
    # 6. 问题总结
    print(f'\n❌ 6. 识别的主要问题:')
    problems = [
        '训练轮数严重不足 (49轮 vs 推荐200+轮)',
        '数据集规模较小 (964张图片)',
        '可能存在严重的类别不平衡',
        '批次大小过小 (8 vs 推荐16-32)',
        '训练时间过短 (23分钟)',
        '可能的学习率设置不当',
        '缺少数据增强策略',
        '没有使用预训练权重'
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f'   {i}. {problem}')
    
    # 7. 改进建议
    print(f'\n💡 7. 具体改进建议:')
    improvements = [
        '增加训练轮数到200-300轮',
        '使用COCO预训练权重初始化',
        '调整学习率策略 (warmup + cosine decay)',
        '增加数据增强 (mosaic, mixup, cutmix)',
        '使用更大的批次大小 (16-32)',
        '添加类别权重平衡损失',
        '使用更大的数据集或数据增强',
        '调整anchor设置和损失函数权重'
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f'   {i}. {improvement}')
    
    return class_counts, bbox_sizes

def create_data_analysis_plots(class_counts, bbox_sizes):
    """创建数据分析图表"""
    print(f'\n📊 生成数据分析图表...')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 类别分布
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    ax1.bar(classes, counts)
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Number of Objects')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 目标大小分布
    ax2.hist(bbox_sizes, bins=50, alpha=0.7)
    ax2.set_title('Object Size Distribution')
    ax2.set_xlabel('Normalized Area (width × height)')
    ax2.set_ylabel('Frequency')
    
    # 3. 类别不平衡可视化
    sorted_counts = sorted(counts, reverse=True)
    ax3.plot(range(len(sorted_counts)), sorted_counts, 'o-')
    ax3.set_title('Class Imbalance')
    ax3.set_xlabel('Class Rank')
    ax3.set_ylabel('Number of Objects')
    ax3.set_yscale('log')
    
    # 4. 累积分布
    cumsum = np.cumsum(sorted_counts)
    ax4.plot(range(len(cumsum)), cumsum / cumsum[-1] * 100, 'o-')
    ax4.set_title('Cumulative Distribution')
    ax4.set_xlabel('Class Rank')
    ax4.set_ylabel('Cumulative Percentage (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'runs/train/gold_yolo_n_voc_rtx40604/data_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'   数据分析图表保存到: {output_file}')

if __name__ == '__main__':
    class_counts, bbox_sizes = analyze_training_failure()
    if class_counts and bbox_sizes:
        create_data_analysis_plots(class_counts, bbox_sizes)
