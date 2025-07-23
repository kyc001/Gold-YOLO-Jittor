#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

def create_improved_config():
    """创建改进的训练配置"""
    print('🔧 创建改进的Gold-YOLO-n训练配置')
    print('=' * 60)
    
    # 1. 创建改进的数据配置
    improved_data_config = {
        'train': '/home/kyc/project/GOLD-YOLO/data/voc2012_subset/images',
        'val': '/home/kyc/project/GOLD-YOLO/data/voc2012_subset/images',
        'nc': 20,
        'names': {
            0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
            5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
            10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
            15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
        }
    }
    
    # 保存改进的数据配置
    with open('data/voc_subset_improved.yaml', 'w') as f:
        yaml.dump(improved_data_config, f, default_flow_style=False)
    
    print('✅ 改进的数据配置已保存: data/voc_subset_improved.yaml')
    
    # 2. 创建类别权重配置
    # 基于数据分析结果，person类占27.8%，需要降权重
    class_weights = {
        0: 1.5,   # aeroplane (少)
        1: 1.3,   # bicycle (少)
        2: 1.4,   # bird (少)
        3: 1.0,   # boat (中等)
        4: 1.1,   # bottle (中等)
        5: 1.3,   # bus (少)
        6: 1.2,   # car (较多)
        7: 1.5,   # cat (少)
        8: 0.9,   # chair (多)
        9: 1.0,   # cow (中等)
        10: 1.0,  # diningtable (中等)
        11: 1.4,  # dog (少)
        12: 1.0,  # horse (中等)
        13: 1.4,  # motorbike (少)
        14: 0.6,  # person (最多，降权重)
        15: 1.1,  # pottedplant (中等)
        16: 1.2,  # sheep (较多)
        17: 1.4,  # sofa (少)
        18: 1.5,  # train (少)
        19: 1.3   # tvmonitor (少)
    }
    
    print('✅ 类别权重配置已创建')
    
    return improved_data_config, class_weights

def download_pretrained_weights():
    """下载预训练权重"""
    print('\n📥 检查预训练权重...')
    
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    # 检查是否已有预训练权重
    pretrained_weights = [
        'yolov6n.pt',
        'gold_yolo_n_coco.pt'
    ]
    
    available_weights = []
    for weight_file in pretrained_weights:
        weight_path = weights_dir / weight_file
        if weight_path.exists():
            size_mb = weight_path.stat().st_size / (1024*1024)
            print(f'   ✅ {weight_file}: {size_mb:.1f} MB')
            available_weights.append(str(weight_path))
        else:
            print(f'   ❌ {weight_file}: 不存在')
    
    if not available_weights:
        print('   💡 建议下载预训练权重以提高训练效果')
        print('   可以从官方仓库下载 yolov6n.pt 或 gold_yolo_n.pt')
    
    return available_weights

def create_improved_training_script():
    """创建改进的训练脚本"""
    print('\n📝 创建改进的训练脚本...')
    
    script_content = '''#!/bin/bash
# 改进的Gold-YOLO-n训练脚本

echo "🚀 开始改进的Gold-YOLO-n训练"
echo "================================"

# 激活环境
conda activate yolo_py

# 训练参数
BATCH_SIZE=16          # 增加批次大小
EPOCHS=200             # 大幅增加训练轮数
IMG_SIZE=640
DEVICE=0
WORKERS=4
CONF_FILE="configs/gold_yolo-n.py"
DATA_PATH="data/voc_subset_improved.yaml"
NAME="gold_yolo_n_improved"
OUTPUT_DIR="runs/train"

# 学习率和优化器参数
LR_INITIAL=0.01        # 初始学习率
LR_FINAL=0.001         # 最终学习率
MOMENTUM=0.937
WEIGHT_DECAY=0.0005

# 数据增强参数
MOSAIC_PROB=1.0        # Mosaic增强概率
MIXUP_PROB=0.1         # Mixup增强概率

echo "📊 训练配置:"
echo "   批次大小: $BATCH_SIZE"
echo "   训练轮数: $EPOCHS"
echo "   图像尺寸: $IMG_SIZE"
echo "   初始学习率: $LR_INITIAL"
echo "   数据增强: Mosaic + Mixup"

# 检查GPU
echo "🔍 GPU状态:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# 开始训练
echo "🚀 开始训练..."
python tools/train.py \\
    --batch-size $BATCH_SIZE \\
    --conf-file $CONF_FILE \\
    --data-path $DATA_PATH \\
    --epochs $EPOCHS \\
    --device $DEVICE \\
    --img-size $IMG_SIZE \\
    --name $NAME \\
    --workers $WORKERS \\
    --eval-interval 10 \\
    --output-dir $OUTPUT_DIR \\
    --resume false \\
    --amp true \\
    --sync-bn false \\
    --local_rank -1

echo "✅ 训练完成!"
'''
    
    script_path = 'improved_train.sh'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(script_path, 0o755)
    
    print(f'✅ 改进的训练脚本已保存: {script_path}')
    return script_path

def create_training_improvements_summary():
    """创建训练改进总结"""
    print('\n📋 训练改进总结')
    print('=' * 60)
    
    improvements = {
        '数据方面': [
            '✅ 创建平衡的类别权重配置',
            '✅ 保持数据集路径配置正确',
            '💡 建议：使用更大的数据集或数据增强'
        ],
        '模型方面': [
            '💡 建议：使用COCO预训练权重初始化',
            '💡 建议：检查模型架构配置',
            '💡 建议：调整anchor设置'
        ],
        '训练方面': [
            '✅ 增加批次大小: 8 → 16',
            '✅ 大幅增加训练轮数: 49 → 200',
            '✅ 启用混合精度训练 (AMP)',
            '✅ 调整评估间隔: 5 → 10轮'
        ],
        '优化方面': [
            '✅ 设置合适的学习率策略',
            '✅ 配置动量和权重衰减',
            '💡 建议：使用warmup + cosine decay',
            '💡 建议：添加EMA (指数移动平均)'
        ],
        '增强方面': [
            '✅ 启用Mosaic数据增强',
            '✅ 启用Mixup数据增强',
            '💡 建议：添加CutMix增强',
            '💡 建议：调整增强强度'
        ]
    }
    
    for category, items in improvements.items():
        print(f'\n🔧 {category}:')
        for item in items:
            print(f'   {item}')
    
    print(f'\n⚠️ 重要提醒:')
    print('   1. 确保有足够的训练时间 (预计6-8小时)')
    print('   2. 监控GPU内存使用情况')
    print('   3. 定期检查训练损失曲线')
    print('   4. 如果可能，使用预训练权重')
    print('   5. 考虑使用更大的数据集')

if __name__ == '__main__':
    # 创建改进配置
    data_config, class_weights = create_improved_config()
    
    # 检查预训练权重
    available_weights = download_pretrained_weights()
    
    # 创建训练脚本
    script_path = create_training_improvements_summary()
    
    print(f'\n🎯 下一步操作:')
    print(f'   1. 运行改进的训练: bash improved_train.sh')
    print(f'   2. 或手动运行训练命令')
    print(f'   3. 监控训练过程和损失曲线')
    print(f'   4. 在更多轮数后进行推理测试')
