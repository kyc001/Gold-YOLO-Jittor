#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gold-YOLO PyTorch基准训练
新芽第二阶段：建立PyTorch训练基准
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
import torch
import torch.distributed as dist

# 添加项目路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.general import increment_name, find_latest_checkpoint

def create_coco_annotations():
    """创建COCO格式的标注文件"""
    import json
    from pathlib import Path
    
    # 数据路径
    data_root = Path("/home/kyc/project/GOLD-YOLO/data/coco2017_val")
    images_dir = data_root / "images"
    splits_dir = data_root / "splits"
    
    # 检查是否已有标注文件
    train_ann_file = splits_dir / "instances_train.json"
    val_ann_file = splits_dir / "instances_val.json"
    
    if train_ann_file.exists() and val_ann_file.exists():
        print(f"✅ 标注文件已存在")
        return str(train_ann_file), str(val_ann_file)
    
    # 读取现有的分割标注
    train_split_file = splits_dir / "train_annotations.json"
    val_split_file = splits_dir / "test_annotations.json"
    
    if not train_split_file.exists() or not val_split_file.exists():
        print(f"❌ 找不到分割标注文件")
        return None, None
    
    # 转换为COCO格式
    def convert_to_coco_format(split_file, output_file):
        with open(split_file, 'r') as f:
            data = json.load(f)
        
        # 创建COCO格式
        coco_data = {
            "images": data.get("images", []),
            "annotations": data.get("annotations", []),
            "categories": data.get("categories", [])
        }
        
        # 确保目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(coco_data, f)
        
        print(f"✅ 转换完成: {output_file}")
        return str(output_file)
    
    train_ann = convert_to_coco_format(train_split_file, train_ann_file)
    val_ann = convert_to_coco_format(val_split_file, val_ann_file)
    
    return train_ann, val_ann

def update_data_config():
    """更新数据配置文件"""
    train_ann, val_ann = create_coco_annotations()
    
    if not train_ann or not val_ann:
        print(f"❌ 无法创建标注文件")
        return None
    
    # 更新YAML配置
    config_file = ROOT / "data" / "coco_small.yaml"
    
    config_data = {
        'path': '/home/kyc/project/GOLD-YOLO/data/coco2017_val',
        'train': 'images',
        'val': 'images',
        'train_ann': train_ann,
        'val_ann': val_ann,
        'nc': 80,
        'names': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    print(f"✅ 数据配置更新: {config_file}")
    return str(config_file)

def main():
    """主训练函数"""
    print("🎯 Gold-YOLO PyTorch基准训练")
    print("新芽第二阶段：建立PyTorch训练基准")
    print("=" * 60)
    
    # 解析参数
    parser = argparse.ArgumentParser(description='Gold-YOLO PyTorch Baseline Training')
    parser.add_argument('--data-path', default=None, type=str, help='path of dataset')
    parser.add_argument('--conf-file', default='./configs/gold_yolo-s.py', type=str, help='config file')
    parser.add_argument('--img-size', default=640, type=int, help='train image size')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size (small for baseline)')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs (small for baseline)')
    parser.add_argument('--workers', default=4, type=int, help='number of workers')
    parser.add_argument('--device', default='0', type=str, help='cuda device')
    parser.add_argument('--eval-interval', default=10, type=int, help='evaluate interval')
    parser.add_argument('--output-dir', default='./runs/train', type=str, help='output directory')
    parser.add_argument('--name', default='gold_yolo_s_baseline', type=str, help='experiment name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    
    args = parser.parse_args()
    
    # 更新数据配置
    if args.data_path is None:
        args.data_path = update_data_config()
        if args.data_path is None:
            print("❌ 无法配置数据集")
            return
    
    # 设置设备
    device = select_device(args.device)
    
    # 设置随机种子
    set_random_seed(1, deterministic=True)
    
    # 创建输出目录
    save_dir = Path(args.output_dir) / args.name
    save_dir = increment_name(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    cfg = Config.fromfile(args.conf_file)
    
    # 更新配置
    cfg.model.pretrained = None  # 从头训练
    cfg.solver.lr0 = 0.01  # 学习率
    cfg.solver.epochs = args.epochs
    cfg.solver.warmup_epochs = 3
    
    # 保存配置 - 修复YAML序列化问题
    try:
        # 将Config对象转换为字典
        cfg_dict = {
            'model': dict(cfg.model),
            'solver': dict(cfg.solver),
            'data_aug': dict(cfg.data_aug),
            'use_checkpoint': cfg.use_checkpoint
        }
        save_yaml(cfg_dict, save_dir / 'args.yaml')
    except Exception as e:
        print(f"⚠️ 配置保存失败: {e}")
        # 继续训练，不因为配置保存失败而中断
    
    # 创建训练器
    trainer = Trainer(args, cfg, device)
    
    print(f"🚀 开始PyTorch基准训练")
    print(f"   配置文件: {args.conf_file}")
    print(f"   数据集: {args.data_path}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   训练轮次: {args.epochs}")
    print(f"   输出目录: {save_dir}")
    print(f"   设备: {device}")
    
    try:
        # 开始训练
        trainer.train()
        
        print(f"\n🎉 PyTorch基准训练完成！")
        print(f"💾 模型保存在: {save_dir}")
        print(f"📊 训练日志: {save_dir / 'train_batch0.jpg'}")
        print(f"📈 损失曲线: {save_dir / 'results.png'}")
        
        # 检查最佳模型
        best_model = save_dir / 'weights' / 'best_ckpt.pt'
        last_model = save_dir / 'weights' / 'last_ckpt.pt'
        
        if best_model.exists():
            print(f"✅ 最佳模型: {best_model}")
        if last_model.exists():
            print(f"✅ 最新模型: {last_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print(f"\n🎯 PyTorch基准训练成功！")
        print(f"💡 现在可以用这个基准来对齐Jittor版本")
    else:
        print(f"\n❌ PyTorch基准训练失败！")
        print(f"💡 需要检查配置和数据集")
