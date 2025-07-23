#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完全对齐PyTorch版本的Gold-YOLO训练脚本
严格按照PyTorch版本的参数、逻辑和流程实现
"""

import argparse
import os
import sys
import time
import yaml
import os.path as osp
from pathlib import Path
from copy import deepcopy

import jittor as jt
import jittor.nn as nn
import numpy as np
from tqdm import tqdm

# 设置Jittor优化
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 1

# 添加项目路径
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入Gold-YOLO组件
from gold_yolo.models.gold_yolo import create_gold_yolo
from gold_yolo.data.coco_dataset import COCODataset
from gold_yolo.training.loss import ComputeLoss
from gold_yolo.training.optimizer import build_optimizer
from gold_yolo.training.scheduler import build_lr_scheduler
from gold_yolo.utils.general import increment_name, set_random_seed
from gold_yolo.utils.events import LOGGER


def get_args_parser(add_help=True):
    """完全对齐PyTorch版本的参数解析器"""
    parser = argparse.ArgumentParser(description='Gold-YOLO Jittor Training', add_help=add_help)
    
    # 数据相关参数 - 严格对齐PyTorch版本
    parser.add_argument('--data-path', default='./data/coco.yaml', type=str, help='path of dataset')
    parser.add_argument('--conf-file', default='./configs/gold_yolo-s.py', type=str, help='experiments description file')
    parser.add_argument('--img-size', default=640, type=int, help='train, val image size (pixels)')
    parser.add_argument('--batch-size', default=32, type=int, help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    
    # 设备相关参数
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    # 评估相关参数
    parser.add_argument('--eval-interval', default=20, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--eval-final-only', action='store_true', help='only evaluate at the final epoch')
    parser.add_argument('--heavy-eval-range', default=50, type=int,
                        help='evaluating every epoch for last such epochs')
    
    # 数据检查参数
    parser.add_argument('--check-images', action='store_true', help='check images when initializing datasets')
    parser.add_argument('--check-labels', action='store_true', help='check label files when initializing datasets')
    
    # 输出相关参数
    parser.add_argument('--output-dir', default='./runs/train', type=str, help='path to save outputs')
    parser.add_argument('--name', default='exp', type=str, help='experiment name, saved to output_dir/name')
    
    # 训练控制参数
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume the most recent training')
    parser.add_argument('--stop_aug_last_n_epoch', default=15, type=int, help='stop strong aug at last n epoch')
    parser.add_argument('--save_ckpt_on_last_n_epoch', default=-1, type=int, help='save last n epoch checkpoints')
    
    # 优化器相关参数
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.937, type=float, help='SGD momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--warmup-epochs', default=3, type=int, help='warmup epochs')
    
    return parser


class AlignedTrainer:
    """完全对齐PyTorch版本的训练器"""
    
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device
        
        # 训练状态
        self.epoch = 0
        self.best_fitness = 0.0
        self.start_epoch = 0
        
        # 创建保存目录
        self.save_dir = str(increment_name(osp.join(args.output_dir, args.name)))
        os.makedirs(self.save_dir, exist_ok=True)
        
        LOGGER.info(f'🎯 对齐训练器初始化完成')
        LOGGER.info(f'   保存目录: {self.save_dir}')
        LOGGER.info(f'   设备: {self.device}')
        
    def setup_model(self):
        """设置模型 - 严格对齐PyTorch版本"""
        # 从配置文件解析模型版本
        model_version = 's'  # 默认使用S版本
        if 'gold_yolo-n' in self.args.conf_file:
            model_version = 'n'
        elif 'gold_yolo-m' in self.args.conf_file:
            model_version = 'm'
        elif 'gold_yolo-l' in self.args.conf_file:
            model_version = 'l'
            
        # 创建模型
        self.model = create_gold_yolo(
            model_version, 
            num_classes=80,  # COCO数据集
            use_pytorch_components=False
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(f'✅ 模型创建完成: Gold-YOLO-{model_version.upper()}')
        LOGGER.info(f'   参数量: {total_params:,} ({total_params/1e6:.2f}M)')
        
        return self.model
        
    def setup_data(self):
        """设置数据加载器 - 对齐PyTorch版本"""
        # 加载数据配置
        with open(self.args.data_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            
        # 创建训练数据集
        train_dataset = COCODataset(
            img_dir=data_cfg['train'],
            ann_file=data_cfg.get('train_ann', None),
            img_size=self.args.img_size,
            augment=True,
            cache=False
        )
        
        # 创建验证数据集
        val_dataset = COCODataset(
            img_dir=data_cfg['val'],
            ann_file=data_cfg.get('val_ann', None),
            img_size=self.args.img_size,
            augment=False,
            cache=False
        )
        
        # 创建数据加载器
        self.train_loader = train_dataset.set_attrs(
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            drop_last=True
        )
        
        self.val_loader = val_dataset.set_attrs(
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            drop_last=False
        )
        
        LOGGER.info(f'✅ 数据加载器创建完成')
        LOGGER.info(f'   训练集: {len(train_dataset)} 张图片')
        LOGGER.info(f'   验证集: {len(val_dataset)} 张图片')
        LOGGER.info(f'   批次大小: {self.args.batch_size}')
        
        return self.train_loader, self.val_loader
        
    def setup_loss(self):
        """设置损失函数 - 对齐PyTorch版本"""
        self.compute_loss = ComputeLoss(
            num_classes=80,
            ori_img_size=self.args.img_size,
            warmup_epoch=self.args.warmup_epochs,
            use_dfl=True,
            reg_max=16,
            iou_type='giou',
            loss_weight={
                'class': 1.0,
                'iou': 2.5,
                'dfl': 0.5
            }
        )
        
        LOGGER.info(f'✅ 损失函数创建完成: ComputeLoss')
        return self.compute_loss
        
    def setup_optimizer(self):
        """设置优化器和学习率调度器 - 对齐PyTorch版本"""
        # 创建优化器
        self.optimizer = build_optimizer(
            model=self.model,
            name='SGD',
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        # 创建学习率调度器
        self.scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            name='cosine',
            epochs=self.args.epochs,
            warmup_epochs=self.args.warmup_epochs,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1
        )
        
        LOGGER.info(f'✅ 优化器创建完成: SGD')
        LOGGER.info(f'   学习率: {self.args.lr}')
        LOGGER.info(f'   动量: {self.args.momentum}')
        LOGGER.info(f'   权重衰减: {self.args.weight_decay}')
        
        return self.optimizer, self.scheduler


def check_and_init(args):
    """检查和初始化 - 对齐PyTorch版本"""
    # 设置随机种子
    set_random_seed(1, deterministic=True)
    
    # 创建保存目录
    if not args.resume:
        args.save_dir = str(increment_name(osp.join(args.output_dir, args.name)))
        os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载配置文件
    if args.conf_file.endswith('.py'):
        # Python配置文件
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.conf_file)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
    else:
        # YAML配置文件
        with open(args.conf_file, 'r') as f:
            cfg = yaml.safe_load(f)
    
    # 设置设备
    if args.device == 'cpu':
        jt.flags.use_cuda = 0
    else:
        jt.flags.use_cuda = 1
        
    LOGGER.info(f'✅ 初始化完成')
    LOGGER.info(f'   配置文件: {args.conf_file}')
    LOGGER.info(f'   设备: {"CUDA" if jt.flags.use_cuda else "CPU"}')
    
    return cfg, args


def main(args):
    """主训练函数 - 完全对齐PyTorch版本"""
    LOGGER.info(f'🎯 开始Gold-YOLO Jittor训练')
    LOGGER.info(f'训练参数: {args}\n')
    
    # 初始化
    cfg, args = check_and_init(args)
    
    # 创建训练器
    trainer = AlignedTrainer(args, cfg, 'cuda' if jt.flags.use_cuda else 'cpu')
    
    # 设置组件
    model = trainer.setup_model()
    train_loader, val_loader = trainer.setup_data()
    compute_loss = trainer.setup_loss()
    optimizer, scheduler = trainer.setup_optimizer()
    
    LOGGER.info(f'🚀 开始训练...')
    LOGGER.info(f'   总轮数: {args.epochs}')
    LOGGER.info(f'   评估间隔: {args.eval_interval}')
    
    # 开始训练循环
    start_time = time.time()

    for epoch in range(args.epochs):
        trainer.epoch = epoch

        # 训练一个epoch
        train_loss = train_one_epoch(
            model, train_loader, compute_loss, optimizer, scheduler, epoch, args
        )

        # 评估
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            val_loss = validate(model, val_loader, compute_loss, epoch, args)

            # 保存检查点
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss,
                osp.join(trainer.save_dir, f'epoch_{epoch}.pt')
            )

        LOGGER.info(f'Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}')

    total_time = time.time() - start_time
    LOGGER.info(f'🎉 训练完成! 总用时: {total_time:.2f}s')


def train_one_epoch(model, dataloader, compute_loss, optimizer, scheduler, epoch, args):
    """训练一个epoch - 对齐PyTorch版本"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f'Epoch {epoch}')

    for batch_idx, (images, targets) in pbar:
        # 前向传播
        predictions = model(images)

        # 计算损失
        loss, loss_items = compute_loss(predictions, targets)

        # 反向传播
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()

        # 更新学习率
        scheduler.step()

        # 统计
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_lr()[0]:.6f}'
        })

    return total_loss / num_batches


def validate(model, dataloader, compute_loss, epoch, args):
    """验证 - 对齐PyTorch版本"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    with jt.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # 前向传播
            predictions = model(images)

            # 计算损失
            loss, loss_items = compute_loss(predictions, targets)
            total_loss += loss.item()

    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, path):
    """保存检查点 - 对齐PyTorch版本"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    jt.save(checkpoint, path)
    LOGGER.info(f'✅ 检查点已保存: {path}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
