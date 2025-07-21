#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
配置工具类 - 处理Gold-YOLO配置
"""

import os
import sys
from pathlib import Path


class Config:
    """配置类 - 简化版本"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, Config(**value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name):
        # 如果属性不存在，返回None而不是抛出异常
        return None


def load_config(config_path):
    """加载配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 添加配置文件目录到Python路径
    config_dir = config_path.parent
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))
    
    # 导入配置模块
    config_name = config_path.stem
    config_module = __import__(config_name)
    
    # 提取配置
    model_cfg = getattr(config_module, 'model', {})
    solver_cfg = getattr(config_module, 'solver', {})
    data_aug_cfg = getattr(config_module, 'data_aug', {})
    
    return Config(
        model=Config(**model_cfg),
        solver=Config(**solver_cfg),
        data_aug=Config(**data_aug_cfg)
    )


def get_gold_yolo_s_config():
    """获取Gold-YOLO-s的默认配置"""
    
    # Extra配置
    extra_cfg = Config(
        norm_cfg=Config(type='SyncBN', requires_grad=True),
        depths=2,
        fusion_in=960,
        ppa_in=704,
        fusion_act=Config(type='ReLU6'),
        fuse_block_num=3,
        embed_dim_p=128,
        embed_dim_n=704,
        key_dim=8,
        num_heads=4,
        mlp_ratios=1,
        attn_ratios=2,
        c2t_stride=2,
        drop_path_rate=0.1,
        trans_channels=[128, 64, 128, 256],
        pool_mode='torch'
    )
    
    # 模型配置
    model_cfg = Config(
        type='GoldYOLO-s',
        pretrained=None,
        depth_multiple=0.33,
        width_multiple=0.50,
        backbone=Config(
            type='EfficientRep',
            num_repeats=[1, 6, 12, 18, 6],
            out_channels=[64, 128, 256, 512, 1024],
            fuse_P2=True,
            cspsppf=True
        ),
        neck=Config(
            type='RepGDNeck',
            num_repeats=[12, 12, 12, 12],
            out_channels=[256, 128, 128, 256, 256, 512],
            extra_cfg=extra_cfg
        ),
        head=Config(
            type='EffiDeHead',
            in_channels=[128, 256, 512],
            num_layers=3,
            begin_indices=24,
            anchors=3,
            anchors_init=[[10, 13, 19, 19, 33, 23],
                          [30, 61, 59, 59, 59, 119],
                          [116, 90, 185, 185, 373, 326]],
            out_indices=[17, 20, 23],
            strides=[8, 16, 32],
            atss_warmup_epoch=0,
            iou_type='giou',
            use_dfl=True,
            reg_max=16,
            distill_weight=Config(
                class_weight=1.0,
                dfl=1.0,
            ),
        )
    )
    
    # 优化器配置
    solver_cfg = Config(
        optim='SGD',
        lr_scheduler='Cosine',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1
    )
    
    # 数据增强配置
    data_aug_cfg = Config(
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    return Config(
        model=model_cfg,
        solver=solver_cfg,
        data_aug=data_aug_cfg
    )


def apply_scaling_factors(config):
    """应用缩放因子到配置"""
    model_cfg = config.model
    
    # 应用depth_multiple和width_multiple
    if hasattr(model_cfg.backbone, 'out_channels'):
        base_channels = model_cfg.backbone.out_channels
        scaled_channels = [int(ch * model_cfg.width_multiple) for ch in base_channels]
        model_cfg.backbone.out_channels = scaled_channels
    
    if hasattr(model_cfg.backbone, 'num_repeats'):
        base_repeats = model_cfg.backbone.num_repeats
        scaled_repeats = [max(1, int(rep * model_cfg.depth_multiple)) for rep in base_repeats]
        model_cfg.backbone.num_repeats = scaled_repeats
    
    # 更新neck的通道配置
    if hasattr(model_cfg.neck, 'out_channels'):
        # 根据backbone的输出调整neck的输入
        backbone_channels = model_cfg.backbone.out_channels
        neck_channels = [
            backbone_channels[2],  # 256 -> scaled
            backbone_channels[2] // 2,  # 128 -> scaled
            backbone_channels[2] // 2,  # 128 -> scaled
            backbone_channels[2],  # 256 -> scaled
            backbone_channels[2],  # 256 -> scaled
            backbone_channels[3],  # 512 -> scaled
        ]
        model_cfg.neck.out_channels = neck_channels
    
    # 更新head的输入通道
    if hasattr(model_cfg.head, 'in_channels'):
        neck_channels = model_cfg.neck.out_channels
        head_in_channels = [neck_channels[1], neck_channels[3], neck_channels[5]]
        model_cfg.head.in_channels = head_in_channels
    
    return config
