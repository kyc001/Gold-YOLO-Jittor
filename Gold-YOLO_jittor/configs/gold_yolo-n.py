#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gold-YOLO-n Jittor配置文件
新芽第二阶段：与PyTorch版本对齐的配置
"""

# 模型配置
model = dict(
    type='GoldYOLO-n',
    num_classes=20,
    depth_multiple=0.33,
    width_multiple=0.25,
    
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,
        cspsppf=True
    ),
    
    neck=dict(
        type='RepGDNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        extra_cfg=dict(
            norm_cfg=dict(type='BatchNorm2d'),
            depths=2,
            fusion_in=480,
            embed_dim_p=96,
            embed_dim_n=352,
            key_dim=8,
            num_heads=4,
            attn_ratio=2,
            mlp_ratio=2,
            act_cfg=dict(type='ReLU'),
            norm_cfg_transformer=dict(type='LayerNorm')
        )
    ),
    
    head=dict(
        type='EffiDeHead',
        num_layers=3,
        use_dfl=False,
        reg_max=0,
        num_classes=20
    )
)

# 训练配置
training_mode = "conv_silu"

# 数据配置
data_path = "./data/coco.yaml"
num_classes = 20

# 输入配置
input_size = [640, 640]
batch_size = 16

# 优化器配置
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.937,
    weight_decay=0.0005
)

# 学习率调度
lr_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=300,
    eta_min=0.0001
)

# 训练配置
epochs = 300
warmup_epochs = 3
eval_interval = 10
save_interval = 10
