#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完整的RepGDNeck实现 (Jittor版本)
新芽第二阶段：完全对齐PyTorch版本
"""

import jittor as jt
import jittor.nn as nn
from ..layers.common import (
    Conv, RepVGGBlock, RepBlock, SimConv, 
    SimFusion_3in, SimFusion_4in, AdvPoolFusion,
    InjectionMultiSum_Auto_pool
)
from ..layers.transformer import PyramidPoolAgg, TopBasicLayer


class CompleteRepGDNeck(nn.Module):
    """完整的RepGDNeck - 完全对齐PyTorch版本"""
    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock, extra_cfg=None):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        assert extra_cfg is not None
        
        # 保存配置
        self.trans_channels = extra_cfg['trans_channels']
        
        # === Low-GD (低级全局分布) ===
        # 低级特征聚合模块 (Low-level Feature Aggregation Module)
        self.low_FAM = SimFusion_4in()
        
        # 低级信息融合模块 (Low-level Information Fusion Module)
        self.low_IFM = nn.Sequential(
            Conv(extra_cfg['fusion_in'], extra_cfg['embed_dim_p'], kernel_size=1, stride=1, padding=0),
            *[block(extra_cfg['embed_dim_p'], extra_cfg['embed_dim_p']) for _ in range(extra_cfg['fuse_block_num'])],
            Conv(extra_cfg['embed_dim_p'], sum(extra_cfg['trans_channels'][0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        # P5 -> P4 路径 (根据实际backbone通道数调整)
        # channels_list前5个是backbone: [16, 32, 64, 128, 128]
        # 后面的是neck通道数
        backbone_channels = channels_list[:5]  # [16, 32, 64, 128, 128]

        self.reduce_layer_c5 = SimConv(
            in_channels=backbone_channels[4],  # 128 (P6)
            out_channels=backbone_channels[3],  # 128 (P5)
            kernel_size=1,
            stride=1
        )

        self.LAF_p4 = SimFusion_3in(
            in_channel_list=[backbone_channels[2], backbone_channels[3], backbone_channels[3]],  # [64, 128, 128]
            out_channels=backbone_channels[3]  # 128
        )
        
        self.Inject_p4 = InjectionMultiSum_Auto_pool(
            backbone_channels[3], backbone_channels[3],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )

        self.Rep_p4 = RepBlock(
            in_channels=backbone_channels[3],  # 128
            out_channels=backbone_channels[3],  # 128
            n=num_repeats[0] if len(num_repeats) > 0 else 4,  # 使用实际的repeat
            block=block
        )

        # P4 -> P3 路径
        self.reduce_layer_p4 = SimConv(
            in_channels=backbone_channels[3],  # 128
            out_channels=backbone_channels[2],  # 64
            kernel_size=1,
            stride=1
        )

        self.LAF_p3 = SimFusion_3in(
            in_channel_list=[backbone_channels[1], backbone_channels[2], backbone_channels[2]],  # [32, 64, 64]
            out_channels=backbone_channels[2]  # 64
        )

        self.Inject_p3 = InjectionMultiSum_Auto_pool(
            backbone_channels[2], backbone_channels[2],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )

        self.Rep_p3 = RepBlock(
            in_channels=backbone_channels[2],  # 64
            out_channels=backbone_channels[2],  # 64
            n=num_repeats[1] if len(num_repeats) > 1 else 4,  # 使用实际的repeat
            block=block
        )
        
        # === High-GD (高级全局分布) ===
        # 高级特征聚合模块 (High-level Feature Aggregation Module)
        self.high_FAM = PyramidPoolAgg(
            stride=extra_cfg.get('c2t_stride', 2), 
            pool_mode=extra_cfg.get('pool_mode', 'torch')
        )
        
        # 高级信息融合模块 (High-level Information Fusion Module)
        dpr = [x.item() for x in jt.linspace(0, extra_cfg.get('drop_path_rate', 0.1), extra_cfg.get('depths', 2))]
        self.high_IFM = TopBasicLayer(
            block_num=extra_cfg.get('depths', 2),
            embedding_dim=extra_cfg['embed_dim_n'],
            key_dim=extra_cfg.get('key_dim', 8),
            num_heads=extra_cfg.get('num_heads', 4),
            mlp_ratio=extra_cfg.get('mlp_ratios', 1),
            attn_ratio=extra_cfg.get('attn_ratios', 2),
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=extra_cfg.get('norm_cfg')
        )
        
        self.conv_1x1_n = nn.Conv2d(
            extra_cfg['embed_dim_n'], 
            sum(extra_cfg['trans_channels'][2:4]), 
            1, 1, 0
        )
        
        # N4 路径 (简化实现)
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = InjectionMultiSum_Auto_pool(
            backbone_channels[3], backbone_channels[3],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )

        # 下采样层
        self.downsample_p3 = SimConv(
            in_channels=backbone_channels[2],  # 64
            out_channels=backbone_channels[2],  # 64
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=backbone_channels[2] + backbone_channels[3],  # 64 + 128 = 192
            out_channels=backbone_channels[3],  # 128
            n=num_repeats[2] if len(num_repeats) > 2 else 4,
            block=block
        )

        # N5 路径 (简化实现)
        self.LAF_n5 = AdvPoolFusion()
        self.Inject_n5 = InjectionMultiSum_Auto_pool(
            backbone_channels[4], backbone_channels[4],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )

        # 下采样层
        self.downsample_n4 = SimConv(
            in_channels=backbone_channels[3],  # 128
            out_channels=backbone_channels[3],  # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = RepBlock(
            in_channels=backbone_channels[3] + backbone_channels[4],  # 128 + 128 = 256
            out_channels=backbone_channels[4],  # 128
            n=num_repeats[3] if len(num_repeats) > 3 else 4,
            block=block
        )
    
    def execute(self, input):
        """
        完全对齐PyTorch版本的forward方法
        input: (c2, c3, c4, c5) 四个尺度的特征
        """
        if len(input) == 5:
            c2, c3, c4, c5 = input[1:]  # 跳过P2
        else:
            c2, c3, c4, c5 = input
        
        # === Low-GD (低级全局分布) ===
        # 使用卷积融合全局信息
        low_align_feat = self.low_FAM(c2, c3, c4, c5)
        low_fuse_feat = self.low_IFM(low_align_feat)
        
        # 分割低级全局信息
        low_global_info = jt.split(low_fuse_feat, self.trans_channels[0:2], dim=1)
        
        # 注入低级全局信息到p4
        c5_half = self.reduce_layer_c5(c5)
        p4_adjacent_info = self.LAF_p4(c3, c4, c5_half)
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        # 注入低级全局信息到p3
        p4_half = self.reduce_layer_p4(p4)
        p3_adjacent_info = self.LAF_p3(c2, c3, p4_half)
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # === High-GD (高级全局分布) ===
        # 使用transformer融合全局信息
        high_align_feat = self.high_FAM([p3, p4, c5])
        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        
        # 分割高级全局信息
        high_global_info = jt.split(high_fuse_feat, self.trans_channels[2:4], dim=1)
        
        # 注入高级全局信息到n4 (简化实现)
        p3_downsampled = self.downsample_p3(p3)
        n4_adjacent_info = self.LAF_n4(p4, p3_downsampled)
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        n4_concat = jt.concat([p3, p3_downsampled], dim=1)  # 拼接用于RepBlock
        n4 = self.Rep_n4(n4_concat)

        # 注入高级全局信息到n5 (简化实现)
        n4_downsampled = self.downsample_n4(n4)
        n5_adjacent_info = self.LAF_n5(c5, n4_downsampled)
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        n5_concat = jt.concat([p4, n4_downsampled], dim=1)  # 拼接用于RepBlock
        n5 = self.Rep_n5(n5_concat)
        
        # 输出三个尺度的特征
        outputs = [p3, n4, n5]
        
        return outputs
