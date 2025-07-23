#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GDNeck - 严格对齐PyTorch版本的M版本Neck
新芽第二阶段：深入修复架构不一致问题
"""

import jittor as jt
import jittor.nn as nn
from ..layers.common import *
from ..layers.transformer import *


class GDNeck(nn.Module):
    """
    GDNeck - 用于Gold-YOLO M版本
    严格对齐PyTorch版本的GDNeck实现
    """
    
    def __init__(self, channels_list, num_repeats, block=RepVGGBlock, csp_e=2/3, extra_cfg=None):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        assert extra_cfg is not None
        
        self.channels_list = channels_list
        self.csp_e = csp_e
        
        # 严格对齐PyTorch版本的配置参数
        self.trans_channels = extra_cfg.get('trans_channels', [192, 96, 192, 384])
        self.embed_dim_p = extra_cfg.get('embed_dim_p', 192)
        self.embed_dim_n = extra_cfg.get('embed_dim_n', 1056)
        self.fusion_in = extra_cfg.get('fusion_in', 1440)
        self.fuse_block_num = extra_cfg.get('fuse_block_num', 3)
        self.depths = extra_cfg.get('depths', 2)
        self.num_heads = extra_cfg.get('num_heads', 4)
        
        print(f"🔧 构建GDNeck: fusion_in={self.fusion_in}, embed_dim_p={self.embed_dim_p}, embed_dim_n={self.embed_dim_n}")
        print(f"   trans_channels={self.trans_channels}, csp_e={self.csp_e}")
        
        # Backbone通道数
        backbone_channels = channels_list[:5] if len(channels_list) >= 5 else channels_list
        
        # === Low-GD (低级全局分布) ===
        self.low_FAM = SimFusion_4in(
            in_channel_list=backbone_channels[1:5],
            out_channels=self.embed_dim_p
        )
        
        self.low_IFM = nn.Sequential(
            Conv(self.embed_dim_p, self.embed_dim_p, kernel_size=1, stride=1, padding=0),
            *[RepVGGBlock(self.embed_dim_p, self.embed_dim_p) for _ in range(self.fuse_block_num)],
            Conv(self.embed_dim_p, sum(self.trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        
        # === High-GD (高级全局分布) ===
        self.high_FAM = SimFusion_3in(
            in_channel_list=backbone_channels[2:5],
            out_channels=self.embed_dim_n
        )
        
        self.high_IFM = nn.Sequential(
            Conv(self.embed_dim_n, self.embed_dim_n, kernel_size=1, stride=1, padding=0),
            *[RepVGGBlock(self.embed_dim_n, self.embed_dim_n) for _ in range(self.fuse_block_num)],
            Conv(self.embed_dim_n, sum(self.trans_channels[2:4]), kernel_size=1, stride=1, padding=0),
        )
        
        # === LAF (局部聚合融合) ===
        self.reduce_layer_c5 = Conv(backbone_channels[4], backbone_channels[3], 1, 1, 0)
        self.reduce_layer_p4 = Conv(backbone_channels[3], backbone_channels[2], 1, 1, 0)
        
        self.LAF_p4 = SimFusion_3in(
            in_channel_list=[backbone_channels[2], backbone_channels[3], backbone_channels[3]],
            out_channels=backbone_channels[3]
        )
        
        self.LAF_p3 = SimFusion_3in(
            in_channel_list=[backbone_channels[1], backbone_channels[2], backbone_channels[2]],
            out_channels=backbone_channels[2]
        )
        
        # === Inject (注入) ===
        self.Inject_p4 = InjectionMultiSum_Auto_pool(
            sum(self.trans_channels[0:2]), backbone_channels[3],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )
        
        self.Inject_p3 = InjectionMultiSum_Auto_pool(
            sum(self.trans_channels[0:2]), backbone_channels[2],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )
        
        # === 下采样和上采样 ===
        self.downsample_p3 = SimConv(backbone_channels[2], backbone_channels[2], 3, 2)
        self.downsample_n4 = SimConv(backbone_channels[3], backbone_channels[3], 3, 2)
        
        self.LAF_n4 = AdvPoolFusion()
        self.LAF_n5 = AdvPoolFusion()
        
        self.Inject_n4 = InjectionMultiSum_Auto_pool(
            backbone_channels[3] + backbone_channels[2], backbone_channels[3],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )
        
        self.Inject_n5 = InjectionMultiSum_Auto_pool(
            backbone_channels[4] + backbone_channels[3], backbone_channels[4],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )
        
        # === RepBlocks ===
        self.Rep_p4 = RepBlock(backbone_channels[3], backbone_channels[3], num_repeats[0], block)
        self.Rep_p3 = RepBlock(backbone_channels[2], backbone_channels[2], num_repeats[1], block)
        self.Rep_n4 = RepBlock(backbone_channels[3], backbone_channels[3], num_repeats[2], block)
        self.Rep_n5 = RepBlock(backbone_channels[4], backbone_channels[4], num_repeats[3], block)
        
    def execute(self, inputs):
        """前向传播"""
        c1, c2, c3, c4, c5 = inputs
        
        # === Low-GD ===
        low_fuse_feat = self.low_FAM([c2, c3, c4, c5])
        low_global_info = list(jt.split(self.low_IFM(low_fuse_feat), self.trans_channels[0:2], dim=1))
        
        # === High-GD ===
        high_fuse_feat = self.high_FAM([c3, c4, c5])
        high_global_info = list(jt.split(self.high_IFM(high_fuse_feat), self.trans_channels[2:4], dim=1))
        
        # === LAF ===
        c5_half = self.reduce_layer_c5(c5)
        p4 = self.LAF_p4([c3, c4, c5_half])
        p4 = self.Inject_p4(low_global_info[0], p4)
        p4 = self.Rep_p4(p4)
        
        p4_half = self.reduce_layer_p4(p4)
        p3 = self.LAF_p3([c2, c3, p4_half])
        p3 = self.Inject_p3(low_global_info[1], p3)
        p3 = self.Rep_p3(p3)
        
        # === 下采样路径 ===
        p3_downsampled = self.downsample_p3(p3)
        n4_adjacent_info = self.LAF_n4([p4, p3_downsampled])
        n4 = self.Inject_n4(high_global_info[0], n4_adjacent_info)
        n4 = self.Rep_n4(n4)
        
        n4_downsampled = self.downsample_n4(n4)
        n5_adjacent_info = self.LAF_n5([c5, n4_downsampled])
        n5 = self.Inject_n5(high_global_info[1], n5_adjacent_info)
        n5 = self.Rep_n5(n5)
        
        return [p3, n4, n5]


class GDNeck2(GDNeck):
    """
    GDNeck2 - 用于Gold-YOLO L版本
    继承GDNeck，增加更多的深度和头数
    """
    
    def __init__(self, channels_list, num_repeats, block=RepVGGBlock, csp_e=0.5, extra_cfg=None):
        # 修改L版本的特殊配置
        if extra_cfg:
            extra_cfg['depths'] = 3  # L版本使用更深的网络
            extra_cfg['num_heads'] = 8  # L版本使用更多的注意力头
            
        super().__init__(channels_list, num_repeats, block, csp_e, extra_cfg)
        
        print(f"🔧 构建GDNeck2 (L版本): depths={self.depths}, num_heads={self.num_heads}")
