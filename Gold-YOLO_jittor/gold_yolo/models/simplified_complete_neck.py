#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
简化但完整的RepGDNeck实现 (Jittor版本)
新芽第二阶段：先让模型运行起来，再逐步完善
"""

import jittor as jt
import jittor.nn as nn
from ..layers.common import (
    Conv, RepVGGBlock, RepBlock, SimConv, 
    SimFusion_3in, SimFusion_4in, AdvPoolFusion,
    InjectionMultiSum_Auto_pool
)
from ..layers.transformer import PyramidPoolAgg, TopBasicLayer


class SimplifiedCompleteRepGDNeck(nn.Module):
    """简化但完整的RepGDNeck - 逐步完善版本"""
    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock, extra_cfg=None):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        assert extra_cfg is not None
        
        # 保存配置
        self.trans_channels = extra_cfg['trans_channels']
        
        # 获取backbone通道数
        backbone_channels = channels_list[:5]  # [16, 32, 64, 128, 128]
        
        # === Low-GD (低级全局分布) - 简化版 ===
        # 低级特征聚合模块
        self.low_FAM = SimFusion_4in()
        
        # 低级信息融合模块 - 使用实际通道数
        self.low_IFM = nn.Sequential(
            Conv(448, extra_cfg['embed_dim_p'], kernel_size=1, stride=1, padding=0),  # 448 -> 24
            *[RepVGGBlock(extra_cfg['embed_dim_p'], extra_cfg['embed_dim_p']) for _ in range(extra_cfg['fuse_block_num'])],
            Conv(extra_cfg['embed_dim_p'], sum(extra_cfg['trans_channels'][0:2]), kernel_size=1, stride=1, padding=0),  # 24 -> 24
        )
        
        # === 自顶向下路径 ===
        # P6 -> P5
        self.reduce_layer_c5 = SimConv(
            in_channels=backbone_channels[4],  # 128
            out_channels=backbone_channels[3],  # 128
            kernel_size=1,
            stride=1
        )
        
        # 通道匹配层 - 根据实际backbone输出调整
        # backbone_channels = [16, 32, 64, 128, 128] 对应 [P2, P3, P4, P5, P6]
        # 但实际使用的是 c2=P3(64), c3=P4(128), c4=P5(128), c5=P6(128)
        self.p5_match = SimConv(backbone_channels[4], backbone_channels[3], 1, 1)  # 128->128
        self.p4_match = SimConv(backbone_channels[3], backbone_channels[3], 1, 1)  # 128->128
        self.p3_match = SimConv(backbone_channels[2], backbone_channels[3], 1, 1)  # 64->128
        
        self.LAF_p4 = SimFusion_3in(
            in_channel_list=[backbone_channels[3], backbone_channels[3], backbone_channels[3]],  # [128, 128, 128]
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
            n=max(1, int(num_repeats[0] * 0.5)),  # 减少重复次数
            block=block
        )
        
        # P5 -> P4
        self.reduce_layer_p4 = SimConv(
            in_channels=backbone_channels[3],  # 128
            out_channels=backbone_channels[2],  # 64
            kernel_size=1,
            stride=1
        )
        
        self.LAF_p3 = SimFusion_3in(
            in_channel_list=[backbone_channels[2], backbone_channels[2], backbone_channels[2]],  # [64, 64, 64]
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
            n=max(1, int(num_repeats[1] * 0.5)),  # 减少重复次数
            block=block
        )
        
        # === High-GD (高级全局分布) - 简化版 ===
        # 高级特征聚合模块
        self.high_FAM = PyramidPoolAgg(
            stride=extra_cfg.get('c2t_stride', 2), 
            pool_mode=extra_cfg.get('pool_mode', 'torch')
        )
        
        # 高级信息融合模块 - 简化版
        self.high_IFM = TopBasicLayer(
            block_num=1,  # 减少transformer层数
            embedding_dim=extra_cfg['embed_dim_n'],
            key_dim=extra_cfg.get('key_dim', 8),
            num_heads=extra_cfg.get('num_heads', 4),
            mlp_ratio=extra_cfg.get('mlp_ratios', 1),
            attn_ratio=extra_cfg.get('attn_ratios', 2),
            drop=0, attn_drop=0,
            drop_path=[0.0],  # 简化drop_path
            norm_cfg=extra_cfg.get('norm_cfg')
        )
        
        # 简化的高级特征融合
        # 预计算拼接后的通道数: p3(64) + p4(128) + c5(128) = 320
        high_concat_channels = backbone_channels[2] + backbone_channels[3] + backbone_channels[4]  # 64+128+128=320
        self.high_fuse_conv = Conv(
            high_concat_channels,
            sum(extra_cfg['trans_channels'][2:4]),
            1, 1
        )
        
        # === 自底向上路径 - 简化版 ===
        # P3 -> N4
        self.downsample_p3 = SimConv(
            in_channels=backbone_channels[2],  # 64
            out_channels=backbone_channels[2],  # 64
            kernel_size=3,
            stride=2
        )
        
        self.LAF_n4 = AdvPoolFusion()
        self.Inject_n4 = InjectionMultiSum_Auto_pool(
            backbone_channels[3], backbone_channels[3],
            norm_cfg=extra_cfg.get('norm_cfg'), 
            activations=nn.ReLU6
        )
        
        self.Rep_n4 = RepBlock(
            in_channels=backbone_channels[2] + backbone_channels[3],  # 64 + 128 = 192
            out_channels=backbone_channels[3],  # 128
            n=max(1, int(num_repeats[2] * 0.5)) if len(num_repeats) > 2 else 2,
            block=block
        )
        
        # N4 -> N5
        self.downsample_n4 = SimConv(
            in_channels=backbone_channels[3],  # 128
            out_channels=backbone_channels[3],  # 128
            kernel_size=3,
            stride=2
        )
        
        self.LAF_n5 = AdvPoolFusion()
        self.Inject_n5 = InjectionMultiSum_Auto_pool(
            backbone_channels[4], backbone_channels[4],
            norm_cfg=extra_cfg.get('norm_cfg'),
            activations=nn.ReLU6
        )
        
        self.Rep_n5 = RepBlock(
            in_channels=backbone_channels[3] + backbone_channels[4],  # 128 + 128 = 256
            out_channels=backbone_channels[4],  # 128
            n=max(1, int(num_repeats[3] * 0.5)) if len(num_repeats) > 3 else 2,
            block=block
        )
    
    def execute(self, input):
        """
        简化但完整的forward方法
        """
        if len(input) == 5:
            c2, c3, c4, c5 = input[1:]  # 跳过P2，使用P3,P4,P5,P6
        else:
            c2, c3, c4, c5 = input
        
        # === Low-GD (低级全局分布) ===
        # 使用卷积融合全局信息
        low_align_feat = self.low_FAM(c2, c3, c4, c5)
        low_fuse_feat = self.low_IFM(low_align_feat)
        
        # 分割低级全局信息
        low_global_info = jt.split(low_fuse_feat, self.trans_channels[0:2], dim=1)
        
        # === 自顶向下路径 ===
        # P6 -> P5
        c5_half = self.reduce_layer_c5(c5)  # 128 -> 128
        
        # 通道匹配 - c3是P4(128), c4是P5(128), c5是P6(128)
        c3_matched = self.p4_match(c3)  # 128 -> 128
        c4_matched = self.p5_match(c4)  # 128 -> 128
        c5_matched = self.p5_match(c5_half)  # 128 -> 128
        
        # 注入低级全局信息到p4
        p4_adjacent_info = self.LAF_p4(c3_matched, c4_matched, c5_matched)
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        # P5 -> P4
        p4_half = self.reduce_layer_p4(p4)  # 128 -> 64
        
        # 通道匹配
        c2_matched = SimConv(c2.shape[1], 64, 1, 1)(c2) if c2.shape[1] != 64 else c2  # 动态匹配
        c3_matched_p3 = SimConv(c3.shape[1], 64, 1, 1)(c3) if c3.shape[1] != 64 else c3  # 动态匹配
        
        # 注入低级全局信息到p3
        p3_adjacent_info = self.LAF_p3(c2_matched, c3_matched_p3, p4_half)
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # === High-GD (高级全局分布) - 极简版 ===
        # 简化：直接使用卷积代替transformer
        # 将p3, p4, c5拼接并处理
        target_size = p3.shape[2:]
        p4_resized = jt.nn.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        c5_resized = jt.nn.interpolate(c5, size=target_size, mode='bilinear', align_corners=False)

        high_concat = jt.concat([p3, p4_resized, c5_resized], dim=1)  # 拼接所有特征

        # 使用预定义的卷积层
        high_fuse_feat = self.high_fuse_conv(high_concat)

        # 分割高级全局信息
        high_global_info = jt.split(high_fuse_feat, self.trans_channels[2:4], dim=1)
        
        # === 自底向上路径 ===
        # P3 -> N4
        p3_downsampled = self.downsample_p3(p3)
        n4_adjacent_info = self.LAF_n4(p4, p3_downsampled)
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])
        n4_concat = jt.concat([p3, p3_downsampled], dim=1)
        n4 = self.Rep_n4(n4_concat)
        
        # N4 -> N5
        n4_downsampled = self.downsample_n4(n4)
        n5_adjacent_info = self.LAF_n5(c5, n4_downsampled)
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])
        n5_concat = jt.concat([p4, n4_downsampled], dim=1)
        n5 = self.Rep_n5(n5_concat)
        
        # 输出三个尺度的特征
        outputs = [p3, n4, n5]
        
        return outputs
