#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完全修复的RepGDNeck实现 (Jittor版本)
新芽第二阶段：深度完善架构 - 精确通道匹配
"""

import jittor as jt
import jittor.nn as nn
from ..layers.common import (
    Conv, RepVGGBlock, RepBlock, SimConv, 
    SimFusion_3in, SimFusion_4in, AdvPoolFusion,
    InjectionMultiSum_Auto_pool
)


class FixedRepGDNeck(nn.Module):
    """完全修复的RepGDNeck - 精确通道匹配版本"""
    def __init__(self, extra_cfg=None):
        super().__init__()
        
        # 根据系统分析的精确通道配置
        # Backbone输出: [P2(32), P3(64), P4(128), P5(128), P6(128)]
        # Neck使用: [c2(64), c3(128), c4(128), c5(128)] 对应 [P3, P4, P5, P6]
        
        self.c2_channels = 64   # P3
        self.c3_channels = 128  # P4
        self.c4_channels = 128  # P5
        self.c5_channels = 128  # P6
        
        # trans_channels配置
        self.trans_channels = [16, 8, 16, 32]  # Nano版本
        
        # === Low-GD (低级全局分布) ===
        self.low_FAM = SimFusion_4in()
        
        # Low-IFM: 448 -> 24 -> 24
        self.low_IFM = nn.Sequential(
            Conv(448, 24, kernel_size=1, stride=1, padding=0),
            RepVGGBlock(24, 24),
            RepVGGBlock(24, 24),
            RepVGGBlock(24, 24),
            Conv(24, 24, kernel_size=1, stride=1, padding=0)
        )
        
        # === 自顶向下路径 ===
        # P6(128) -> P5(128)
        self.reduce_layer_c5 = SimConv(128, 128, kernel_size=1, stride=1)
        
        # LAF_p4: [P4(128), P5(128), P6_reduced(128)] -> 128
        self.LAF_p4 = SimFusion_3in(
            in_channel_list=[128, 128, 128], 
            out_channels=128
        )
        
        self.Inject_p4 = InjectionMultiSum_Auto_pool(128, 128)
        self.Rep_p4 = RepBlock(in_channels=128, out_channels=128, n=2, block=RepVGGBlock)
        
        # P5(128) -> P4(128)
        self.reduce_layer_p4 = SimConv(128, 128, kernel_size=1, stride=1)
        
        # LAF_p3: [P3(64), P4(128), P5_reduced(128)] -> 128
        # 需要通道匹配层将P3从64调整到128
        self.p3_channel_match = SimConv(64, 128, kernel_size=1, stride=1)
        
        self.LAF_p3 = SimFusion_3in(
            in_channel_list=[128, 128, 128], 
            out_channels=128
        )
        
        self.Inject_p3 = InjectionMultiSum_Auto_pool(128, 128)
        self.Rep_p3 = RepBlock(in_channels=128, out_channels=128, n=2, block=RepVGGBlock)
        
        # === High-GD (高级全局分布) ===
        # high_fuse_conv: p3(128) + p4(128) + c5(128) = 384 -> 48
        self.high_fuse_conv = Conv(384, 48, kernel_size=1, stride=1)
        
        # === 自底向上路径 ===
        # P3(128) -> N4
        self.downsample_p3 = SimConv(128, 128, kernel_size=3, stride=2)
        
        # LAF_n4: 直接使用concat，不使用AdvPoolFusion
        self.Inject_n4 = InjectionMultiSum_Auto_pool(256, 128)  # 256 -> 128
        self.Rep_n4 = RepBlock(in_channels=256, out_channels=128, n=2, block=RepVGGBlock)

        # N4(128) -> N5
        self.downsample_n4 = SimConv(128, 128, kernel_size=3, stride=2)

        # LAF_n5: 直接使用concat，不使用AdvPoolFusion
        self.Inject_n5 = InjectionMultiSum_Auto_pool(256, 128)  # 256 -> 128
        self.Rep_n5 = RepBlock(in_channels=256, out_channels=128, n=2, block=RepVGGBlock)
    
    def execute(self, input):
        """
        完全修复的forward方法 - 精确通道匹配
        input: backbone输出 [P2, P3, P4, P5, P6]
        """
        if len(input) == 5:
            P2, P3, P4, P5, P6 = input
            # 使用后4个特征: [P3, P4, P5, P6] -> [c2, c3, c4, c5]
            c2, c3, c4, c5 = P3, P4, P5, P6
        else:
            # 如果只有4个特征
            c2, c3, c4, c5 = input
        
        # 验证通道数
        assert c2.shape[1] == self.c2_channels, f"c2通道数不匹配: {c2.shape[1]} vs {self.c2_channels}"
        assert c3.shape[1] == self.c3_channels, f"c3通道数不匹配: {c3.shape[1]} vs {self.c3_channels}"
        assert c4.shape[1] == self.c4_channels, f"c4通道数不匹配: {c4.shape[1]} vs {self.c4_channels}"
        assert c5.shape[1] == self.c5_channels, f"c5通道数不匹配: {c5.shape[1]} vs {self.c5_channels}"
        
        # === Low-GD (低级全局分布) ===
        # 特征聚合: c2(64) + c3(128) + c4(128) + c5(128) = 448
        low_align_feat = self.low_FAM(c2, c3, c4, c5)
        assert low_align_feat.shape[1] == 448, f"fusion输入通道数不匹配: {low_align_feat.shape[1]} vs 448"
        
        # 信息融合: 448 -> 24
        low_fuse_feat = self.low_IFM(low_align_feat)
        assert low_fuse_feat.shape[1] == 24, f"low_IFM输出通道数不匹配: {low_fuse_feat.shape[1]} vs 24"
        
        # 分割低级全局信息: 24 -> [16, 8]
        low_global_info = jt.split(low_fuse_feat, self.trans_channels[0:2], dim=1)
        
        # === 自顶向下路径 ===
        # P6(128) -> P5(128)
        c5_reduced = self.reduce_layer_c5(c5)  # 128 -> 128
        
        # LAF_p4: [P4(128), P5(128), P6_reduced(128)] -> 128
        p4_adjacent_info = self.LAF_p4(c3, c4, c5_reduced)
        assert p4_adjacent_info.shape[1] == 128, f"LAF_p4输出通道数不匹配: {p4_adjacent_info.shape[1]} vs 128"
        
        # 注入低级全局信息
        p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])  # 注入16通道信息
        p4 = self.Rep_p4(p4)
        
        # P5(128) -> P4(128)
        p4_reduced = self.reduce_layer_p4(p4)  # 128 -> 128
        
        # LAF_p3: [P3_matched(128), P4(128), P5_reduced(128)] -> 128
        c2_matched = self.p3_channel_match(c2)  # 64 -> 128
        p3_adjacent_info = self.LAF_p3(c2_matched, c3, p4_reduced)
        assert p3_adjacent_info.shape[1] == 128, f"LAF_p3输出通道数不匹配: {p3_adjacent_info.shape[1]} vs 128"
        
        # 注入低级全局信息
        p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])  # 注入8通道信息
        p3 = self.Rep_p3(p3)
        
        # === High-GD (高级全局分布) ===
        # 特征拼接: p3(128) + p4(128) + c5(128) = 384
        target_size = p3.shape[2:]
        p4_resized = jt.nn.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        c5_resized = jt.nn.interpolate(c5, size=target_size, mode='bilinear', align_corners=False)
        
        high_concat = jt.concat([p3, p4_resized, c5_resized], dim=1)
        assert high_concat.shape[1] == 384, f"high_concat通道数不匹配: {high_concat.shape[1]} vs 384"
        
        # 高级信息融合: 384 -> 48
        high_fuse_feat = self.high_fuse_conv(high_concat)
        assert high_fuse_feat.shape[1] == 48, f"high_fuse_feat通道数不匹配: {high_fuse_feat.shape[1]} vs 48"
        
        # 分割高级全局信息: 48 -> [16, 32]
        high_global_info = jt.split(high_fuse_feat, self.trans_channels[2:4], dim=1)
        
        # === 自底向上路径 ===
        # P3(128) -> N4
        p3_downsampled = self.downsample_p3(p3)  # 128 -> 128
        
        # LAF_n4: P4(128) + P3_down(128) -> concat(256)
        # 直接concat，确保尺寸匹配
        if p4.shape[2:] != p3_downsampled.shape[2:]:
            p3_downsampled = jt.nn.interpolate(p3_downsampled, size=p4.shape[2:], mode='bilinear', align_corners=False)

        n4_adjacent_info = jt.concat([p4, p3_downsampled], dim=1)  # 直接concat
        assert n4_adjacent_info.shape[1] == 256, f"n4_adjacent_info通道数不匹配: {n4_adjacent_info.shape[1]} vs 256"

        # 注入高级全局信息
        n4 = self.Inject_n4(n4_adjacent_info, high_global_info[0])  # 注入16通道信息
        n4 = self.Rep_n4(n4_adjacent_info)  # 使用原始concat结果

        # N4(128) -> N5
        n4_downsampled = self.downsample_n4(n4)  # 128 -> 128

        # LAF_n5: P6(128) + N4_down(128) -> concat(256)
        # 直接concat，确保尺寸匹配
        if c5.shape[2:] != n4_downsampled.shape[2:]:
            n4_downsampled = jt.nn.interpolate(n4_downsampled, size=c5.shape[2:], mode='bilinear', align_corners=False)

        n5_adjacent_info = jt.concat([c5, n4_downsampled], dim=1)  # 直接concat
        assert n5_adjacent_info.shape[1] == 256, f"n5_adjacent_info通道数不匹配: {n5_adjacent_info.shape[1]} vs 256"

        # 注入高级全局信息
        n5 = self.Inject_n5(n5_adjacent_info, high_global_info[1])  # 注入32通道信息
        n5 = self.Rep_n5(n5_adjacent_info)  # 使用原始concat结果
        
        # 输出三个尺度的特征: [P3, N4, N5]
        outputs = [p3, n4, n5]
        
        return outputs
