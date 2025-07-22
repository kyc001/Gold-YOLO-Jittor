#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
增强版RepGDNeck实现 (Jittor版本)
新芽第二阶段：方案A2+A3 - 完善检测头和融合模块
"""

import jittor as jt
import jittor.nn as nn
from ..layers.common import (
    Conv, RepVGGBlock, RepBlock, SimConv, 
    SimFusion_3in, SimFusion_4in
)
from ..layers.advanced_fusion import (
    InformationAlignmentModule, AdvancedPoolingFusion,
    EnhancedInjectionModule, GatherDistributeModule
)


class EnhancedRepGDNeck(nn.Module):
    """增强版RepGDNeck - 集成高级融合模块"""
    
    def __init__(self, extra_cfg=None):
        super().__init__()
        
        # 通道配置
        self.c2_channels = 64   # P3
        self.c3_channels = 128  # P4
        self.c4_channels = 128  # P5
        self.c5_channels = 128  # P6
        
        # trans_channels配置
        self.trans_channels = [16, 8, 16, 32]  # Nano版本
        
        # === Low-GD (低级全局分布) - 增强版 ===
        # 使用高级融合模块替代简单的SimFusion_4in
        self.low_FAM = AdvancedPoolingFusion(
            in_channels_list=[64, 128, 128, 128],  # [c2, c3, c4, c5]
            out_channels=128,
            fusion_type='adaptive'
        )
        
        # 增强的信息融合模块
        self.low_IFM = nn.Sequential(
            Conv(128, 64, kernel_size=1, stride=1, padding=0),  # 降维
            GatherDistributeModule(64, 32, num_heads=2),  # GD机制
            RepVGGBlock(64, 64),
            RepVGGBlock(64, 64),
            Conv(64, 24, kernel_size=1, stride=1, padding=0)  # 输出24通道
        )
        
        # === 自顶向下路径 - 增强版 ===
        # P6(128) -> P5(128)
        self.reduce_layer_c5 = SimConv(128, 128, kernel_size=1, stride=1)
        
        # 使用IAM替代简单的LAF
        self.IAM_p4 = InformationAlignmentModule(
            in_channels=128 * 3,  # 拼接后的通道数
            out_channels=128,
            key_dim=8,
            num_heads=4,
            attn_ratio=2
        )
        
        # 增强的注入模块
        self.Inject_p4 = EnhancedInjectionModule(
            main_channels=128,
            injection_channels=16,  # low_global_info[0]
            out_channels=128
        )
        
        self.Rep_p4 = RepBlock(in_channels=128, out_channels=128, n=3, block=RepVGGBlock)
        
        # P5(128) -> P4(128)
        self.reduce_layer_p4 = SimConv(128, 128, kernel_size=1, stride=1)
        
        # P3通道匹配
        self.p3_channel_match = SimConv(64, 128, kernel_size=1, stride=1)
        
        # 使用IAM替代简单的LAF
        self.IAM_p3 = InformationAlignmentModule(
            in_channels=128 * 3,  # 拼接后的通道数
            out_channels=128,
            key_dim=8,
            num_heads=4,
            attn_ratio=2
        )
        
        # 增强的注入模块
        self.Inject_p3 = EnhancedInjectionModule(
            main_channels=128,
            injection_channels=8,  # low_global_info[1]
            out_channels=128
        )
        
        self.Rep_p3 = RepBlock(in_channels=128, out_channels=128, n=3, block=RepVGGBlock)
        
        # === High-GD (高级全局分布) - 增强版 ===
        # 使用高级融合替代简单拼接
        self.high_FAM = AdvancedPoolingFusion(
            in_channels_list=[128, 128, 128],  # [p3, p4, c5]
            out_channels=128,
            fusion_type='adaptive'
        )
        
        # 增强的信息融合
        self.high_IFM = nn.Sequential(
            GatherDistributeModule(128, 64, num_heads=4),  # GD机制
            Conv(128, 48, kernel_size=1, stride=1)  # 输出48通道
        )
        
        # === 自底向上路径 - 增强版 ===
        # P3(128) -> N4
        self.downsample_p3 = SimConv(128, 128, kernel_size=3, stride=2)
        
        # 增强的注入模块
        self.Inject_n4 = EnhancedInjectionModule(
            main_channels=256,  # concat后的通道数
            injection_channels=16,  # high_global_info[0]
            out_channels=128
        )
        
        self.Rep_n4 = RepBlock(in_channels=128, out_channels=128, n=3, block=RepVGGBlock)
        
        # N4(128) -> N5
        self.downsample_n4 = SimConv(128, 128, kernel_size=3, stride=2)
        
        # 增强的注入模块
        self.Inject_n5 = EnhancedInjectionModule(
            main_channels=256,  # concat后的通道数
            injection_channels=32,  # high_global_info[1]
            out_channels=128
        )
        
        self.Rep_n5 = RepBlock(in_channels=128, out_channels=128, n=3, block=RepVGGBlock)
    
    def execute(self, input):
        """
        增强版forward方法
        input: backbone输出 [P2, P3, P4, P5, P6]
        """
        if len(input) == 5:
            P2, P3, P4, P5, P6 = input
            c2, c3, c4, c5 = P3, P4, P5, P6
        else:
            c2, c3, c4, c5 = input
        
        # === Low-GD (低级全局分布) - 增强版 ===
        # 使用高级融合模块
        low_align_feat = self.low_FAM([c2, c3, c4, c5])
        
        # 增强的信息融合
        low_fuse_feat = self.low_IFM(low_align_feat)
        
        # 分割低级全局信息: 24 -> [16, 8]
        low_global_info = jt.split(low_fuse_feat, self.trans_channels[0:2], dim=1)
        
        # === 自顶向下路径 - 增强版 ===
        # P6(128) -> P5(128)
        c5_reduced = self.reduce_layer_c5(c5)
        
        # 使用IAM进行信息对齐
        target_size = c3.shape[2:]
        c4_resized = jt.nn.interpolate(c4, size=target_size, mode='bilinear', align_corners=False)
        c5_resized = jt.nn.interpolate(c5_reduced, size=target_size, mode='bilinear', align_corners=False)
        
        p4_concat = jt.concat([c3, c4_resized, c5_resized], dim=1)  # 128*3=384
        p4_aligned = self.IAM_p4(p4_concat)
        
        # 增强的注入
        p4 = self.Inject_p4(p4_aligned, low_global_info[0])
        p4 = self.Rep_p4(p4)
        
        # P5(128) -> P4(128)
        p4_reduced = self.reduce_layer_p4(p4)
        
        # 使用IAM进行信息对齐 - 确保所有特征尺寸一致
        c2_matched = self.p3_channel_match(c2)  # 64 -> 128

        # 使用c2的尺寸作为目标尺寸
        p3_target_size = c2.shape[2:]
        c3_resized = jt.nn.interpolate(c3, size=p3_target_size, mode='bilinear', align_corners=False)
        p4_resized = jt.nn.interpolate(p4_reduced, size=p3_target_size, mode='bilinear', align_corners=False)

        p3_concat = jt.concat([c2_matched, c3_resized, p4_resized], dim=1)  # 128*3=384
        p3_aligned = self.IAM_p3(p3_concat)
        
        # 增强的注入
        p3 = self.Inject_p3(p3_aligned, low_global_info[1])
        p3 = self.Rep_p3(p3)
        
        # === High-GD (高级全局分布) - 增强版 ===
        # 使用高级融合模块
        high_align_feat = self.high_FAM([p3, p4, c5])
        
        # 增强的信息融合
        high_fuse_feat = self.high_IFM(high_align_feat)
        
        # 分割高级全局信息: 48 -> [16, 32]
        high_global_info = jt.split(high_fuse_feat, self.trans_channels[2:4], dim=1)
        
        # === 自底向上路径 - 增强版 ===
        # P3(128) -> N4
        p3_downsampled = self.downsample_p3(p3)
        
        # 直接concat
        if p4.shape[2:] != p3_downsampled.shape[2:]:
            p3_downsampled = jt.nn.interpolate(p3_downsampled, size=p4.shape[2:], mode='bilinear', align_corners=False)
        
        n4_concat = jt.concat([p4, p3_downsampled], dim=1)  # 256
        
        # 增强的注入
        n4 = self.Inject_n4(n4_concat, high_global_info[0])
        n4 = self.Rep_n4(n4)
        
        # N4(128) -> N5
        n4_downsampled = self.downsample_n4(n4)
        
        # 直接concat
        if c5.shape[2:] != n4_downsampled.shape[2:]:
            n4_downsampled = jt.nn.interpolate(n4_downsampled, size=c5.shape[2:], mode='bilinear', align_corners=False)
        
        n5_concat = jt.concat([c5, n4_downsampled], dim=1)  # 256
        
        # 增强的注入
        n5 = self.Inject_n5(n5_concat, high_global_info[1])
        n5 = self.Rep_n5(n5)
        
        # 输出三个尺度的特征: [P3, N4, N5]
        outputs = [p3, n4, n5]
        
        return outputs
