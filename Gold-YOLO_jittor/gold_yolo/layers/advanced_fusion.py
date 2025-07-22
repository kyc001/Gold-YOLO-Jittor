#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
高级融合模块实现 (Jittor版本)
新芽第二阶段：方案A3 - 增强融合模块
"""

import jittor as jt
import jittor.nn as nn
import math
from .common import Conv, RepVGGBlock


class InformationAlignmentModule(nn.Module):
    """信息对齐模块 (IAM) - Gold-YOLO的核心创新"""
    
    def __init__(self, in_channels, out_channels, key_dim=8, num_heads=4, attn_ratio=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.attn_ratio = attn_ratio
        
        # 特征变换
        self.input_proj = Conv(in_channels, out_channels, 1, 1)
        
        # 多头注意力机制
        self.attention = MultiScaleAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            key_dim=key_dim,
            attn_ratio=attn_ratio
        )
        
        # 输出投影
        self.output_proj = Conv(out_channels, out_channels, 1, 1)
        
        # 残差连接
        self.residual = nn.Identity() if in_channels == out_channels else Conv(in_channels, out_channels, 1, 1)
    
    def execute(self, x):
        """
        x: 输入特征 [B, C, H, W]
        """
        # 残差连接
        residual = self.residual(x)
        
        # 特征变换
        x = self.input_proj(x)
        
        # 多尺度注意力
        x = self.attention(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        # 残差连接
        return x + residual


class MultiScaleAttention(nn.Module):
    """多尺度注意力机制"""
    
    def __init__(self, embed_dim, num_heads=4, key_dim=8, attn_ratio=2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.key_dim ** -0.5
        
        # Q, K, V投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # 多尺度池化
        self.pool_scales = [1, 2, 4]
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in self.pool_scales
        ])
        
        # 位置编码
        self.pos_embed = nn.Parameter(jt.randn(1, embed_dim, 1, 1) * 0.02)

        # 多尺度特征降维
        self.dim_reduce = Conv(embed_dim * (1 + len(self.pool_scales)), embed_dim, 1, 1)

        self.dropout = nn.Dropout(0.1)
    
    def execute(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 多尺度特征提取
        multi_scale_feats = []
        for pool in self.pools:
            pooled = pool(x)  # [B, C, scale, scale]
            # 上采样回原始尺寸
            upsampled = jt.nn.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            multi_scale_feats.append(upsampled)
        
        # 拼接多尺度特征
        x_ms = jt.concat([x] + multi_scale_feats, dim=1)  # [B, C*(1+len(scales)), H, W]
        
        # 使用预定义的降维层
        x_ms = self.dim_reduce(x_ms)
        
        # 重塑为序列格式进行注意力计算
        x_seq = x_ms.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 生成Q, K, V
        qkv = self.qkv(x_seq).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = jt.nn.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn)
        
        # 重塑回空间格式
        x_out = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        return x_out


class AdvancedPoolingFusion(nn.Module):
    """高级池化融合模块"""
    
    def __init__(self, in_channels_list, out_channels, fusion_type='adaptive'):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # 通道对齐
        self.channel_align = nn.ModuleList([
            Conv(ch, out_channels, 1, 1) for ch in in_channels_list
        ])
        
        # 自适应权重
        if fusion_type == 'adaptive':
            self.adaptive_weights = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(out_channels, out_channels // 4, 1),
                    nn.ReLU(),
                    nn.Conv2d(out_channels // 4, out_channels, 1),
                    nn.Sigmoid()
                ) for _ in in_channels_list
            ])
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            Conv(out_channels * len(in_channels_list), out_channels, 3, 1),
            RepVGGBlock(out_channels, out_channels),
            Conv(out_channels, out_channels, 1, 1)
        )
    
    def execute(self, features):
        """
        features: 多个特征的列表 [feat1, feat2, ...]
        """
        # 获取目标尺寸
        target_size = features[0].shape[2:]
        
        aligned_features = []
        for i, feat in enumerate(features):
            # 通道对齐
            feat_aligned = self.channel_align[i](feat)
            
            # 尺寸对齐
            if feat_aligned.shape[2:] != target_size:
                feat_aligned = jt.nn.interpolate(feat_aligned, size=target_size, mode='bilinear', align_corners=False)
            
            # 自适应权重
            if self.fusion_type == 'adaptive':
                weight = self.adaptive_weights[i](feat_aligned)
                feat_aligned = feat_aligned * weight
            
            aligned_features.append(feat_aligned)
        
        # 拼接融合
        fused_feat = jt.concat(aligned_features, dim=1)
        
        # 融合卷积
        output = self.fusion_conv(fused_feat)
        
        return output


class EnhancedInjectionModule(nn.Module):
    """增强的注入模块"""
    
    def __init__(self, main_channels, injection_channels, out_channels):
        super().__init__()
        
        self.main_channels = main_channels
        self.injection_channels = injection_channels
        self.out_channels = out_channels
        
        # 主特征处理
        self.main_conv = Conv(main_channels, out_channels, 1, 1)
        
        # 注入特征处理
        self.injection_conv = Conv(injection_channels, out_channels, 1, 1)
        
        # 注意力门控 - 输出通道数应该匹配concat_feat
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * 2, 1),  # 输出与concat_feat相同的通道数
            nn.Sigmoid()
        )
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            Conv(out_channels * 2, out_channels, 3, 1),
            RepVGGBlock(out_channels, out_channels)
        )
        
        # 输出激活
        self.activation = nn.ReLU6()
    
    def execute(self, main_feat, injection_feat):
        """
        main_feat: 主要特征
        injection_feat: 注入特征
        """
        # 特征处理
        main_processed = self.main_conv(main_feat)
        injection_processed = self.injection_conv(injection_feat)
        
        # 尺寸对齐
        if injection_processed.shape[2:] != main_processed.shape[2:]:
            injection_processed = jt.nn.interpolate(
                injection_processed, 
                size=main_processed.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 拼接特征
        concat_feat = jt.concat([main_processed, injection_processed], dim=1)
        
        # 注意力门控
        attention = self.attention_gate(concat_feat)
        gated_feat = concat_feat * attention
        
        # 融合卷积
        fused_feat = self.fusion_conv(gated_feat)
        
        return self.activation(fused_feat)


class GatherDistributeModule(nn.Module):
    """聚集分发模块 - Gold-YOLO的GD机制"""
    
    def __init__(self, in_channels, embed_dim, num_heads=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 聚集阶段
        self.gather_conv = Conv(in_channels, embed_dim, 1, 1)
        self.gather_attention = MultiScaleAttention(embed_dim, num_heads)
        
        # 分发阶段
        self.distribute_conv = Conv(embed_dim, in_channels, 1, 1)
        
        # 残差连接
        self.residual = nn.Identity() if in_channels == embed_dim else Conv(in_channels, embed_dim, 1, 1)
    
    def execute(self, x):
        """
        x: 输入特征 [B, C, H, W]
        """
        # 残差连接
        residual = x
        
        # 聚集阶段
        gathered = self.gather_conv(x)
        gathered = self.gather_attention(gathered)
        
        # 分发阶段
        distributed = self.distribute_conv(gathered)
        
        # 残差连接
        return distributed + residual
