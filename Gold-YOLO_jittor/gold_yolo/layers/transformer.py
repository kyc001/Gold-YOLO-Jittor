#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gold-YOLO Transformer组件 (Jittor版本)
新芽第二阶段：完整架构实现
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
from .common import Conv


class PyramidPoolAgg(nn.Module):
    """金字塔池化聚合"""
    def __init__(self, stride=2, pool_mode='torch'):
        super().__init__()
        self.stride = stride
        self.pool_mode = pool_mode
        
        # 不同尺度的池化
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(4),
            nn.AdaptiveAvgPool2d(8)
        ])
    
    def execute(self, inputs):
        # inputs可能是单个tensor或tensor列表
        if isinstance(inputs, (list, tuple)):
            # 如果是列表，取第一个作为主要特征
            x = inputs[0]
            # 确保所有特征尺寸一致
            target_size = x.shape[2:]
            aligned_features = []
            for feat in inputs:
                if feat.shape[2:] != target_size:
                    feat = jt.nn.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                aligned_features.append(feat)
            # 拼接所有特征
            x = jt.concat(aligned_features, dim=1)
        else:
            x = inputs

        B, C, H, W = x.shape

        # 多尺度池化
        features = []
        for pool in self.pools:
            pooled = pool(x)
            # 上采样回原始尺寸
            upsampled = jt.nn.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=False)
            features.append(upsampled)

        # 拼接所有特征
        return jt.concat(features, dim=1)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads, key_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = key_dim or embed_dim // num_heads
        self.scale = self.key_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def execute(self, x):
        B, N, C = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = jt.nn.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., key_dim=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, key_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, drop=drop)
        
        # 简化drop_path实现
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
    
    def execute(self, x):
        # 注意力分支
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP分支
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class TopBasicLayer(nn.Module):
    """顶层基础层"""
    def __init__(self, block_num, embedding_dim, key_dim, num_heads, mlp_ratio=4., 
                 attn_ratio=2, drop=0., attn_drop=0., drop_path=None, norm_cfg=None):
        super().__init__()
        
        if drop_path is None:
            drop_path = [0.] * block_num
        
        # Transformer块序列
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(
                TransformerBlock(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    key_dim=key_dim,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
                )
            )
        
        self.norm = nn.LayerNorm(embedding_dim)
    
    def execute(self, x):
        # x shape: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 恢复空间维度: [B, H*W, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class InjectionMultiSum_Auto_pool(nn.Module):
    """注入多重求和自动池化 - 增强版"""
    def __init__(self, in_channels, out_channels, norm_cfg=None, activations=nn.ReLU):
        super().__init__()
        
        # 特征变换
        self.conv_transform = Conv(in_channels, out_channels, 1, 1)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.activation = activations()
    
    def execute(self, x, injection=None):
        # 特征变换
        x = self.conv_transform(x)

        # 如果有注入特征
        if injection is not None:
            # 确保尺寸和通道数都匹配
            if injection.shape[2:] != x.shape[2:]:
                injection = jt.nn.interpolate(injection, size=x.shape[2:], mode='bilinear', align_corners=False)

            # 确保通道数匹配
            if injection.shape[1] != x.shape[1]:
                # 使用1x1卷积调整通道数
                channel_match = nn.Conv2d(injection.shape[1], x.shape[1], 1, 1, 0)
                injection = channel_match(injection)

            x = x + injection

        # 应用注意力
        att = self.attention(x)
        x = x * att

        return self.activation(x)


class C2T_Attention(nn.Module):
    """卷积到Transformer的注意力模块"""
    def __init__(self, in_channels, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 特征投影
        self.proj = nn.Conv2d(in_channels, embed_dim, 1)
        
        # 位置编码 - 深入修复Parameter警告
        # 在Jittor中，直接创建变量即可，不需要Parameter包装
        self.pos_embed = jt.randn(1, embed_dim, 1, 1)
        
        # 注意力
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # 输出投影
        self.out_proj = nn.Conv2d(embed_dim, in_channels, 1)
    
    def execute(self, x):
        B, C, H, W = x.shape
        
        # 特征投影
        x_proj = self.proj(x)
        
        # 添加位置编码
        x_proj = x_proj + self.pos_embed
        
        # 重塑为序列格式
        x_seq = x_proj.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
        
        # 应用注意力
        x_att = self.attention(x_seq)
        
        # 恢复空间格式
        x_att = x_att.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        
        # 输出投影
        out = self.out_proj(x_att)
        
        return out + x  # 残差连接
