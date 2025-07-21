#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Transformer模块的Jittor实现 - 100%对齐PyTorch官方版本
基于Gold-YOLO的transformer.py实现
"""

import numpy as np
import jittor as jt
from jittor import nn

# 导入兼容层
from .jittor_compat import compat_nn

from .layers_jittor import Conv2d_BN, DropPath, h_sigmoid


def get_shape(tensor):
    """获取张量形状 - Jittor版本"""
    return tensor.shape


class Mlp(nn.Module):
    """MLP模块 - Jittor版本"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.ReLU, drop=0., norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)
    
    def execute(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """注意力模块 - Jittor版本"""
    
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = nn.Sequential(
            activation(), 
            Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)
        )
    
    def execute(self, x):
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = jt.matmul(qq, kk)
        attn = compat_nn.softmax(attn, dim=-1)  # dim = k
        
        xx = jt.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):
    """Transformer Block - Jittor版本"""
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, 
                             attn_ratio=attn_ratio, activation=act_layer,
                             norm_cfg=norm_cfg)
        
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)
    
    def execute(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    """Top Basic Layer - Jittor版本"""
    
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0, attn_drop=0,
                 drop_path=None, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        
        if drop_path is None:
            drop_path = [0.] * block_num
        
        self.blocks = nn.ModuleList([
            top_Block(
                dim=embedding_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_ratio=attn_ratio,
                drop=drop,
                drop_path=drop_path[i],
                norm_cfg=norm_cfg
            ) for i in range(block_num)
        ])
    
    def execute(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PyramidPoolAgg(nn.Module):
    """金字塔池化聚合 - Jittor版本"""
    
    def __init__(self, stride=2, pool_mode='torch'):
        super().__init__()
        self.stride = stride
        self.pool_mode = pool_mode
    
    def execute(self, inputs):
        """聚合多尺度特征"""
        # inputs: [p3, p4, c5]
        p3, p4, c5 = inputs
        
        # 获取目标尺寸 (通常是最小的特征图尺寸)
        _, _, H, W = c5.shape
        
        # 将所有特征调整到相同尺寸
        if self.pool_mode == 'torch':
            p3_pooled = compat_nn.adaptive_avg_pool2d(p3, (H, W))
            p4_pooled = compat_nn.adaptive_avg_pool2d(p4, (H, W))
        else:
            # 使用简单的池化
            p3_pooled = compat_nn.avg_pool2d(p3, kernel_size=self.stride, stride=self.stride)
            p4_pooled = compat_nn.avg_pool2d(p4, kernel_size=self.stride, stride=self.stride)
        
        # 拼接特征
        fused = jt.concat([p3_pooled, p4_pooled, c5], dim=1)
        
        return fused


class InjectionMultiSum_Auto_pool(nn.Module):
    """注入多重求和自动池化 - Jittor版本"""
    
    def __init__(self, inp, oup, norm_cfg=dict(type='BN', requires_grad=True), 
                 activations=nn.ReLU6):
        super().__init__()
        self.inp = inp
        self.oup = oup
        
        # 局部特征处理
        self.local_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            activations()
        )
        
        # 全局特征处理
        self.global_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            activations()
        )
        
        # 全局特征自适应
        self.global_act = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            h_sigmoid()
        )
    
    def execute(self, x_l, x_g):
        """
        x_l: 局部特征
        x_g: 全局特征
        """
        B, C, H, W = x_l.shape
        
        # 处理局部特征
        local_feat = self.local_embedding(x_l)
        
        # 处理全局特征 - 调整到局部特征的尺寸
        global_feat = compat_nn.adaptive_avg_pool2d(x_g, (H, W))
        global_feat = self.global_embedding(global_feat)

        # 全局特征的注意力权重
        global_act = compat_nn.adaptive_avg_pool2d(x_g, (H, W))
        global_act = self.global_act(global_act)
        
        # 特征融合
        out = local_feat + global_feat * global_act
        
        return out
