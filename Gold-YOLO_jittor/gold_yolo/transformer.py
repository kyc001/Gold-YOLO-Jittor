# Jittor implementation of Gold-YOLO transformer components
# Migrated from PyTorch version

import numpy as np
import jittor as jt
from jittor import nn
import jittor.nn as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_yolo.layers import Conv2d_BN, DropPath, h_sigmoid


def get_shape(tensor):
    """Get tensor shape"""
    return tensor.shape


class Mlp(nn.Module):
    """Multi-layer perceptron"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features)
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
    """Multi-head attention mechanism"""
    
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4, activation=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1)
        self.to_k = Conv2d_BN(dim, nh_kd, 1)
        self.to_v = Conv2d_BN(dim, self.dh, 1)
        
        if activation is None:
            activation = nn.ReLU
        self.proj = nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0))
    
    def execute(self, x):  # x (B,C,H,W)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = jt.matmul(qq, kk)
        attn = jt.nn.softmax(attn * self.scale, dim=-1)  # dim = k
        
        xx = jt.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer)
        
        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def execute(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    """Basic layer with multiple transformer blocks"""
    
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer))
    
    def execute(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


def jittor_adaptive_avg_pool2d(x, output_size):
    """Jittor adaptive average pooling using interpolation"""
    if isinstance(output_size, (list, tuple, np.ndarray)):
        if len(output_size) == 2:
            target_h, target_w = int(output_size[0]), int(output_size[1])
        else:
            target_h = target_w = int(output_size[0])
    else:
        target_h = target_w = int(output_size)

    # Use interpolation as a simple alternative to adaptive pooling
    return jt.nn.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)


class PyramidPoolAgg(nn.Module):
    """Pyramid pooling aggregation"""
    
    def __init__(self, stride, pool_mode='jittor'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'jittor':
            self.pool = jittor_adaptive_avg_pool2d
        else:
            self.pool = jittor_adaptive_avg_pool2d
    
    def execute(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        
        output_size = np.array([H, W])
        
        out = [self.pool(inp, output_size) for inp in inputs]
        
        return jt.concat(out, dim=1)


def get_avg_pool():
    """Get adaptive average pooling function"""
    return jittor_adaptive_avg_pool2d


class ConvModule(nn.Module):
    """Simple ConvModule replacement for Jittor"""

    def __init__(self, in_channels, out_channels, kernel_size, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        if norm_cfg is not None:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if act_cfg is not None:
            self.act = nn.ReLU()
        else:
            self.act = None

    def execute(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class InjectionMultiSum_Auto_pool(nn.Module):
    """Injection multi-sum with auto pooling"""

    def __init__(self, inp: int, oup: int, activations=None, global_inp=None) -> None:
        super().__init__()

        if not global_inp:
            global_inp = inp

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, act_cfg=None)
        self.act = h_sigmoid()

    def execute(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = jt.nn.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = jt.nn.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out
