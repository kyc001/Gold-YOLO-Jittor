# 2023.09.18-Changed for transformer of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# This file editing from https://github.com/hustvl/TopFormer/blob/main/mmseg/models/backbones/topformer.py
"""
GOLD-YOLO Jittor版本 - Transformer模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import numpy as np
import jittor as jt
import jittor.nn as nn

from .layers import Conv2d_BN, DropPath, h_sigmoid, ConvModule, build_norm_layer


def get_shape(tensor):
    shape = tensor.shape
    # Jittor没有ONNX导出，直接返回shape
    return shape


def onnx_AdaptiveAvgPool2d(x, output_size):
    """ONNX兼容的自适应平均池化"""
    if isinstance(output_size, np.ndarray):
        output_size = output_size.astype(np.int32)
        input_size = np.array(x.shape[-2:])
        stride_size = np.floor(input_size / output_size).astype(np.int32)
        kernel_size = input_size - (output_size - 1) * stride_size

        # 转换为Python int类型
        kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        stride_size = (int(stride_size[0]), int(stride_size[1]))

        avg = jt.pool.AvgPool2d(kernel_size=kernel_size, stride=stride_size)
        x = avg(x)
        return x
    else:
        return jt.pool.AdaptiveAvgPool2d(output_size)(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
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

        # 如果activation为None，使用ReLU6作为默认激活函数
        if activation is None:
            activation = nn.ReLU6

        self.proj = nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
    
    def execute(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = jt.matmul(qq, kk)
        attn = nn.softmax(attn, dim=-1)  # dim = k
        
        xx = jt.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,
                              norm_cfg=norm_cfg)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)
    
    def execute(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True), act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                embedding_dim, key_dim, num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg, act_layer=act_layer))

    def execute(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


def get_avg_pool():
    """获取平均池化函数，Jittor版本直接返回adaptive_avg_pool2d"""
    return lambda x, size: jt.pool.AdaptiveAvgPool2d(size)(x)


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = lambda x, size: jt.pool.AdaptiveAvgPool2d(size)(x)
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d

    def execute(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, 'pool'):
            self.pool = lambda x, size: jt.pool.AdaptiveAvgPool2d(size)(x)

        # Jittor没有ONNX导出检查，直接使用onnx_AdaptiveAvgPool2d
        self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return jt.concat(out, dim=1)


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(self, inp: int, oup: int, norm_cfg=dict(type='BN', requires_grad=True),
                 activations=None, global_inp=None):
        super().__init__()
        self.norm_cfg = norm_cfg

        if not global_inp:
            global_inp = inp

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def execute(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
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
            sig_act = nn.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = nn.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out
