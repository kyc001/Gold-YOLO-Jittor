# 2023.09.18-Implement the model layers for Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
GOLD-YOLO Jittor版本 - 基础层模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import jittor as jt
import jittor.nn as nn


class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        # 添加卷积层
        conv = nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        self.add_module('c', conv)
        
        # 添加BatchNorm层
        bn = nn.BatchNorm2d(b)
        # Jittor的初始化方式
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)  # Jittor使用jt.rand
    random_tensor = jt.floor(random_tensor)  # binarize，Jittor使用jt.floor
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def execute(self, x):
        return drop_path(x, self.drop_prob, self.training)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()  # Jittor的ReLU6不需要inplace参数
    
    def execute(self, x):
        return self.relu(x + 3) / 6


class JittorBatchNorm2d(nn.Module):
    """Jittor版本的BatchNorm2d，兼容mmcv的build_norm_layer接口"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
    
    def execute(self, x):
        return self.bn(x)
    
    @property
    def weight(self):
        return self.bn.weight
    
    @property
    def bias(self):
        return self.bn.bias
    
    @property
    def running_mean(self):
        return self.bn.running_mean
    
    @property
    def running_var(self):
        return self.bn.running_var


def build_norm_layer(cfg, num_features, postfix=''):
    """构建归一化层，兼容mmcv接口"""
    if cfg is None:
        return None, None

    cfg_type = cfg.get('type', 'BN') if isinstance(cfg, dict) else cfg

    if cfg_type in ['BN', 'BN2d']:
        layer = JittorBatchNorm2d(num_features)
        name = 'bn' + postfix
    elif cfg_type == 'SyncBN':
        # Jittor中SyncBN就是BatchNorm2d
        layer = JittorBatchNorm2d(num_features)
        name = 'bn' + postfix
    elif cfg_type == 'GN':
        # GroupNorm - 使用Jittor的GroupNorm
        num_groups = cfg.get('num_groups', 32) if isinstance(cfg, dict) else 32
        layer = nn.GroupNorm(num_groups, num_features)
        name = 'gn' + postfix
    else:
        # 默认使用BatchNorm2d
        print(f"⚠️  未知的norm类型 {cfg_type}，使用BatchNorm2d替代")
        layer = JittorBatchNorm2d(num_features)
        name = 'bn' + postfix

    return name, layer


class ConvModule(nn.Module):
    """Jittor版本的ConvModule，兼容mmcv接口"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        
        # 卷积层
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias
        )
        
        # 归一化层
        self.norm = None
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        
        # 激活层
        self.act = None
        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.act = nn.ReLU()
            elif act_cfg['type'] == 'SiLU':
                self.act = nn.SiLU()
            elif act_cfg['type'] == 'ReLU6':
                self.act = nn.ReLU6()
    
    def execute(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Identity(nn.Module):
    """恒等映射层"""
    
    def __init__(self):
        super().__init__()
    
    def execute(self, x):
        return x


# 为了兼容性，创建一些常用的激活函数别名
class ReLU(nn.ReLU):
    pass


class ReLU6(nn.ReLU6):
    pass


class SiLU(nn.SiLU):
    pass


class GELU(nn.GELU):
    pass
