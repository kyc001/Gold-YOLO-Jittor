# 2023.09.18-Implement the model layers for Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
GOLD-YOLO Jittor版本 - 通用模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import numpy as np
import jittor as jt
import jittor.nn as nn

from yolov6.layers.common import SimConv
from .transformer import onnx_AdaptiveAvgPool2d


class AdvPoolFusion(nn.Module):
    def execute(self, x1, x2):
        # Jittor直接使用adaptive_avg_pool2d
        N, C, H, W = x2.shape
        output_size = (H, W)  # Jittor使用tuple而不是numpy array
        x1 = jt.pool.AdaptiveAvgPool2d(output_size)(x1)

        return jt.concat([x1, x2], 1)  # Jittor使用concat替代cat


class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        # 为每个输入创建转换层，确保输出通道数一致
        # in_channel_list应该有3个元素，对应3个输入
        if len(in_channel_list) == 2:
            # 兼容旧的配置格式，假设前两个输入通道数相同
            in_channels = [in_channel_list[0], in_channel_list[0], in_channel_list[1]]
        else:
            in_channels = in_channel_list

        self.cv0 = SimConv(in_channels[0], out_channels, 1, 1)  # 处理x[0]
        self.cv1 = SimConv(in_channels[1], out_channels, 1, 1)  # 处理x[1]
        self.cv2 = SimConv(in_channels[2], out_channels, 1, 1)  # 处理x[2]
        self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)
        self.downsample = lambda x, size: jt.pool.AdaptiveAvgPool2d(size)(x)

    def execute(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)

        # 处理三个输入，确保它们都有相同的通道数
        x0 = self.downsample(x[0], output_size)
        x0 = self.cv0(x0)  # 转换通道数

        x1 = self.cv1(x[1])  # 转换通道数

        x2 = nn.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        x2 = self.cv2(x2)  # 转换通道数

        return self.cv_fuse(jt.concat((x0, x1, x2), dim=1))


class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = lambda x, size: jt.pool.AdaptiveAvgPool2d(size)(x)
    
    def execute(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = (H, W)  # Jittor使用tuple
        
        # Jittor没有ONNX导出检查，直接使用adaptive_avg_pool2d
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = nn.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = jt.concat([x_l, x_m, x_s, x_n], 1)
        return out


class JittorAdaptiveAvgPool2d(nn.Module):
    """Jittor版本的自适应平均池化，兼容不同输入格式"""
    
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size
    
    def execute(self, x, output_size=None):
        if output_size is None:
            output_size = self.output_size
        
        # 处理不同的输入格式
        if isinstance(output_size, np.ndarray):
            output_size = tuple(output_size.tolist())
        elif isinstance(output_size, (list, tuple)):
            output_size = tuple(output_size)
        
        return jt.pool.AdaptiveAvgPool2d(output_size)(x)


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def execute(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def execute(self, x):
        avg_out = jt.mean(x, dim=1, keepdim=True)
        max_out = jt.max(x, dim=1, keepdim=True)[0]
        x_cat = jt.concat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def execute(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def execute(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeatureFusion(nn.Module):
    """特征融合模块"""
    
    def __init__(self, in_channels_list, out_channels, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            total_channels = sum(in_channels_list)
            self.fusion_conv = nn.Conv2d(total_channels, out_channels, 1, 1, 0)
        elif fusion_type == 'add':
            # 确保所有输入通道数相同
            assert all(c == in_channels_list[0] for c in in_channels_list)
            if in_channels_list[0] != out_channels:
                self.fusion_conv = nn.Conv2d(in_channels_list[0], out_channels, 1, 1, 0)
            else:
                self.fusion_conv = nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def execute(self, features):
        # 将所有特征调整到相同尺寸
        target_size = features[0].shape[2:]
        aligned_features = []
        
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = nn.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        if self.fusion_type == 'concat':
            fused = jt.concat(aligned_features, dim=1)
        elif self.fusion_type == 'add':
            fused = sum(aligned_features)
        
        fused = self.fusion_conv(fused)
        fused = self.bn(fused)
        fused = self.act(fused)
        
        return fused
