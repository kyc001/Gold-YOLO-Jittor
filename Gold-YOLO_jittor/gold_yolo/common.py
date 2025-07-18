# Jittor implementation of Gold-YOLO common components
# Migrated from PyTorch version

import numpy as np
import jittor as jt
from jittor import nn
import jittor.nn as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.common import SimConv
from gold_yolo.transformer import jittor_adaptive_avg_pool2d


class AdvPoolFusion(nn.Module):
    """Advanced pooling fusion"""
    
    def execute(self, x1, x2):
        self.pool = jittor_adaptive_avg_pool2d
        
        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)
        
        return jt.concat([x1, x2], dim=1)


class SimFusion_3in(nn.Module):
    """Simple fusion for 3 inputs"""
    
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channel_list[0], out_channels, 1, 1)
        self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)
        self.downsample = jittor_adaptive_avg_pool2d
    
    def execute(self, x):
        N, C, H, W = x[1].shape
        output_size = np.array([H, W])
        
        x0 = self.downsample(x[0], output_size)
        x1 = self.cv1(x[1])
        x2 = jt.nn.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(jt.concat((x0, x1, x2), dim=1))


class SimFusion_4in(nn.Module):
    """Simple fusion for 4 inputs"""
    
    def __init__(self):
        super().__init__()
        self.avg_pool = jittor_adaptive_avg_pool2d
    
    def execute(self, x):
        # Handle different numbers of inputs
        if len(x) == 5:
            # Use all 5 features: c2, c3, c4, c5, c6
            x_c2, x_c3, x_c4, x_c5, x_c6 = x
            # Use c4 as the reference size (typically 40x40)
            B, C, H, W = x_c4.shape
            output_size = np.array([H, W])

            # Resize all features to the same size
            x_c2 = self.avg_pool(x_c2, output_size)  # 160x160 -> 40x40
            x_c3 = self.avg_pool(x_c3, output_size)  # 80x80 -> 40x40
            # x_c4 stays the same (40x40)
            x_c5 = self.avg_pool(x_c5, output_size)  # 20x20 -> 40x40 (upsampled)
            x_c6 = jt.nn.interpolate(x_c6, size=(H, W), mode='bilinear', align_corners=False)  # 10x10 -> 40x40

            out = jt.concat([x_c2, x_c3, x_c4, x_c5, x_c6], dim=1)
        else:
            # Original 4-input version
            x_l, x_m, x_s, x_n = x
            B, C, H, W = x_s.shape
            output_size = np.array([H, W])

            x_l = self.avg_pool(x_l, output_size)
            x_m = self.avg_pool(x_m, output_size)
            x_n = jt.nn.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

            out = jt.concat([x_l, x_m, x_s, x_n], dim=1)
        return out
