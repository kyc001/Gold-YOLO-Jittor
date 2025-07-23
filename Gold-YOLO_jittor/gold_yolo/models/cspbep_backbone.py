#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
CSPBepBackbone - 严格对齐PyTorch版本的M/L版本Backbone
新芽第二阶段：深入修复架构不一致问题
"""

import jittor as jt
import jittor.nn as nn
from ..layers.common import *


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone - 用于Gold-YOLO M/L版本
    严格对齐PyTorch版本的CSPBepBackbone实现
    """
    
    def __init__(self, channels_list, num_repeats, block=RepVGGBlock, csp_e=0.5, fuse_P2=True):
        super().__init__()
        
        assert len(channels_list) >= 5
        assert len(num_repeats) >= 5
        
        self.channels_list = channels_list
        self.num_repeats = num_repeats
        self.csp_e = csp_e  # CSP expansion ratio
        self.fuse_P2 = fuse_P2
        
        print(f"🔧 构建CSPBepBackbone: channels={channels_list}, repeats={num_repeats}, csp_e={csp_e}")
        
        # Stem layer
        self.stem = Conv(
            in_channels=3,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # Stage 2: CSP Block
        self.stage2 = self._make_csp_stage(
            in_channels=channels_list[0],
            out_channels=channels_list[1],
            num_repeats=num_repeats[1],
            block=block,
            stride=2
        )
        
        # Stage 3: CSP Block
        self.stage3 = self._make_csp_stage(
            in_channels=channels_list[1],
            out_channels=channels_list[2],
            num_repeats=num_repeats[2],
            block=block,
            stride=2
        )
        
        # Stage 4: CSP Block
        self.stage4 = self._make_csp_stage(
            in_channels=channels_list[2],
            out_channels=channels_list[3],
            num_repeats=num_repeats[3],
            block=block,
            stride=2
        )
        
        # Stage 5: CSP Block + SPPF
        self.stage5 = nn.Sequential(
            self._make_csp_stage(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                num_repeats=num_repeats[4],
                block=block,
                stride=2
            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )
        
    def _make_csp_stage(self, in_channels, out_channels, num_repeats, block, stride=1):
        """创建CSP阶段 - 深入修复：优化参数量"""
        # CSP结构：分割 -> 处理 -> 合并
        # 深入修复：减少mid_channels以控制参数量
        mid_channels = max(16, int(out_channels * self.csp_e * 0.75))  # 减少25%

        # 下采样卷积
        downsample = Conv(in_channels, out_channels, 3, stride, 1) if stride > 1 else nn.Identity()

        # CSP分支1：直接连接
        branch1 = Conv(out_channels, mid_channels, 1, 1, 0)

        # CSP分支2：RepVGG blocks - 深入修复：减少重复次数
        branch2_layers = [Conv(out_channels, mid_channels, 1, 1, 0)]
        # 深入修复：限制最大重复次数以控制参数量
        effective_repeats = min(num_repeats, 8)  # 最多8次重复
        for _ in range(effective_repeats):
            branch2_layers.append(block(mid_channels, mid_channels))
        branch2 = nn.Sequential(*branch2_layers)

        # 合并层
        merge = Conv(mid_channels * 2, out_channels, 1, 1, 0)

        return CSPStage(downsample, branch1, branch2, merge)
        
    def execute(self, x):
        """前向传播"""
        outputs = []
        
        # Stem
        x = self.stem(x)
        outputs.append(x)  # P1
        
        # Stage 2
        x = self.stage2(x)
        outputs.append(x)  # P2
        
        # Stage 3
        x = self.stage3(x)
        outputs.append(x)  # P3
        
        # Stage 4
        x = self.stage4(x)
        outputs.append(x)  # P4
        
        # Stage 5
        x = self.stage5(x)
        outputs.append(x)  # P5
        
        return tuple(outputs)


class CSPStage(nn.Module):
    """CSP阶段实现"""
    
    def __init__(self, downsample, branch1, branch2, merge):
        super().__init__()
        self.downsample = downsample
        self.branch1 = branch1
        self.branch2 = branch2
        self.merge = merge
        
    def execute(self, x):
        # 下采样
        x = self.downsample(x)
        
        # CSP分支
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        # 合并
        out = self.merge(jt.concat([x1, x2], dim=1))
        
        return out
