#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
SimpleRepPAN - 严格对齐PyTorch版本的RepPANNeck
新芽第二阶段：使用简单的RepPAN替代复杂的RepGDNeck
"""

import jittor as jt
import jittor.nn as nn
from ..layers.common import *


class SimpleRepPAN(nn.Module):
    """
    SimpleRepPAN - 严格对齐PyTorch版本的RepPANNeck
    这是一个简单的RepPAN实现，没有复杂的LAF、IFM等组件
    """
    
    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        print(f"🔧 构建SimpleRepPAN: channels={channels_list}, repeats={num_repeats}")
        
        # 确保有足够的通道数和重复次数
        if len(channels_list) < 11:
            # 扩展channels_list以匹配PyTorch版本的索引
            extended_channels = channels_list + [channels_list[-1]] * (11 - len(channels_list))
            channels_list = extended_channels
            
        if len(num_repeats) < 9:
            # 扩展num_repeats以匹配PyTorch版本的索引
            extended_repeats = num_repeats + [num_repeats[-1]] * (9 - len(num_repeats))
            num_repeats = extended_repeats
        
        # 严格按照PyTorch版本的RepPANNeck实现
        # Rep_p4: channels_list[3] + channels_list[5] -> channels_list[5]
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=block
        )

        # Rep_p3: channels_list[2] + channels_list[6] -> channels_list[6]
        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=block
        )

        # Rep_n3: channels_list[6] + channels_list[7] -> channels_list[8]
        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=block
        )

        # Rep_n4: channels_list[5] + channels_list[9] -> channels_list[10]
        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=block
        )
        
        # 下采样和上采样层
        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1
        )
        
        self.upsample0 = Transpose(
            in_channels=channels_list[5],
            out_channels=channels_list[5]
        )
        
        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        )
        
        self.upsample1 = Transpose(
            in_channels=channels_list[6],
            out_channels=channels_list[6]
        )
        
        self.downsample2 = SimConv(
            in_channels=channels_list[6],
            out_channels=channels_list[7],
            kernel_size=3,
            stride=2
        )
        
        self.downsample1 = SimConv(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2
        )
        
        print(f"🔧 SimpleRepPAN构建完成")
        
    def execute(self, inputs):
        """前向传播 - 严格对齐PyTorch版本"""
        # 处理EfficientRep的5个输出，取后3个用于neck
        if len(inputs) == 5:
            # EfficientRep输出: [P1, P2, P3, P4, P5]
            # 我们需要: [P3, P4, P5] 对应 [x2, x1, x0]
            x2, x1, x0 = inputs[2], inputs[3], inputs[4]
        elif len(inputs) == 3:
            # 如果是3个输入，直接使用
            x2, x1, x0 = inputs[0], inputs[1], inputs[2]
        else:
            # 处理其他情况
            x2 = inputs[-3] if len(inputs) >= 3 else inputs[-1]
            x1 = inputs[-2] if len(inputs) >= 2 else inputs[-1]
            x0 = inputs[-1]
        
        # 上采样路径
        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = jt.concat([upsample_feat0, x1], dim=1)
        f_out0 = self.Rep_p4(f_concat_layer0)
        
        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = jt.concat([upsample_feat1, x2], dim=1)
        f_out1 = self.Rep_p3(f_concat_layer1)
        
        # 下采样路径
        downsample_feat1 = self.downsample2(f_out1)
        p_concat_layer1 = jt.concat([downsample_feat1, fpn_out1], dim=1)
        p_out1 = self.Rep_n3(p_concat_layer1)
        
        downsample_feat0 = self.downsample1(p_out1)
        p_concat_layer2 = jt.concat([downsample_feat0, fpn_out0], dim=1)
        p_out0 = self.Rep_n4(p_concat_layer2)
        
        return [f_out1, p_out1, p_out0]


class CSPRepPAN(SimpleRepPAN):
    """
    CSPRepPAN - 用于M/L版本的CSP版本RepPAN
    """
    
    def __init__(self, channels_list=None, num_repeats=None, block=RepVGGBlock, csp_e=0.5, extra_cfg=None):
        # 忽略extra_cfg，使用简单的RepPAN结构
        super().__init__(channels_list, num_repeats, block)
        
        self.csp_e = csp_e
        print(f"🔧 构建CSPRepPAN: csp_e={csp_e}")
