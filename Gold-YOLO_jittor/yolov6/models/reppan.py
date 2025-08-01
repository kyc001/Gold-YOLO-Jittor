"""
GOLD-YOLO Jittor版本 - RepPAN Neck
从PyTorch版本迁移到Jittor框架
"""

import jittor.nn as nn
from yolov6.layers.common import *


class RepPANNeck(nn.Module):
    """Rep-PAN
    The default neck of YOLOv6.
    """
    
    def __init__(
            self,
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,
            extra_cfg=None
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        self.Rep_p4 = RepBlock(
                in_channels=channels_list[3] + channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                block=block,
        )
        
        self.Rep_p3 = RepBlock(
                in_channels=channels_list[2] + channels_list[6],
                out_channels=channels_list[6],
                n=num_repeats[6],
                block=block,
        )
        
        self.Rep_n3 = RepBlock(
                in_channels=channels_list[6] + channels_list[7],
                out_channels=channels_list[7],
                n=num_repeats[7],
                block=block,
        )
        
        self.Rep_n4 = RepBlock(
                in_channels=channels_list[5] + channels_list[8],
                out_channels=channels_list[8],
                n=num_repeats[8],
                block=block,
        )
        
        self.reduce_layer0 = ConvWrapper(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=1,
                stride=1,
        )
        
        self.upsample0 = nn.Upsample(scale_factor=2)
        self.reduce_layer1 = ConvWrapper(
                in_channels=channels_list[5],
                out_channels=channels_list[6],
                kernel_size=1,
                stride=1,
        )
        
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.downsample2 = ConvWrapper(
                in_channels=channels_list[6],
                out_channels=channels_list[7],
                kernel_size=3,
                stride=2,
        )
        
        self.downsample1 = ConvWrapper(
                in_channels=channels_list[7],
                out_channels=channels_list[8],
                kernel_size=3,
                stride=2,
        )
    
    def execute(self, input):
        """Jittor版本的前向传播"""
        (x2, x1, x0) = input
        
        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = jt.concat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)
        
        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = jt.concat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)
        
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = jt.concat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)
        
        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = jt.concat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)
        
        outputs = [pan_out2, pan_out1, pan_out0]
        
        return outputs


class CSPRepPANNeck(nn.Module):
    """CSP-RepPAN
    """
    
    def __init__(
            self,
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,
            csp_e=float(1) / 2,
            extra_cfg=None
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        self.Rep_p4 = BepC3(
                in_channels=channels_list[3] + channels_list[5],
                out_channels=channels_list[5],
                n=num_repeats[5],
                e=csp_e,
                block=block,
        )
        
        self.Rep_p3 = BepC3(
                in_channels=channels_list[2] + channels_list[6],
                out_channels=channels_list[6],
                n=num_repeats[6],
                e=csp_e,
                block=block,
        )
        
        self.Rep_n3 = BepC3(
                in_channels=channels_list[6] + channels_list[7],
                out_channels=channels_list[7],
                n=num_repeats[7],
                e=csp_e,
                block=block,
        )
        
        self.Rep_n4 = BepC3(
                in_channels=channels_list[5] + channels_list[8],
                out_channels=channels_list[8],
                n=num_repeats[8],
                e=csp_e,
                block=block,
        )
        
        self.reduce_layer0 = ConvWrapper(
                in_channels=channels_list[4],
                out_channels=channels_list[5],
                kernel_size=1,
                stride=1,
        )
        
        self.upsample0 = nn.Upsample(scale_factor=2)
        self.reduce_layer1 = ConvWrapper(
                in_channels=channels_list[5],
                out_channels=channels_list[6],
                kernel_size=1,
                stride=1,
        )
        
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.downsample2 = ConvWrapper(
                in_channels=channels_list[6],
                out_channels=channels_list[7],
                kernel_size=3,
                stride=2,
        )
        
        self.downsample1 = ConvWrapper(
                in_channels=channels_list[7],
                out_channels=channels_list[8],
                kernel_size=3,
                stride=2,
        )
    
    def execute(self, input):
        """Jittor版本的前向传播"""
        (x2, x1, x0) = input
        
        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = jt.concat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)
        
        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = jt.concat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)
        
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = jt.concat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)
        
        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = jt.concat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)
        
        outputs = [pan_out2, pan_out1, pan_out0]
        
        return outputs
