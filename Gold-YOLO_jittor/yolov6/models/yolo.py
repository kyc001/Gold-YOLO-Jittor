# 2023.09.18-Changed for yolo model of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本主模型文件
从PyTorch版本迁移到Jittor框架
"""

import math
import jittor as jt
import jittor.nn as nn

from yolov6.layers.common import *
from yolov6.utils.jittor_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER


class Model(nn.Module):
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False,
                 distill_ns=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, num_layers,
                                                              fuse_ab=fuse_ab, distill_ns=distill_ns)
        
        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()
        
        # Init weights
        initialize_weights(self)
    
    def execute(self, x):
        """Jittor版本的前向传播，使用execute替代forward"""
        # Jittor不支持ONNX导出检查，简化逻辑
        export_mode = False
        x = self.backbone(x)
        x = self.neck(x)
        if export_mode == False:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode is True else [x, featmaps]
    
    def _apply(self, fn):
        """Jittor版本的_apply方法"""
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    """Upward revision the value x to make it evenly divisible by the divisor."""
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    """构建网络的主函数，从配置文件构建backbone、neck和head"""
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]
    
    # 获取backbone和neck的额外配置
    block = get_block(config.model.backbone.type)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    neck_extra_cfg = getattr(config.model.neck, 'extra_cfg', None)
    
    if 'CSP' in config.model.backbone.type:
        backbone = BACKBONE(
                in_channels=channels,
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                csp_e=config.model.backbone.csp_e,
                fuse_P2=fuse_P2,
                cspsppf=cspsppf
        )
        
        neck = NECK(
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                csp_e=config.model.neck.csp_e,
                extra_cfg=neck_extra_cfg
        )
    else:
        backbone = BACKBONE(
                in_channels=channels,
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                fuse_P2=fuse_P2,
                cspsppf=cspsppf
        )
        
        neck = NECK(
                channels_list=channels_list,
                num_repeats=num_repeat,
                block=block,
                extra_cfg=neck_extra_cfg
        )
    
    # 构建检测头
    if distill_ns:
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)
    
    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)
    
    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)
    
    return backbone, neck, head


def get_block(block_name):
    """根据block名称获取对应的block类"""
    if block_name == 'RepVGGBlock':
        from yolov6.layers.common import RepVGGBlock
        return RepVGGBlock
    elif block_name == 'ConvWrapper':
        from yolov6.layers.common import ConvWrapper
        return ConvWrapper
    else:
        raise NotImplementedError(f"Block {block_name} not implemented")


def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False):
    """构建完整模型的函数"""
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns)
    # Jittor中不需要.to(device)，会自动处理设备
    return model
