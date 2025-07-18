# Jittor implementation of Gold-YOLO model
# Migrated from PyTorch version

import jittor as jt
from jittor import nn
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import EfficientRep, CSPBepBackbone
from models.neck import RepGDNeck
from models.head import Detect, build_effidehead_layer
from layers.common import RepVGGBlock


class Model(nn.Module):
    """Gold-YOLO model with backbone, neck and head.
    The default parts are EfficientRep Backbone, RepGDNeck and
    Efficient Decoupled Head.
    """
    
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False, distill_ns=False):
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, num_layers,
                                                              fuse_ab=fuse_ab, distill_ns=distill_ns)
        
        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()
        
        # Init weights
        self.initialize_weights()
    
    def execute(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        featmaps = []
        featmaps.extend(x)
        x = self.detect(x)

        # 训练时返回 [检测输出, 特征图]，推理时只返回检测输出
        # 这与PyTorch版本保持一致
        if self.training:
            return [x, featmaps]
        else:
            return x
    
    def initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)


def make_divisible(x, divisor):
    """Upward revision the value x to make it evenly divisible by the divisor."""
    return math.ceil(x / divisor) * divisor


def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    """Build Gold-YOLO network"""
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
    
    block = RepVGGBlock  # Use RepVGGBlock as default
    
    # Build backbone
    if config.model.backbone.type == 'EfficientRep':
        backbone = EfficientRep(
            in_channels=channels,
            channels_list=channels_list[:6],
            num_repeats=num_repeat[:6],
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )
    elif config.model.backbone.type == 'CSPBepBackbone':
        backbone = CSPBepBackbone(
            in_channels=channels,
            channels_list=channels_list[:6],
            num_repeats=num_repeat[:6],
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )
    else:
        raise ValueError(f"Unsupported backbone type: {config.model.backbone.type}")
    
    # Build neck
    neck_extra_cfg = config.model.neck.extra_cfg if hasattr(config.model.neck, 'extra_cfg') else None
    
    if config.model.neck.type == 'RepGDNeck':
        neck = RepGDNeck(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            extra_cfg=neck_extra_cfg
        )
    else:
        raise ValueError(f"Unsupported neck type: {config.model.neck.type}")
    
    # Build head
    if fuse_ab:
        # Use fused head (simplified for now)
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)
    else:
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)
    
    return backbone, neck, head


def build_model(cfg, num_classes, fuse_ab=False, distill_ns=False):
    """Build Gold-YOLO model"""
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns)
    return model
