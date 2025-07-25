"""
GOLD-YOLO Jittor版本 - Efficient Decoupled Head
从PyTorch版本迁移到Jittor框架
"""

import jittor as jt
import jittor.nn as nn
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware design, the decoupled head is optimized with
    hybridchannels methods.
    '''
    
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True,
                 reg_max=16):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [jt.zeros(1)] * num_layers  # 使用jt.zeros替代torch.zeros
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]  # strides computed during build
        self.stride = jt.array(stride)  # 使用jt.array替代torch.tensor
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        # 只在use_dfl=True时创建proj_conv，避免无用参数
        if self.use_dfl:
            self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        
        # 构建检测头层
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # 从head_layers中提取各层
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
    
    def initialize_biases(self):
        """初始化偏置"""
        for conv in self.cls_preds:
            # Jittor使用不同的初始化方式
            bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            conv.bias.data = jt.full_like(conv.bias.data, bias_value)

        for conv in self.reg_preds:
            # Jittor使用不同的初始化方式
            conv.bias.data = jt.ones_like(conv.bias.data)
        
        if self.use_dfl and hasattr(self, 'proj_conv'):
            # 使用更好的初始化，避免梯度消失
            nn.init.xavier_uniform_(self.proj_conv.weight)
    
    def execute(self, x):
        """Jittor版本的前向传播"""
        if self.training:
            cls_score_list = []
            reg_distri_list = []
            
            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                
                cls_output = jt.sigmoid(cls_output)  # 使用jt.sigmoid
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
            
            return jt.concat(cls_score_list, dim=1), jt.concat(reg_distri_list, dim=1)
        
        else:
            cls_score_list = []
            reg_dist_list = []
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=None, is_eval=True, mode='af')
            
            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                
                if self.use_dfl and hasattr(self, 'proj_conv'):
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(nn.softmax(reg_output, dim=1))
                
                cls_output = jt.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            cls_score_list = jt.concat(cls_score_list, dim=-1).permute(0, 2, 1)
            reg_dist_list = jt.concat(reg_dist_list, dim=-1).permute(0, 2, 1)
            
            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xyxy')
            pred_bboxes *= stride_tensor
            return jt.concat([pred_bboxes, cls_score_list], dim=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    """构建EffiDeHead层 - channels_list是head的输入通道列表，如[32, 64, 128]"""

    head_layers = []

    # 为每个检测层创建5个模块：stem, cls_conv, reg_conv, cls_pred, reg_pred
    for i in range(num_layers):
        ch = channels_list[i]  # 当前层的输入通道数

        # stem
        head_layers.append(Conv(
            in_channels=ch,
            out_channels=ch,
            kernel_size=1,
            stride=1
        ))

        # cls_conv
        head_layers.append(Conv(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            stride=1
        ))

        # reg_conv
        head_layers.append(Conv(
            in_channels=ch,
            out_channels=ch,
            kernel_size=3,
            stride=1
        ))

        # cls_pred
        head_layers.append(nn.Conv2d(
            in_channels=ch,
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ))

        # reg_pred
        head_layers.append(nn.Conv2d(
            in_channels=ch,
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ))

    return head_layers
