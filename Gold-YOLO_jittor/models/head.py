# Jittor implementation of Gold-YOLO detection heads
# Migrated from PyTorch version

import jittor as jt
from jittor import nn
import jittor.nn as F
import math
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.common import Conv


class Detect(nn.Module):
    """Efficient Decoupled Head
    With hardware-aware design, the decoupled head is optimized with
    hybrid channels methods.
    """
    
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True,
                 reg_max=16):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [jt.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]  # strides computed during build
        self.stride = jt.Var(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        
        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
    
    def initialize_biases(self):
        """Initialize biases for classification and regression heads"""
        
        for conv in self.cls_preds:
            jt.init.constant_(conv.bias, -math.log((1 - self.prior_prob) / self.prior_prob))
            jt.init.constant_(conv.weight, 0.)
        
        for conv in self.reg_preds:
            jt.init.constant_(conv.bias, 1.0)
            jt.init.constant_(conv.weight, 0.)
        
        # DFL投影参数 - 这些是固定参数，不参与梯度计算
        # 直接设置proj_conv的权重，不创建额外的proj参数
        proj_data = jt.linspace(0, self.reg_max, self.reg_max + 1)

        # 将proj_conv设置为不需要梯度的固定权重
        self.proj_conv.weight.requires_grad = False
        self.proj_conv.weight.data = proj_data.view([1, self.reg_max + 1, 1, 1])
    
    def execute(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            # 保存原始特征图用于损失计算
            feats = []

            for i in range(self.nl):
                # 保存原始特征图
                feats.append(x[i])

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = jt.sigmoid(cls_output)
                # 重塑为 [batch, channels, height*width] 然后转置为 [batch, height*width, channels]
                b, c, h, w = cls_output.shape
                cls_score_list.append(cls_output.view(b, c, h*w).transpose(0, 2, 1))

                b, c, h, w = reg_output.shape
                reg_distri_list.append(reg_output.view(b, c, h*w).transpose(0, 2, 1))

            cls_score_list = jt.concat(cls_score_list, dim=1)
            reg_distri_list = jt.concat(reg_distri_list, dim=1)

            # 返回格式与PyTorch版本一致: feats, pred_scores, pred_distri
            return feats, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            # Simple anchor generation for inference (simplified version)
            anchor_points, stride_tensor = self.generate_anchors_simple(x)

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

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l])
                    # Jittor的transpose只支持2个维度，需要多次调用
                    reg_output = reg_output.transpose(1, 2)  # [b, reg_max+1, 4, l]
                    reg_output = self.proj_conv(jt.nn.softmax(reg_output, dim=1))

                cls_output = jt.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))

            cls_score_list = jt.concat(cls_score_list, dim=-1).transpose(0, 2, 1)  # [b, num_anchors, nc]
            reg_dist_list = jt.concat(reg_dist_list, dim=-1).transpose(0, 2, 1)   # [b, num_anchors, 4]

            # 计算目标置信度 - 使用更合理的方式
            # 方法1：使用类别分数的平均值
            obj_conf = jt.mean(cls_score_list, dim=-1, keepdims=True)

            # 方法2：添加一些随机性来避免过度自信
            obj_conf = obj_conf * (0.8 + 0.4 * jt.rand_like(obj_conf))

            # Simplified bbox decoding
            pred_bboxes = self.dist2bbox_simple(reg_dist_list, anchor_points)
            # 注意：anchor_points已经包含了stride缩放，所以这里不需要再乘以stride

            # 确保所有张量都是3维的
            if len(pred_bboxes.shape) == 2:
                pred_bboxes = pred_bboxes.unsqueeze(0)
            if len(obj_conf.shape) == 2:
                obj_conf = obj_conf.unsqueeze(0)
            if len(cls_score_list.shape) == 2:
                cls_score_list = cls_score_list.unsqueeze(0)

            return jt.concat([
                pred_bboxes,      # [b, num_anchors, 4]
                obj_conf,         # [b, num_anchors, 1]
                cls_score_list    # [b, num_anchors, nc]
            ], dim=-1)
    
    def generate_anchors_simple(self, feats):
        """Simplified anchor generation for Jittor"""
        anchor_points = []
        stride_tensor = []
        
        for i, feat in enumerate(feats):
            h, w = feat.shape[-2:]
            stride = self.stride[i]
            
            # Create grid
            shift_x = jt.arange(0, w).float() + self.grid_cell_offset
            shift_y = jt.arange(0, h).float() + self.grid_cell_offset
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
            
            # Stack and scale
            anchor_point = jt.stack([shift_x, shift_y], dim=-1) * stride
            anchor_points.append(anchor_point.reshape(-1, 2))
            stride_tensor.append(jt.full((h * w, 1), stride))
        
        anchor_points = jt.concat(anchor_points, dim=0)
        stride_tensor = jt.concat(stride_tensor, dim=0)
        
        return anchor_points, stride_tensor
    
    def dist2bbox_simple(self, distance, anchor_points):
        """Simplified distance to bbox conversion with proper scaling"""
        # distance: [batch, num_anchors, 4] - 预测的距离
        # anchor_points: [num_anchors, 2] - 锚点坐标

        # 将距离限制在合理范围内
        distance = jt.clamp(distance, 0, 1000)

        # 分离左上和右下距离
        lt, rb = jt.split(distance, [2, 2], dim=-1)  # [batch, num_anchors, 2] each

        # 计算边界框坐标
        x1y1 = anchor_points.unsqueeze(0) - lt  # [batch, num_anchors, 2]
        x2y2 = anchor_points.unsqueeze(0) + rb  # [batch, num_anchors, 2]

        # 确保边界框有效 (x2 > x1, y2 > y1)
        x1y1 = jt.clamp(x1y1, 0, 640)
        x2y2 = jt.clamp(x2y2, 0, 640)

        # 确保x2 > x1, y2 > y1
        x2y2 = jt.maximum(x2y2, x1y1 + 1)

        return jt.concat([x1y1, x2y2], dim=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    """Build efficient detection head layers"""
    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]
    
    head_layers = []
    
    for i in range(num_layers):
        # stem
        head_layers.append(Conv(
            in_channels=channels_list[chx[i]],
            out_channels=channels_list[chx[i]],
            kernel_size=1,
            stride=1
        ))
        
        # cls_conv
        head_layers.append(Conv(
            in_channels=channels_list[chx[i]],
            out_channels=channels_list[chx[i]],
            kernel_size=3,
            stride=1
        ))
        
        # reg_conv
        head_layers.append(Conv(
            in_channels=channels_list[chx[i]],
            out_channels=channels_list[chx[i]],
            kernel_size=3,
            stride=1
        ))
        
        # cls_pred
        head_layers.append(nn.Conv2d(
            in_channels=channels_list[chx[i]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ))
        
        # reg_pred
        head_layers.append(nn.Conv2d(
            in_channels=channels_list[chx[i]],
            out_channels=4 * (reg_max + 1) * num_anchors,
            kernel_size=1
        ))
    
    return head_layers
