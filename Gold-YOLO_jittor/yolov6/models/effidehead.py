#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Efficient Decoupled Head for Gold-YOLO - Jittor Implementation
Aligned with PyTorch version
"""

import jittor as jt
from jittor import nn
import math
import numpy as np

from yolov6.layers.common import Conv
from yolov6.utils.general import dist2bbox, generate_anchors


class Detect(nn.Module):
    """Efficient Decoupled Head - 完全对齐PyTorch版本"""
    
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes
        self.no = num_classes + 5
        self.nl = num_layers
        self.grid = [jt.zeros(1)] * num_layers  # 使用jt.zeros
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]
        self.stride = jt.array(stride)  # 使用jt.array
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)  # 正确的参数顺序

        # 初始化proj_conv权重 - 完全对齐PyTorch版本
        proj_weight = jt.linspace(0, self.reg_max, self.reg_max + 1).view(1, self.reg_max + 1, 1, 1)
        self.proj_conv.weight = proj_weight
        self.proj_conv.weight.requires_grad = False
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
            idx = i * 5  # PyTorch版本使用5而不是6
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
    
    def initialize_biases(self):
        """正确的Jittor参数初始化方式"""
        for conv in self.cls_preds:
            # Jittor方式：直接修改.data
            bias_val = -math.log((1 - self.prior_prob) / self.prior_prob)
            conv.bias.data = jt.full_like(conv.bias, bias_val)
            conv.weight.data = jt.zeros_like(conv.weight)
        
        for conv in self.reg_preds:
            # Jittor方式：直接修改.data
            conv.bias.data = jt.ones_like(conv.bias)
            conv.weight.data = jt.zeros_like(conv.weight)
        
        # 初始化投影层 - Jittor方式
        proj_data = jt.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj_conv.weight.data = proj_data.view(1, self.reg_max + 1, 1, 1)
    
    def execute(self, x):
        """前向传播 - 完全对齐PyTorch版本"""
        if self.training:
            # 训练模式：返回特征用于损失计算
            cls_score_list = []
            reg_distri_list = []
            
            for i in range(self.nl):
                # 保存原始输入，避免修改影响计算图
                stem_out = self.stems[i](x[i])
                x[i] = stem_out  # 更新x[i]用于返回

                # 使用stem输出进行分支计算
                cls_feat = self.cls_convs[i](stem_out)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](stem_out)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = jt.sigmoid(cls_output)
                # 使用PyTorch相同的操作：flatten(2).permute((0, 2, 1))
                cls_score_list.append(cls_output.flatten(2).permute(0, 2, 1))
                reg_distri_list.append(reg_output.flatten(2).permute(0, 2, 1))
            
            cls_score_list = jt.concat(cls_score_list, dim=1)  # axis=1 -> dim=1
            reg_distri_list = jt.concat(reg_distri_list, dim=1)
            
            return x, cls_score_list, reg_distri_list
        else:
            # 推理模式：返回最终检测结果
            cls_score_list = []
            reg_dist_list = []
            # 完全对齐PyTorch版本的anchor生成
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, is_eval=True, mode='af')
            
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
                    # 完全对齐PyTorch版本：reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l])
                    reg_output = reg_output.permute(0, 2, 1, 3)  # [B, reg_max+1, 4, L]
                    reg_output = self.proj_conv(jt.nn.softmax(reg_output, dim=1))
                
                cls_output = jt.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            cls_score_list = jt.concat(cls_score_list, dim=-1).permute(0, 2, 1)
            reg_dist_list = jt.concat(reg_dist_list, dim=-1).permute(0, 2, 1)
            
            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            
            # 返回格式与PyTorch完全一致：[pred_bboxes, obj_conf, cls_scores]
            return jt.concat([
                pred_bboxes,
                jt.ones((b, pred_bboxes.shape[1], 1)),  # 硬编码置信度为1.0
                cls_score_list
            ], dim=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    """构建高效解耦头层 - 严格对齐PyTorch版本Gold-YOLO-N"""
    # Gold-YOLO-N配置: channels_list = [128, 256, 512]
    # 直接使用通道列表，不需要复杂的索引
    if len(channels_list) == 3:
        chx = [0, 1, 2]  # 对应 [128, 256, 512]
        print(f"✅ Gold-YOLO-N检测头配置: {channels_list}")
    else:
        # 兼容其他配置
        chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    head_layers = nn.Sequential(
        # stem0
        Conv(
            c1=channels_list[chx[0]],
            c2=channels_list[chx[0]],
            k=1,
            s=1
        ),
        # cls_conv0
        Conv(
            c1=channels_list[chx[0]],
            c2=channels_list[chx[0]],
            k=3,
            s=1
        ),
        # reg_conv0
        Conv(
            c1=channels_list[chx[0]],
            c2=channels_list[chx[0]],
            k=3,
            s=1
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem1
        Conv(
            c1=channels_list[chx[1]],
            c2=channels_list[chx[1]],
            k=1,
            s=1
        ),
        # cls_conv1
        Conv(
            c1=channels_list[chx[1]],
            c2=channels_list[chx[1]],
            k=3,
            s=1
        ),
        # reg_conv1
        Conv(
            c1=channels_list[chx[1]],
            c2=channels_list[chx[1]],
            k=3,
            s=1
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # stem2
        Conv(
            c1=channels_list[chx[2]],
            c2=channels_list[chx[2]],
            k=1,
            s=1
        ),
        # cls_conv2
        Conv(
            c1=channels_list[chx[2]],
            c2=channels_list[chx[2]],
            k=3,
            s=1
        ),
        # reg_conv2
        Conv(
            c1=channels_list[chx[2]],
            c2=channels_list[chx[2]],
            k=3,
            s=1
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
    )
    
    return head_layers
