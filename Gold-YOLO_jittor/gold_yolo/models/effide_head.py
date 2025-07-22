#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完整的EffiDeHead实现 (Jittor版本)
新芽第二阶段：方案A2 - 完善检测头
"""

import jittor as jt
import jittor.nn as nn
import math
from ..layers.common import Conv


class EffiDeHead(nn.Module):
    """完整的Efficient Decoupled Head - 对齐PyTorch版本"""
    
    def __init__(self, num_classes=20, num_layers=3, inplace=True, head_channels=None, use_dfl=True, reg_max=16):
        super().__init__()
        
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers (3个尺度)
        self.prior_prob = 1e-2
        self.inplace = inplace
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        
        # 步长配置
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]
        self.stride = jt.array(stride)
        
        # DFL投影层
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        
        # 头部通道配置 (对应neck输出的3个尺度)
        if head_channels is None:
            head_channels = [128, 128, 128]  # P3, N4, N5的通道数
        
        # 初始化解耦头部
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # 为每个检测层构建头部
        for i in range(num_layers):
            ch = head_channels[i]
            
            # Stem层
            self.stems.append(Conv(ch, ch, kernel_size=1, stride=1))
            
            # 分类卷积
            self.cls_convs.append(Conv(ch, ch, kernel_size=3, stride=1))
            
            # 回归卷积
            self.reg_convs.append(Conv(ch, ch, kernel_size=3, stride=1))
            
            # 分类预测
            self.cls_preds.append(nn.Conv2d(ch, num_classes, kernel_size=1))
            
            # 回归预测 (DFL格式)
            if use_dfl:
                self.reg_preds.append(nn.Conv2d(ch, 4 * (reg_max + 1), kernel_size=1))
            else:
                self.reg_preds.append(nn.Conv2d(ch, 4, kernel_size=1))
    
    def initialize_biases(self):
        """初始化偏置"""
        # 分类头偏置初始化
        for conv in self.cls_preds:
            b = conv.bias.view(-1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = nn.Parameter(w, requires_grad=True)
        
        # 回归头偏置初始化
        for conv in self.reg_preds:
            b = conv.bias.view(-1)
            b.data.fill_(1.0)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = nn.Parameter(w, requires_grad=True)
        
        # DFL投影参数
        if self.use_dfl:
            self.proj = nn.Parameter(jt.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
            self.proj_conv.weight = nn.Parameter(
                self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                requires_grad=False
            )
    
    def execute(self, x):
        """
        前向传播
        x: [P3, N4, N5] 三个尺度的特征
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_eval(x)
    
    def _forward_train(self, x):
        """训练时的前向传播"""
        cls_score_list = []
        reg_distri_list = []
        
        for i in range(self.nl):
            # Stem处理
            feat = self.stems[i](x[i])
            
            # 分类分支
            cls_feat = self.cls_convs[i](feat)
            cls_output = self.cls_preds[i](cls_feat)
            cls_output = jt.sigmoid(cls_output)
            
            # 回归分支
            reg_feat = self.reg_convs[i](feat)
            reg_output = self.reg_preds[i](reg_feat)
            
            # 重塑输出格式
            cls_score_list.append(cls_output.flatten(2).permute(0, 2, 1))
            reg_distri_list.append(reg_output.flatten(2).permute(0, 2, 1))
        
        # 拼接所有尺度
        cls_score_list = jt.concat(cls_score_list, dim=1)
        reg_distri_list = jt.concat(reg_distri_list, dim=1)
        
        return x, cls_score_list, reg_distri_list
    
    def _forward_eval(self, x):
        """推理时的前向传播"""
        cls_score_list = []
        reg_dist_list = []
        
        # 生成anchor points (简化版)
        anchor_points, stride_tensor = self._generate_anchors(x)
        
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            l = h * w
            
            # Stem处理
            feat = self.stems[i](x[i])
            
            # 分类分支
            cls_feat = self.cls_convs[i](feat)
            cls_output = self.cls_preds[i](cls_feat)
            cls_output = jt.sigmoid(cls_output)
            
            # 回归分支
            reg_feat = self.reg_convs[i](feat)
            reg_output = self.reg_preds[i](reg_feat)
            
            # DFL处理
            if self.use_dfl:
                reg_output = reg_output.reshape(b, 4, self.reg_max + 1, l).permute(0, 2, 1, 3)
                reg_output = self.proj_conv(jt.nn.softmax(reg_output, dim=1))
            
            # 重塑输出
            cls_score_list.append(cls_output.reshape(b, self.nc, l))
            reg_dist_list.append(reg_output.reshape(b, 4, l))
        
        # 拼接所有尺度
        cls_score_list = jt.concat(cls_score_list, dim=-1).permute(0, 2, 1)
        reg_dist_list = jt.concat(reg_dist_list, dim=-1).permute(0, 2, 1)
        
        # 转换为边界框 (简化版)
        pred_bboxes = self._dist2bbox(reg_dist_list, anchor_points)
        pred_bboxes *= stride_tensor
        
        # 拼接最终输出
        b = pred_bboxes.shape[0]
        objectness = jt.ones((b, pred_bboxes.shape[1], 1))
        
        return jt.concat([pred_bboxes, objectness, cls_score_list], dim=-1)
    
    def _generate_anchors(self, x):
        """生成anchor points (简化版)"""
        anchor_points = []
        stride_tensor = []
        
        for i, feat in enumerate(x):
            h, w = feat.shape[2:]
            stride = self.stride[i]
            
            # 生成网格点
            shift_x = jt.arange(0, w) + self.grid_cell_offset
            shift_y = jt.arange(0, h) + self.grid_cell_offset
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
            
            # 转换为anchor points
            anchor_point = jt.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
            anchor_point *= stride
            
            anchor_points.append(anchor_point)
            stride_tensor.append(jt.full((anchor_point.shape[0], 1), stride))
        
        anchor_points = jt.concat(anchor_points, dim=0)
        stride_tensor = jt.concat(stride_tensor, dim=0)
        
        return anchor_points, stride_tensor
    
    def _dist2bbox(self, distance, anchor_points):
        """距离转边界框 (简化版)"""
        lt, rb = jt.split(distance, 2, dim=-1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return jt.concat([x1y1, x2y2], dim=-1)


def build_effide_head(neck_channels, num_classes=20, use_dfl=True, reg_max=16):
    """构建完整的EffiDeHead"""
    return EffiDeHead(
        num_classes=num_classes,
        num_layers=3,  # 3个检测尺度
        head_channels=neck_channels,  # [P3, N4, N5]的通道数
        use_dfl=use_dfl,
        reg_max=reg_max
    )
