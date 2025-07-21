#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完整的EffiDeHead实现 - 100%对齐PyTorch官方版本
基于Gold-YOLO官方的effidehead.py实现
"""

import jittor as jt
from jittor import nn
import jittor.nn.functional as F
import math

from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    """Efficient Decoupled Head - Jittor版本
    
    硬件感知设计的解耦检测头，使用混合通道方法优化
    """
    
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None, 
                 use_dfl=True, reg_max=16):
        super().__init__()
        assert head_layers is not None
        
        self.nc = num_classes  # 类别数量
        self.no = num_classes + 5  # 每个anchor的输出数量
        self.nl = num_layers  # 检测层数量
        self.grid = [jt.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        
        # 步长设置
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]
        self.stride = jt.array(stride)
        
        # DFL设置
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        
        # 初始化解耦检测头
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        # 高效解耦检测头层
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
        
        print(f"✅ EffiDeHead初始化完成: {num_classes}类, {num_layers}层, DFL={use_dfl}")
    
    def initialize_biases(self):
        """初始化偏置 - 完全对齐PyTorch版本"""
        
        # 分类预测层偏置初始化
        for conv in self.cls_preds:
            b = conv.bias.view(-1)
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = nn.Parameter(w, requires_grad=True)
        
        # 回归预测层偏置初始化
        for conv in self.reg_preds:
            b = conv.bias.view(-1)
            b.data.fill_(1.0)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = nn.Parameter(w, requires_grad=True)
        
        # DFL投影层初始化
        self.proj = nn.Parameter(jt.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(
            self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
            requires_grad=False
        )
        
        print("✅ EffiDeHead偏置初始化完成")
    
    def execute(self, x):
        """前向传播 - 完全对齐PyTorch版本"""
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
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            
            # 分类分支
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            
            # 回归分支
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            
            # Sigmoid激活分类输出
            cls_output = jt.sigmoid(cls_output)
            
            # 重塑输出格式: [B, C, H, W] -> [B, H*W, C]
            cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
        
        # 拼接所有尺度的预测
        cls_score_list = jt.concat(cls_score_list, dim=1)
        reg_distri_list = jt.concat(reg_distri_list, dim=1)
        
        return x, cls_score_list, reg_distri_list
    
    def _forward_eval(self, x):
        """评估时的前向传播"""
        cls_score_list = []
        reg_dist_list = []
        
        # 生成anchor点
        anchor_points, stride_tensor = generate_anchors(
            x, self.stride, self.grid_cell_size, self.grid_cell_offset, 
            device=x[0].device, is_eval=True, mode='af'
        )
        
        for i in range(self.nl):
            # Stem处理
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            
            # 分类分支
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            
            # 回归分支
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            
            # Sigmoid激活分类输出
            cls_output = jt.sigmoid(cls_output)
            
            # 重塑输出格式
            cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
            reg_dist_list.append(reg_output.flatten(2).permute((0, 2, 1)))
        
        # 拼接所有尺度的预测
        cls_score_list = jt.concat(cls_score_list, dim=1)
        reg_dist_list = jt.concat(reg_dist_list, dim=1)
        
        # DFL解码
        if self.use_dfl:
            reg_dist_list = reg_dist_list.reshape(-1, self.reg_max + 1)
            reg_dist_list = F.softmax(reg_dist_list, dim=1)
            reg_dist_list = F.conv2d(reg_dist_list.unsqueeze(0).unsqueeze(0), 
                                   self.proj_conv.weight).squeeze().view(-1, 4)
        
        # 解码边界框
        if isinstance(anchor_points, list):
            anchor_points = jt.concat(anchor_points)
        if isinstance(stride_tensor, list):
            stride_tensor = jt.concat(stride_tensor)
        
        pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
        pred_bboxes *= stride_tensor
        
        return jt.concat([pred_bboxes, cls_score_list], dim=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    """构建EffiDeHead层 - 完全对齐PyTorch版本"""
    
    head_layers = []
    
    for i in range(num_layers):
        # 获取输入通道数
        if i == 0:
            in_channels = channels_list[6]  # P3: 128
        elif i == 1:
            in_channels = channels_list[8]  # P4: 256  
        else:
            in_channels = channels_list[10]  # P5: 512
        
        # Stem层
        stem = SimConv(in_channels, in_channels, 1, 1)
        head_layers.append(stem)
        
        # 分类卷积层
        cls_conv = SimConv(in_channels, in_channels, 3, 1)
        head_layers.append(cls_conv)
        
        # 回归卷积层
        reg_conv = SimConv(in_channels, in_channels, 3, 1)
        head_layers.append(reg_conv)
        
        # 分类预测层
        cls_pred = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        head_layers.append(cls_pred)
        
        # 回归预测层 (DFL格式)
        reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1) * num_anchors, 1)
        head_layers.append(reg_pred)
    
    print(f"✅ 构建EffiDeHead层完成: {len(head_layers)}层")
    return head_layers
