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
        # 关键修复：在Jittor中，jt.array赋给模块属性会被注册为参数
        # PyTorch中torch.tensor()不会被注册，所以这里用Python list存储
        self._stride_list = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        # 修复关键错误：与PyTorch版本对齐，总是创建proj_conv层
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

    @property
    def stride(self):
        """通过property访问stride，避免被注册为参数"""
        return jt.array(self._stride_list).float32().stop_grad()

    def initialize_biases(self):
        """Initialize cls/reg heads and DFL projection exactly like PyTorch."""
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for conv in self.cls_preds:
            conv.bias.assign(jt.full_like(conv.bias, bias_value))
            conv.weight.assign(jt.zeros_like(conv.weight))

        for conv in self.reg_preds:
            conv.bias.assign(jt.ones_like(conv.bias))
            conv.weight.assign(jt.zeros_like(conv.weight))

        self.proj = jt.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.stop_grad()
        proj_weight = self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach()
        proj_weight = proj_weight.astype(self.proj_conv.weight.dtype)
        self.proj_conv.weight.assign(proj_weight)
        self.proj_conv.weight.stop_grad()
    
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

                # 完全照抄PyTorch版本：训练时也应用sigmoid
                cls_output = jt.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))

                # 修复关键错误：正确处理DFL输出格式
                # 训练时需要保持原始的分布参数，不进行proj_conv变换
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            # 100%对齐PyTorch版本：返回(feats, pred_scores, pred_distri)
            cls_score_list = jt.concat(cls_score_list, dim=1)  # [batch, anchors, num_classes]
            reg_distri_list = jt.concat(reg_distri_list, dim=1)  # [batch, anchors, 4*(reg_max+1)] 或 [batch, anchors, 4]

            # PyTorch版本第62行：feats, pred_scores, pred_distri = outputs
            return x, cls_score_list, reg_distri_list
        
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
                
                # 修复关键错误：与PyTorch版本对齐，总是使用proj_conv（当use_dfl=True时）
                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(nn.softmax(reg_output, dim=1))
                
                # 完全照抄PyTorch版本：推理时也应用sigmoid
                cls_output = jt.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            cls_score_list = jt.concat(cls_score_list, dim=-1).permute(0, 2, 1)
            reg_dist_list = jt.concat(reg_dist_list, dim=-1).permute(0, 2, 1)

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor

            # 完全照抄PyTorch版本的输出格式
            # PyTorch版本第125-132行的完全照抄
            return jt.concat([
                pred_bboxes,      # [b, anchors, 4] 坐标
                jt.ones((b, pred_bboxes.shape[1], 1), dtype=pred_bboxes.dtype),  # 完全照抄：objectness全为1
                cls_score_list    # [b, anchors, 20] 类别分数
            ], dim=-1)


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):
    """构建EffiDeHead层 - channels_list是完整的backbone+neck通道列表"""

    # 与PyTorch版本对齐：使用正确的索引从channels_list中取head输入通道
    if num_layers == 3:
        chx = [6, 8, 10]
    elif num_layers == 4:
        chx = [8, 9, 10, 11]
    else:
        raise ValueError(f"Unsupported num_layers: {num_layers}")

    head_layers = []

    # 为每个检测层创建5个模块：stem, cls_conv, reg_conv, cls_pred, reg_pred
    for i in range(num_layers):
        ch = channels_list[chx[i]]  # 使用正确的索引

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

        head_layers.append(nn.Conv2d(
            in_channels=ch,
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ))

    return head_layers
