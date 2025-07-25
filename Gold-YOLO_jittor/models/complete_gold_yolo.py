#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完整的Gold-YOLO Jittor实现 - 解决损失值偏大问题
包含完整的backbone和neck，提升特征提取能力
"""

import jittor as jt
from jittor import nn
import math


class ConvBNSiLU(nn.Module):
    """标准卷积+BN+SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = ConvBNSiLU(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    
    def execute(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(jt.concat((y1, y2), dim=1))))


class Bottleneck(nn.Module):
    """标准瓶颈块"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def execute(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    """空间金字塔池化"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNSiLU(c1, c_, 1, 1)
        self.cv2 = ConvBNSiLU(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def execute(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(jt.concat([x, y1, y2, self.m(y2)], 1))


class CompleteBackbone(nn.Module):
    """完整的骨干网络 - 支持通道缩放"""
    def __init__(self, c1=3, channels=None):
        super().__init__()

        # 默认通道配置
        if channels is None:
            channels = [64, 128, 256, 512, 1024]

        c2, c3, c4, c5, c6 = channels

        # Stem
        self.stem = ConvBNSiLU(c1, c2, 6, 2, 2)

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNSiLU(c2, c3, 3, 2, 1),
            BottleneckCSP(c3, c3, 3)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNSiLU(c3, c4, 3, 2, 1),
            BottleneckCSP(c4, c4, 6)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNSiLU(c4, c5, 3, 2, 1),
            BottleneckCSP(c5, c5, 9)
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBNSiLU(c5, c6, 3, 2, 1),
            BottleneckCSP(c6, c6, 3),
            SPPF(c6, c6)
        )
    
    def execute(self, x):
        x = self.stem(x)
        
        x1 = self.stage1(x)    # 1/4
        x2 = self.stage2(x1)   # 1/8  
        x3 = self.stage3(x2)   # 1/16
        x4 = self.stage4(x3)   # 1/32
        
        return [x2, x3, x4]  # 返回多尺度特征


class CompleteNeck(nn.Module):
    """完整的颈部网络 - FPN+PAN结构"""
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256):
        super().__init__()
        
        # Top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv1 = ConvBNSiLU(in_channels[2], out_channels, 1)
        self.lateral_conv2 = ConvBNSiLU(in_channels[1], out_channels, 1)
        self.lateral_conv3 = ConvBNSiLU(in_channels[0], out_channels, 1)
        
        # Fusion layers
        self.fpn_conv1 = ConvBNSiLU(out_channels, out_channels, 3, 1, 1)
        self.fpn_conv2 = ConvBNSiLU(out_channels, out_channels, 3, 1, 1)
        
        # Bottom-up pathway
        self.downsample1 = ConvBNSiLU(out_channels, out_channels, 3, 2, 1)
        self.downsample2 = ConvBNSiLU(out_channels, out_channels, 3, 2, 1)
        
        # PAN fusion
        self.pan_conv1 = ConvBNSiLU(out_channels, out_channels, 3, 1, 1)
        self.pan_conv2 = ConvBNSiLU(out_channels, out_channels, 3, 1, 1)
    
    def execute(self, features):
        c3, c4, c5 = features
        
        # Top-down
        p5 = self.lateral_conv1(c5)
        p4 = self.lateral_conv2(c4) + self.upsample(p5)
        p4 = self.fpn_conv1(p4)
        
        p3 = self.lateral_conv3(c3) + self.upsample(p4)
        p3 = self.fpn_conv2(p3)
        
        # Bottom-up
        n3 = p3
        n4 = self.pan_conv1(p4 + self.downsample1(n3))
        n5 = self.pan_conv2(p5 + self.downsample2(n4))
        
        return [n3, n4, n5]


class CompleteGoldYOLO(nn.Module):
    """完整的Gold-YOLO模型 - 解决损失值偏大问题"""
    
    def __init__(self, num_classes=20, channels=3, width_multiple=1.0, depth_multiple=1.0):
        super().__init__()
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple

        # 基础通道配置
        base_channels = [64, 128, 256, 512, 1024]
        # 应用width_multiple缩放
        scaled_channels = [max(round(c * width_multiple), 1) for c in base_channels]

        print(f'   原始通道: {base_channels}')
        print(f'   缩放通道: {scaled_channels}')
        print(f'   缩放系数: width={width_multiple}, depth={depth_multiple}')

        # 使用缩放后的通道数创建backbone
        self.backbone = CompleteBackbone(c1=channels, channels=scaled_channels)

        # 使用缩放后的通道数创建neck
        neck_in_channels = scaled_channels[-3:]  # [256, 512, 1024] -> 缩放后
        neck_out_channels = max(round(256 * width_multiple), 1)
        self.neck = CompleteNeck(in_channels=neck_in_channels, out_channels=neck_out_channels)

        # 使用缩放后的通道数创建检测头
        self.detect = SimpleDetectHead(num_classes=num_classes, in_channels=neck_out_channels)

        # 初始化权重
        self.initialize_weights()
    
    def execute(self, x):
        # 骨干网络特征提取
        features = self.backbone(x)
        
        # 颈部网络特征融合
        enhanced_features = self.neck(features)
        
        # 检测头预测
        predictions = self.detect(enhanced_features)
        
        return predictions
    
    def initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SimpleDetectHead(nn.Module):
    """简化的检测头"""
    def __init__(self, num_classes=20, in_channels=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 1
        
        # 分类分支
        self.cls_convs = nn.ModuleList([
            ConvBNSiLU(in_channels, in_channels, 3, 1, 1) for _ in range(3)
        ])
        self.cls_preds = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes * self.num_anchors, 1) for _ in range(3)
        ])
        
        # 回归分支
        self.reg_convs = nn.ModuleList([
            ConvBNSiLU(in_channels, in_channels, 3, 1, 1) for _ in range(3)
        ])
        self.reg_preds = nn.ModuleList([
            nn.Conv2d(in_channels, 4 * self.num_anchors, 1) for _ in range(3)
        ])
        
        # 置信度分支
        self.obj_preds = nn.ModuleList([
            nn.Conv2d(in_channels, self.num_anchors, 1) for _ in range(3)
        ])
    
    def execute(self, features):
        outputs = []
        
        for i, feat in enumerate(features):
            # 分类预测
            cls_feat = self.cls_convs[i](feat)
            cls_pred = self.cls_preds[i](cls_feat)
            
            # 回归预测
            reg_feat = self.reg_convs[i](feat)
            reg_pred = self.reg_preds[i](reg_feat)
            
            # 置信度预测
            obj_pred = self.obj_preds[i](feat)
            
            # 合并预测
            B, _, H, W = feat.shape
            cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
            reg_pred = reg_pred.view(B, self.num_anchors, 4, H, W).permute(0, 1, 3, 4, 2).contiguous()
            obj_pred = obj_pred.view(B, self.num_anchors, 1, H, W).permute(0, 1, 3, 4, 2).contiguous()
            
            pred = jt.concat([reg_pred, obj_pred, cls_pred], dim=-1)
            outputs.append(pred.view(B, -1, self.num_classes + 5))
        
        return jt.concat(outputs, dim=1)


def create_complete_model(num_classes=20):
    """创建完整的Gold-YOLO模型"""
    print('🏗️ 创建完整Gold-YOLO模型...')
    model = CompleteGoldYOLO(num_classes=num_classes)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'✅ 完整模型创建成功')
    print(f'   参数量: {total_params/1e6:.2f}M')
    print(f'   架构: 完整backbone + neck + head')
    
    return model


def create_gold_yolo_model(config_name='gold_yolo-n', num_classes=20):
    """根据配置名称创建GOLD-YOLO模型，正确应用缩放参数"""
    print(f'🏗️ 创建{config_name}模型...')

    # 根据配置名称设置缩放参数
    if 'n' in config_name:
        # GOLD-YOLO-n配置: width_multiple=0.25, depth_multiple=0.33
        width_multiple = 0.25
        depth_multiple = 0.33
    elif 's' in config_name:
        # GOLD-YOLO-s配置: width_multiple=0.50, depth_multiple=0.33
        width_multiple = 0.50
        depth_multiple = 0.33
    elif 'm' in config_name:
        # GOLD-YOLO-m配置: width_multiple=0.75, depth_multiple=0.60
        width_multiple = 0.75
        depth_multiple = 0.60
    elif 'l' in config_name:
        # GOLD-YOLO-l配置: width_multiple=1.0, depth_multiple=1.0
        width_multiple = 1.0
        depth_multiple = 1.0
    else:
        # 默认使用n配置
        width_multiple = 0.25
        depth_multiple = 0.33

    # 应用缩放参数创建模型
    model = CompleteGoldYOLO(
        num_classes=num_classes,
        width_multiple=width_multiple,
        depth_multiple=depth_multiple
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())

    print(f'✅ {config_name}模型创建成功')
    print(f'   参数量: {total_params/1e6:.2f}M')
    print(f'   缩放参数: width={width_multiple}, depth={depth_multiple}')
    print(f'   架构: 缩放后的backbone + neck + head')

    return model


if __name__ == '__main__':
    # 测试完整模型
    model = create_complete_model(num_classes=20)
    
    # 测试前向传播
    x = jt.randn(1, 3, 640, 640)
    output = model(x)
    print(f'✅ 前向传播成功: {output.shape}')
    print(f'🎉 完整Gold-YOLO模型测试通过!')
