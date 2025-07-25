#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
精确的GOLD-YOLO Jittor实现 - 严格对齐PyTorch版本的参数量
使用真正的EfficientRep Backbone和RepGDNeck
"""

import jittor as jt
from jittor import nn
import math

# 导入已经迁移好的组件
from yolov6.models.efficientrep import EfficientRep
from yolov6.models.effidehead import Detect as EffiDeHead
from gold_yolo.reppan import RepGDNeck


class AccurateGoldYOLO(nn.Module):
    """精确的GOLD-YOLO模型，严格对齐PyTorch版本"""
    
    def __init__(self, num_classes=20, width_multiple=0.25, depth_multiple=0.33):
        super().__init__()
        self.num_classes = num_classes
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple
        
        # GOLD-YOLO-n的精确配置
        backbone_config = {
            'num_repeats': [1, 6, 12, 18, 6],
            'out_channels': [64, 128, 256, 512, 1024],
            'fuse_P2': True,
            'cspsppf': True
        }
        
        neck_config = {
            'num_repeats': [12, 12, 12, 12, 12, 12, 12, 12, 12],  # 需要9个值
            'out_channels': [256, 128, 128, 256, 256, 512, 128, 128, 256, 256, 512],  # 需要11个值
            'extra_cfg': {
                'norm_cfg': {'type': 'BN', 'requires_grad': True},
                'depths': 2,
                'fusion_in': 480,
                'fuse_block_num': 3,
                'embed_dim_p': 96,
                'embed_dim_n': 352,
                'key_dim': 8,
                'num_heads': 4,
                'mlp_ratios': 1,
                'attn_ratios': 2,
                'c2t_stride': 2,
                'drop_path_rate': 0.1,
                'trans_channels': [64, 32, 64, 128],
                'pool_mode': 'torch'
            }
        }
        
        head_config = {
            'in_channels': [128, 256, 512],
            'num_layers': 3,
            'anchors': 3,
            'strides': [8, 16, 32],
            'use_dfl': False,  # nano版本不使用DFL
            'reg_max': 0
        }
        
        # 应用缩放参数
        self._apply_scaling(backbone_config, neck_config, head_config)
        
        # 创建模型组件
        print(f"🏗️ 创建EfficientRep Backbone...")
        self.backbone = EfficientRep(
            channels_list=backbone_config['out_channels'],
            num_repeats=backbone_config['num_repeats'],
            fuse_P2=backbone_config['fuse_P2'],
            cspsppf=backbone_config['cspsppf']
        )
        
        print(f"🏗️ 创建RepGDNeck...")
        # 创建extra_cfg对象
        from types import SimpleNamespace
        extra_cfg = SimpleNamespace(**neck_config['extra_cfg'])
        
        self.neck = RepGDNeck(
            channels_list=neck_config['out_channels'],
            num_repeats=neck_config['num_repeats'],
            extra_cfg=extra_cfg
        )
        
        print(f"🏗️ 创建EffiDeHead...")
        # 创建head_layers，这是Detect类需要的参数
        head_layers = head_config['in_channels']  # [32, 64, 128]

        self.detect = EffiDeHead(
            num_classes=num_classes,
            num_layers=head_config['num_layers'],
            head_layers=head_layers,
            use_dfl=head_config['use_dfl'],
            reg_max=head_config['reg_max']
        )
        
        print(f"✅ 精确GOLD-YOLO模型创建完成")
    
    def _apply_scaling(self, backbone_config, neck_config, head_config):
        """应用width_multiple和depth_multiple缩放"""
        
        # 缩放backbone通道数
        backbone_config['out_channels'] = [
            max(round(c * self.width_multiple), 1) 
            for c in backbone_config['out_channels']
        ]
        
        # 缩放backbone深度
        backbone_config['num_repeats'] = [
            max(round(d * self.depth_multiple), 1) 
            for d in backbone_config['num_repeats']
        ]
        
        # 缩放neck通道数
        neck_config['out_channels'] = [
            max(round(c * self.width_multiple), 1) 
            for c in neck_config['out_channels']
        ]
        
        # 缩放neck深度
        neck_config['num_repeats'] = [
            max(round(d * self.depth_multiple), 1) 
            for d in neck_config['num_repeats']
        ]
        
        # 缩放head通道数
        head_config['in_channels'] = [
            max(round(c * self.width_multiple), 1) 
            for c in head_config['in_channels']
        ]
        
        # 缩放neck的extra_cfg中的通道数
        extra_cfg = neck_config['extra_cfg']
        extra_cfg['fusion_in'] = max(round(extra_cfg['fusion_in'] * self.width_multiple), 1)
        extra_cfg['embed_dim_p'] = max(round(extra_cfg['embed_dim_p'] * self.width_multiple), 1)
        extra_cfg['embed_dim_n'] = max(round(extra_cfg['embed_dim_n'] * self.width_multiple), 1)
        extra_cfg['trans_channels'] = [
            max(round(c * self.width_multiple), 1) 
            for c in extra_cfg['trans_channels']
        ]
        
        print(f"📏 缩放后配置:")
        print(f"   Backbone通道: {backbone_config['out_channels']}")
        print(f"   Backbone深度: {backbone_config['num_repeats']}")
        print(f"   Neck通道: {neck_config['out_channels']}")
        print(f"   Head通道: {head_config['in_channels']}")
    
    def execute(self, x):
        """前向传播"""
        # Backbone特征提取
        features = self.backbone(x)
        
        # Neck特征融合
        neck_features = self.neck(features)
        
        # Head检测
        outputs = self.detect(neck_features)
        
        return outputs


def create_accurate_gold_yolo_model(config_name='gold_yolo-n', num_classes=20):
    """创建精确的GOLD-YOLO模型"""
    print(f'🎯 创建精确的{config_name}模型...')
    
    # 根据配置名称设置缩放参数
    if 'n' in config_name:
        width_multiple, depth_multiple = 0.25, 0.33
    elif 's' in config_name:
        width_multiple, depth_multiple = 0.50, 0.33
    elif 'm' in config_name:
        width_multiple, depth_multiple = 0.75, 0.60
    elif 'l' in config_name:
        width_multiple, depth_multiple = 1.0, 1.0
    else:
        width_multiple, depth_multiple = 0.25, 0.33
    
    # 创建模型
    model = AccurateGoldYOLO(
        num_classes=num_classes,
        width_multiple=width_multiple,
        depth_multiple=depth_multiple
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f'✅ 精确{config_name}模型创建成功')
    print(f'   参数量: {total_params/1e6:.2f}M')
    print(f'   缩放参数: width={width_multiple}, depth={depth_multiple}')
    
    return model


if __name__ == '__main__':
    # 测试精确模型
    model = create_accurate_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 测试前向传播
    x = jt.randn(1, 3, 640, 640)
    with jt.no_grad():
        outputs = model(x)
    
    print(f'✅ 前向传播测试成功')
    print(f'   输入形状: {list(x.shape)}')
    if isinstance(outputs, (list, tuple)):
        print(f'   输出形状: {[list(o.shape) for o in outputs]}')
    else:
        print(f'   输出形状: {list(outputs.shape)}')
    print('🎯 精确Gold-YOLO模型测试完成！')
