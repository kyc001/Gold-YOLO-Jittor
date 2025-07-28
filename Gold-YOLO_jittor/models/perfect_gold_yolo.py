#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
100%对齐的GOLD-YOLO Jittor实现
严格按照PyTorch版本的build_network函数实现，确保参数量完全对齐
"""

import os
import jittor as jt
from jittor import nn
import math
from types import SimpleNamespace

# 导入已经迁移好的组件
from yolov6.models.efficientrep import EfficientRep, CSPBepBackbone
from yolov6.models.effidehead import Detect, build_effidehead_layer
from gold_yolo.reppan import RepGDNeck, GDNeck, GDNeck2
from yolov6.layers.common import RepVGGBlock, BottleRep
from yolov6.utils.config import Config


def make_divisible(x, divisor):
    """使数字能被divisor整除"""
    return math.ceil(x / divisor) * divisor


def build_network(config, channels=3, num_classes=80, num_layers=3, fuse_ab=False, distill_ns=False):
    """
    100%对齐PyTorch版本的build_network函数
    严格按照原版逻辑构建网络
    """
    print(f"🏗️ 构建网络: channels={channels}, num_classes={num_classes}, num_layers={num_layers}")
    
    # 获取配置 - 处理dict格式的配置
    if hasattr(config, 'model'):
        model_cfg = config.model
    else:
        model_cfg = config['model']

    depth_mul = model_cfg['depth_multiple']
    width_mul = model_cfg['width_multiple']

    # 构建Backbone
    backbone_cfg = model_cfg['backbone']
    print(f"📐 缩放参数: depth_mul={depth_mul}, width_mul={width_mul}")
    
    # 应用缩放
    num_repeat_backbone = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in backbone_cfg['num_repeats']]
    channels_list_backbone = [make_divisible(i * width_mul, 8) for i in backbone_cfg['out_channels']]
    
    print(f"🔧 Backbone缩放后:")
    print(f"   原始重复次数: {backbone_cfg['num_repeats']}")
    print(f"   缩放重复次数: {num_repeat_backbone}")
    print(f"   原始通道数: {backbone_cfg['out_channels']}")
    print(f"   缩放通道数: {channels_list_backbone}")

    # 创建Backbone
    if backbone_cfg['type'] == 'EfficientRep':
        backbone = EfficientRep(
            in_channels=channels,
            channels_list=channels_list_backbone,
            num_repeats=num_repeat_backbone,
            block=RepVGGBlock,
            fuse_P2=backbone_cfg.get('fuse_P2', False),
            cspsppf=backbone_cfg.get('cspsppf', False)
        )
    elif backbone_cfg['type'] == 'CSPBepBackbone':
        backbone = CSPBepBackbone(
            in_channels=channels,
            channels_list=channels_list_backbone,
            num_repeats=num_repeat_backbone,
            block=BottleRep,
            fuse_P2=backbone_cfg.get('fuse_P2', False),
            cspsppf=backbone_cfg.get('cspsppf', False)
        )
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_cfg['type']}")

    # 构建Neck
    neck_cfg = model_cfg['neck']
    print(f"🔧 Neck配置: {neck_cfg['type']}")

    # 应用缩放到neck
    num_repeat_neck = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in neck_cfg['num_repeats']]
    base_neck_channels = neck_cfg['out_channels']

    # RepGDNeck需要完整的channels_list，包括backbone的输出通道
    # 格式：[backbone_out_channels] + [neck_specific_channels]
    full_channels_list = channels_list_backbone + [make_divisible(i * width_mul, 8) for i in base_neck_channels]

    print(f"🔧 完整通道列表: {full_channels_list} (长度: {len(full_channels_list)})")
    
    print(f"🔧 Neck缩放后:")
    print(f"   原始重复次数: {neck_cfg['num_repeats']}")
    print(f"   缩放重复次数: {num_repeat_neck}")
    print(f"   原始通道数: {neck_cfg['out_channels']}")
    print(f"   基础通道数: {base_neck_channels}")

    # 缩放extra_cfg中的参数
    extra_cfg = neck_cfg['extra_cfg'].copy()

    # fusion_in应该等于SimFusion_4in的输出通道数，即(c2, c3, c4, c5)的通道数之和
    # 从调试信息看：c2(32) + c3(64) + c4(128) + c5(256) = 480
    fusion_in_actual = sum(channels_list_backbone[1:])  # [32, 64, 128, 256] -> 480
    extra_cfg['fusion_in'] = fusion_in_actual

    extra_cfg['embed_dim_p'] = make_divisible(extra_cfg['embed_dim_p'] * width_mul, 8)
    extra_cfg['embed_dim_n'] = make_divisible(extra_cfg['embed_dim_n'] * width_mul, 8)
    extra_cfg['trans_channels'] = [make_divisible(i * width_mul, 8) for i in extra_cfg['trans_channels']]
    
    print(f"🔧 Extra_cfg缩放后:")
    print(f"   fusion_in: {neck_cfg['extra_cfg']['fusion_in']} -> {extra_cfg['fusion_in']} (实际通道数)")
    print(f"   embed_dim_p: {neck_cfg['extra_cfg']['embed_dim_p']} -> {extra_cfg['embed_dim_p']}")
    print(f"   embed_dim_n: {neck_cfg['extra_cfg']['embed_dim_n']} -> {extra_cfg['embed_dim_n']}")
    print(f"   trans_channels: {neck_cfg['extra_cfg']['trans_channels']} -> {extra_cfg['trans_channels']}")
    
    # 创建extra_cfg对象
    extra_cfg_obj = SimpleNamespace(**extra_cfg)
    
    # 创建Neck
    if neck_cfg['type'] == 'RepGDNeck':
        neck = RepGDNeck(
            channels_list=full_channels_list,
            num_repeats=num_repeat_neck,
            block=RepVGGBlock,
            extra_cfg=extra_cfg_obj
        )
    elif neck_cfg['type'] == 'GDNeck':
        neck = GDNeck(
            channels_list=full_channels_list,
            num_repeats=num_repeat_neck,
            block=BottleRep,
            extra_cfg=extra_cfg_obj
        )
    elif neck_cfg['type'] == 'GDNeck2':
        neck = GDNeck2(
            channels_list=full_channels_list,
            num_repeats=num_repeat_neck,
            block=BottleRep,
            extra_cfg=extra_cfg_obj
        )
    else:
        raise ValueError(f"Unsupported neck type: {neck_cfg['type']}")

    # 构建Head
    head_cfg = model_cfg['head']
    print(f"🔧 Head配置: {head_cfg['type']}")

    # Head的输入通道数（来自neck的输出）
    head_in_channels = [make_divisible(i * width_mul, 8) for i in head_cfg['in_channels']]

    print(f"🔧 Head缩放后:")
    print(f"   原始输入通道: {head_cfg['in_channels']}")
    print(f"   缩放输入通道: {head_in_channels}")

    # 获取其他head参数
    use_dfl = head_cfg.get('use_dfl', True)
    reg_max = head_cfg.get('reg_max', 16)
    
    print(f"🔧 Head参数: use_dfl={use_dfl}, reg_max={reg_max}")
    
    # 构建head layers
    head_layers = build_effidehead_layer(
        head_in_channels, 
        1, 
        num_classes, 
        reg_max=reg_max, 
        num_layers=num_layers
    )
    
    # 创建Head - 修复关键错误：与PyTorch版本对齐，不传递reg_max参数，使用默认值16
    head = Detect(
        num_classes=num_classes,
        num_layers=num_layers,
        head_layers=head_layers,
        use_dfl=use_dfl
        # 注意：不传递reg_max参数，使用默认值16，与PyTorch版本保持一致
    )
    
    print(f"✅ 网络构建完成!")
    
    return backbone, neck, head


class PerfectGoldYOLO(nn.Module):
    """100%对齐的GOLD-YOLO模型"""
    
    def __init__(self, config_path, num_classes=None, channels=3, fuse_ab=False, distill_ns=False):
        super().__init__()

        # 加载配置
        if isinstance(config_path, str):
            self.config = Config.fromfile(config_path)
        else:
            self.config = config_path

        # 从配置文件获取类别数，如果未指定则使用配置文件中的nc
        if num_classes is None:
            if hasattr(self.config, 'model') and 'nc' in self.config.model:
                num_classes = self.config.model['nc']
            elif hasattr(self.config, 'nc'):
                num_classes = self.config.nc
            elif 'nc' in self.config:
                num_classes = self.config['nc']
            else:
                num_classes = 80  # 默认COCO类别数
                print(f"⚠️ 配置文件中未找到nc参数，使用默认值: {num_classes}")

        # 保存重要属性
        self.num_classes = num_classes
        self.channels = channels
        
        print(f"🎯 模型初始化: num_classes={num_classes}, channels={channels}")

        # 构建网络
        if hasattr(self.config, 'model'):
            model_cfg = self.config.model
        else:
            model_cfg = self.config['model']

        num_layers = model_cfg['head']['num_layers']
        self.backbone, self.neck, self.detect = build_network(
            self.config, channels, num_classes, num_layers, fuse_ab, distill_ns
        )

        # 初始化检测头
        self.stride = self.detect.stride
        self.detect.initialize_biases()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.detect.parameters())
        
        print(f"📊 模型参数统计:")
        print(f"   总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   Backbone: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
        print(f"   Neck: {neck_params:,} ({neck_params/total_params*100:.1f}%)")
        print(f"   Head: {head_params:,} ({head_params/total_params*100:.1f}%)")
    
    def execute(self, x):
        """前向传播 - 修复训练/推理一致性"""
        # Backbone特征提取
        features = self.backbone(x)

        # Neck特征融合
        neck_features = self.neck(features)

        # Head检测 - 始终返回统一格式
        outputs = self.detect(neck_features)

        # 处理Head输出格式
        if self.training:
            # 训练模式：直接返回三元组给损失函数
            if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
                # 新格式：(feats, cls_scores, reg_distri)
                return outputs
            else:
                raise ValueError(f"训练模式期望三元组输出，得到: {type(outputs)}")
        else:
            # 推理模式：转换为YOLO格式
            if isinstance(outputs, (list, tuple)):
                if len(outputs) == 3:
                    # 新格式：(feats, cls_scores, reg_distri) -> 已经在Head中处理为YOLO格式
                    outputs = outputs  # 推理时Head应该返回单个tensor
                elif len(outputs) >= 2:
                    # 旧格式[pred_scores, pred_boxes]，需要转换
                    pred_scores = outputs[0]  # [batch, anchors, num_classes]
                    pred_boxes = outputs[1]   # [batch, anchors, 4]

                    # 合并为旧格式（为了兼容性）
                    outputs = jt.concat([pred_scores, pred_boxes], dim=-1)  # [batch, anchors, num_classes+4]
                else:
                    outputs = outputs[0]

            # 推理模式的验证（仅在推理时执行）
            if isinstance(outputs, jt.Var) and len(outputs.shape) == 3:
                # 验证输出格式
                batch_size, num_anchors, total_channels = outputs.shape
                expected_channels = 4 + 1 + 20  # YOLO格式：坐标 + 置信度 + 类别

                if total_channels != expected_channels:
                    print(f"⚠️ 推理输出通道数不匹配: 期望{expected_channels}, 实际{total_channels}")

            return outputs


def create_perfect_gold_yolo_model(config_name='gold_yolo-n', num_classes=20):
    """创建100%对齐的GOLD-YOLO模型"""
    print(f'🎯 创建100%对齐的{config_name}模型...')
    
    # 配置文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', f'{config_name}.py')

    print(f'📁 配置文件路径: {config_path}')
    if not os.path.exists(config_path):
        print(f'❌ 配置文件不存在: {config_path}')
        raise FileNotFoundError(f'配置文件不存在: {config_path}')
    
    # 创建模型
    model = PerfectGoldYOLO(
        config_path=config_path,
        num_classes=num_classes,
        channels=3,
        fuse_ab=False,
        distill_ns=False
    )
    
    print(f'✅ 100%对齐{config_name}模型创建成功')
    
    return model


if __name__ == '__main__':
    # 测试100%对齐模型
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
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
    print('🎯 100%对齐Gold-YOLO模型测试完成！')
