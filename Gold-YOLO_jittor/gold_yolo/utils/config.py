#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
配置管理
"""

class ConfigDict(dict):
    """配置字典"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
    
    def get(self, key, default=None):
        return super().get(key, default)

class Config:
    """配置类"""
    @staticmethod
    def fromfile(config_path):
        """从文件加载配置"""
        # 简化实现，返回默认配置
        config = ConfigDict()
        config.model = ConfigDict()
        config.model.depth_multiple = 0.33
        config.model.width_multiple = 0.25
        config.model.backbone = ConfigDict()
        config.model.backbone.type = 'EfficientRep'
        config.model.backbone.num_repeats = [1, 6, 12, 18, 6]
        config.model.backbone.out_channels = [64, 128, 256, 512, 1024]
        config.model.neck = ConfigDict()
        config.model.neck.type = 'RepGDNeck'
        config.model.neck.num_repeats = [12, 12, 12, 12]
        config.model.neck.out_channels = [256, 128, 128, 256, 256, 512]
        config.model.head = ConfigDict()
        config.model.head.type = 'EffiDeHead'
        config.model.head.num_layers = 3
        config.model.head.use_dfl = False
        config.model.head.reg_max = 0
        config.training_mode = "conv_silu"
        return config
