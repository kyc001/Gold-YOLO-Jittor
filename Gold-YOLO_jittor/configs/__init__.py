#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Jittor配置文件包
"""

from .gold_yolo_n import model as gold_yolo_n_config
from .gold_yolo_s import model as gold_yolo_s_config  
from .gold_yolo_m import model as gold_yolo_m_config
from .gold_yolo_l import model as gold_yolo_l_config

__all__ = [
    'gold_yolo_n_config',
    'gold_yolo_s_config', 
    'gold_yolo_m_config',
    'gold_yolo_l_config'
]
