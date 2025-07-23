#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
通用工具函数 - 简化版
"""

import jittor as jt

def dist2bbox(distance, anchor_points):
    """距离转边界框"""
    lt, rb = jt.split(distance, 2, dim=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return jt.concat([x1y1, x2y2], dim=-1)
