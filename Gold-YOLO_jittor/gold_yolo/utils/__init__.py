"""
Gold-YOLO 工具组件
"""

from .decoder import FullYOLODecoder

# 别名
YOLODecoder = FullYOLODecoder

__all__ = ['FullYOLODecoder', 'YOLODecoder']
