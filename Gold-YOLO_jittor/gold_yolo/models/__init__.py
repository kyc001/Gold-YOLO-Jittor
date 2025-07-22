"""
Gold-YOLO 模型组件
"""

from .gold_yolo import GoldYOLO, FullPyTorchGoldYOLOSmall, EfficientRep, RepVGGBlock, RepBlock

# 便捷构建函数
def build_gold_yolo_small(num_classes=80):
    """构建Gold-YOLO Small模型"""
    return GoldYOLO(num_classes=num_classes)

__all__ = ['GoldYOLO', 'FullPyTorchGoldYOLOSmall', 'EfficientRep', 'RepVGGBlock', 'RepBlock', 'build_gold_yolo_small']
