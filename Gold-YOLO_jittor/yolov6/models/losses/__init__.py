# GOLD-YOLO Jittor版本 - 损失函数模块初始化文件

from .loss import ComputeLoss, VarifocalLoss, BboxLoss

__all__ = [
    'ComputeLoss',
    'VarifocalLoss',
    'BboxLoss'
]
