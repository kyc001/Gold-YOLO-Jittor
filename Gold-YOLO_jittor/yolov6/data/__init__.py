# GOLD-YOLO Jittor版本 - 数据模块初始化文件

from .data_augment import (
    augment_hsv,
    letterbox,
    mixup,
    box_candidates,
    random_affine,
    get_transform_matrix,
    mosaic_augmentation,
)
from .data_load import create_dataloader, TrainValDataLoader, JittorDistributedSampler
from .datasets import TrainValDataset, LoadData, IMG_FORMATS, VID_FORMATS

# PyTorch upstream版本在不同提交中exif_size可能不存在，这里做兼容导出
try:
    from .datasets import exif_size
except Exception:
    exif_size = None

__all__ = [
    # data_augment
    "augment_hsv",
    "letterbox",
    "mixup",
    "box_candidates",
    "random_affine",
    "get_transform_matrix",
    "mosaic_augmentation",
    # data_load
    "create_dataloader",
    "TrainValDataLoader",
    "JittorDistributedSampler",
    # datasets
    "TrainValDataset",
    "LoadData",
    "exif_size",
    "IMG_FORMATS",
    "VID_FORMATS",
]
