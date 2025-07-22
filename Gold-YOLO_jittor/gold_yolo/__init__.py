"""
Gold-YOLO Jittor Implementation
新芽第二阶段：模块化的Gold-YOLO实现

参考PyTorch的组织方式，提供清晰的API接口
"""

__version__ = "1.0.0"
__author__ = "新芽第二阶段"

# 导入核心组件
from .models import GoldYOLO, FullPyTorchGoldYOLOSmall, build_gold_yolo_small
from .utils import FullYOLODecoder, YOLODecoder

# 便捷的构建函数
def build_model(config_path=None, num_classes=80):
    """构建Gold-YOLO模型"""
    return GoldYOLO(num_classes=num_classes)

def build_decoder(input_size=640, num_classes=80):
    """构建YOLO解码器"""
    return FullYOLODecoder(input_size=input_size, num_classes=num_classes)

__all__ = [
    'GoldYOLO',
    'FullPyTorchGoldYOLOSmall',
    'FullYOLODecoder',
    'YOLODecoder',
    'build_model',
    'build_decoder',
    'build_gold_yolo_small'
]
