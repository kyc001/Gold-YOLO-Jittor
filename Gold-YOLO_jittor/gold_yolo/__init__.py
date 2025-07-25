# GOLD-YOLO Jittor版本 - 主模块初始化文件
# 从PyTorch版本迁移到Jittor框架，严格对齐所有功能

from .common import AdvPoolFusion, SimFusion_3in, SimFusion_4in
from .layers import Conv, Conv2d_BN, DropPath, h_sigmoid, drop_path
from .transformer import (
    Mlp, Attention, top_Block, PyramidPoolAgg, 
    TopBasicLayer, InjectionMultiSum_Auto_pool, onnx_AdaptiveAvgPool2d
)
from .reppan import RepGDNeck
from .switch_tool import switch_to_deploy, convert_checkpoint_False, convert_checkpoint_True

__all__ = [
    # common
    'AdvPoolFusion', 'SimFusion_3in', 'SimFusion_4in',
    
    # layers  
    'Conv', 'Conv2d_BN', 'DropPath', 'h_sigmoid', 'drop_path',
    
    # transformer
    'Mlp', 'Attention', 'top_Block', 'PyramidPoolAgg',
    'TopBasicLayer', 'InjectionMultiSum_Auto_pool', 'onnx_AdaptiveAvgPool2d',
    
    # reppan
    'RepGDNeck',
    
    # switch_tool
    'switch_to_deploy', 'convert_checkpoint_False', 'convert_checkpoint_True'
]
