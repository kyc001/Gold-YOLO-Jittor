# GOLD-YOLO Jittor版本 - 主模块初始化文件
"""
GOLD-YOLO Jittor Implementation
基于Jittor框架的Gold-YOLO目标检测模型实现，严格对齐PyTorch版本
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "GOLD-YOLO Jittor Team"
__description__ = "GOLD-YOLO implementation in Jittor framework"

# 导入核心模型
from .models import (
    Model, build_model, build_network,
    EfficientRep, EfficientRep6, CSPBepBackbone,
    Detect, build_effidehead_layer,
    RepPANNeck, CSPRepPANNeck
)

# 导入基础层
from .layers import (
    Conv, SimConv, ConvWrapper, SimConvWrapper,
    SimSPPF, SPPF, SimCSPSPPF, CSPSPPF,
    Transpose, Concat, RepVGGBlock, RepBlock, BottleRep,
    Conv_C3, BepC3, BiFusion, get_block, autopad, conv_bn
)

# 导入工具函数
from .utils import (
    increment_name, find_latest_checkpoint, dist2bbox, bbox2dist,
    xywh2xyxy, box_iou, xyxy2xywh, xywhn2xyxy, xyxy2xywhn,
    clip_coords, scale_coords,
    jittor_distributed_zero_first, time_sync, initialize_weights,
    fuse_conv_and_bn, fuse_model, get_model_info, model_info,
    intersect_dicts, scale_img,
    set_logging, LOGGER, NCOLS, load_yaml, save_yaml,
    write_tblog, write_tbimg, AverageMeter, colorstr
)

# 导入分配器
from .assigners import (
    generate_anchors, make_anchors, dist_calculator,
    select_candidates_in_gts, select_highest_overlaps, 
    iou_calculator, bbox_overlaps, iou2d_calculator,
    ATSSAssigner
)

# 主要导出
__all__ = [
    # 版本信息
    '__version__', '__author__', '__description__',
    
    # 核心模型
    'Model', 'build_model', 'build_network',
    'EfficientRep', 'EfficientRep6', 'CSPBepBackbone',
    'Detect', 'build_effidehead_layer',
    'RepPANNeck', 'CSPRepPANNeck',
    
    # 基础层
    'Conv', 'SimConv', 'ConvWrapper', 'SimConvWrapper',
    'SimSPPF', 'SPPF', 'SimCSPSPPF', 'CSPSPPF',
    'Transpose', 'Concat', 'RepVGGBlock', 'RepBlock', 'BottleRep',
    'Conv_C3', 'BepC3', 'BiFusion', 'get_block', 'autopad', 'conv_bn',
    
    # 工具函数
    'increment_name', 'find_latest_checkpoint', 'dist2bbox', 'bbox2dist',
    'xywh2xyxy', 'box_iou', 'xyxy2xywh', 'xywhn2xyxy', 'xyxy2xywhn',
    'clip_coords', 'scale_coords',
    'jittor_distributed_zero_first', 'time_sync', 'initialize_weights',
    'fuse_conv_and_bn', 'fuse_model', 'get_model_info', 'model_info',
    'intersect_dicts', 'scale_img',
    'set_logging', 'LOGGER', 'NCOLS', 'load_yaml', 'save_yaml',
    'write_tblog', 'write_tbimg', 'AverageMeter', 'colorstr',
    
    # 分配器
    'generate_anchors', 'make_anchors', 'dist_calculator',
    'select_candidates_in_gts', 'select_highest_overlaps', 
    'iou_calculator', 'bbox_overlaps', 'iou2d_calculator',
    'ATSSAssigner'
]


def get_version():
    """获取版本信息"""
    return __version__


def get_model_info():
    """获取模型信息"""
    return {
        'name': 'GOLD-YOLO',
        'framework': 'Jittor',
        'version': __version__,
        'description': __description__,
        'author': __author__
    }


# 模块级别的配置
import jittor as jt

# 设置Jittor的一些默认配置
jt.flags.use_cuda = 1 if jt.has_cuda else 0

# 打印欢迎信息
def print_welcome():
    """打印欢迎信息"""
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    GOLD-YOLO Jittor版本                      ║
    ║                                                              ║
    ║  版本: {__version__:<50} ║
    ║  框架: Jittor                                                ║
    ║  描述: {__description__:<40} ║
    ║                                                              ║
    ║  严格对齐PyTorch版本，确保所有功能完整实现                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

# 可选择性打印欢迎信息
import os
if os.getenv('GOLD_YOLO_WELCOME', '1') == '1':
    print_welcome()
