#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 指数移动平均模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import math
from copy import deepcopy
import jittor as jt
import jittor.nn as nn


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        # Jittor不需要requires_grad_(False)，参数默认不需要梯度
        for param in self.ema.parameters():
            param.stop_grad()  # Jittor的停止梯度方法

    def update(self, model):
        # Jittor不需要torch.no_grad()
        self.updates += 1
        decay = self.decay(self.updates)

        state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
        for k, item in self.ema.state_dict().items():
            if item.dtype in [jt.float32, jt.float64, jt.float16]:  # Jittor的浮点类型检查
                item *= decay
                item += (1 - decay) * state_dict[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from one instance and set them to another instance."""
    for k, item in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, item)


def is_parallel(model):
    '''Return True if model's type is DP or DDP, else False.'''
    # Jittor的并行模型检查
    return hasattr(model, 'module') and isinstance(model.module, nn.Module)


def de_parallel(model):
    '''De-parallelize a model. Return single-GPU model if model's type is DP or DDP.'''
    return model.module if is_parallel(model) else model


class JittorModelEMA:
    """Jittor专用的EMA实现，针对Jittor特性优化"""
    
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """
        Args:
            model: 要应用EMA的模型
            decay: 衰减率
            tau: 衰减时间常数
            updates: 更新次数
        """
        # 创建EMA模型的深拷贝
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = decay
        self.tau = tau
        
        # 停止EMA模型的梯度计算
        for p in self.ema.parameters():
            p.stop_grad()
    
    def update(self, model):
        """更新EMA权重"""
        self.updates += 1
        d = self.decay_fn()
        
        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_float():
                v *= d
                v += (1 - d) * msd[k].detach()
    
    def decay_fn(self):
        """计算衰减率"""
        return self.decay * (1 - math.exp(-self.updates / self.tau))
    
    def copy_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """复制模型属性"""
        copy_attr(self.ema, model, include, exclude)
    
    def clone(self):
        """克隆EMA模型"""
        return deepcopy(self.ema)
    
    def state_dict(self):
        """获取EMA模型的状态字典"""
        return self.ema.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载EMA模型的状态字典"""
        self.ema.load_state_dict(state_dict)


def update_ema(ema_model, model, decay=0.9999):
    """
    更新指数移动平均模型
    
    Args:
        ema_model: EMA模型
        model: 当前训练模型
        decay: 衰减率
    """
    with jt.no_grad():
        msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            if ema_v.dtype.is_float():
                model_v = msd[k].detach()
                ema_v.copy_(ema_v * decay + (1 - decay) * model_v)


def apply_ema(model, ema_model):
    """
    将EMA权重应用到模型
    
    Args:
        model: 目标模型
        ema_model: EMA模型
    """
    with jt.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k in msd.keys():
            if k in esd and esd[k].dtype.is_float():
                msd[k].copy_(esd[k])


class EMAWrapper(nn.Module):
    """EMA包装器，可以直接替换原模型"""
    
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.model = model
        self.ema = JittorModelEMA(model, decay)
        self.device = device
    
    def execute(self, *args, **kwargs):
        """前向传播使用EMA模型"""
        return self.ema.ema(*args, **kwargs)
    
    def update(self):
        """更新EMA权重"""
        self.ema.update(self.model)
    
    def train(self, mode=True):
        """设置训练模式"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self.model.eval()
        self.ema.ema.eval()
        return self
    
    def state_dict(self):
        """获取状态字典"""
        return {
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'updates': self.ema.updates
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.model.load_state_dict(state_dict['model'])
        self.ema.load_state_dict(state_dict['ema'])
        self.ema.updates = state_dict.get('updates', 0)
