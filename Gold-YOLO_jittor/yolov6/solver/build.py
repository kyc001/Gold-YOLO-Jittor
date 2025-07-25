# 2023.09.18-Changed for optimizer implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 优化器构建模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
import math

import jittor as jt
import jittor.nn as nn

from yolov6.utils.events import LOGGER


def build_optimizer(cfg, model):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, jt.Var):  # 使用jt.Var替代nn.Parameter
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # Jittor没有SyncBatchNorm
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, jt.Var):
            g_w.append(v.weight)
    
    assert cfg.solver.optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
    if cfg.solver.optim == 'SGD':
        optimizer = jt.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = jt.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    
    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})
    
    del g_bnw, g_w, g_b
    return optimizer


def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        LOGGER.error('unknown lr scheduler, use Cosine defaulted')
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    
    # Jittor的学习率调度器
    scheduler = JittorLambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf


class JittorLambdaLR:
    """Jittor版本的LambdaLR调度器
    
    对应PyTorch的torch.optim.lr_scheduler.LambdaLR
    """
    
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        
        self.last_epoch = last_epoch
        
        # 保存初始学习率
        for group in self.optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group.get('lr', 0.01)  # 如果没有lr，使用默认值
        
        self.step()
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, lr_lambda in zip(self.optimizer.param_groups, self.lr_lambdas):
            param_group['lr'] = param_group['initial_lr'] * lr_lambda(self.last_epoch)
    
    def get_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def get_last_lr(self):
        """获取最后一次的学习率"""
        return self.get_lr()


class JittorStepLR:
    """Jittor版本的StepLR调度器"""
    
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        
        # 保存初始学习率
        for group in self.optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']
        
        self.step()
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * (self.gamma ** (self.last_epoch // self.step_size))
    
    def get_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class JittorCosineAnnealingLR:
    """Jittor版本的CosineAnnealingLR调度器"""
    
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        
        # 保存初始学习率
        for group in self.optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']
        
        self.step()
    
    def step(self, epoch=None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min + (param_group['initial_lr'] - self.eta_min) * \
                               (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
    
    def get_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def build_optimizer_v2(args, cfg, model):
    """构建优化器的增强版本，支持更多配置选项"""
    g_bnw, g_w, g_b = [], [], []
    
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, jt.Var):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # Jittor没有SyncBatchNorm
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, jt.Var):
            g_w.append(v.weight)
    
    LOGGER.info(f"Optimizer groups: {len(g_bnw)} .bias, {len(g_w)} conv.weight, {len(g_b)} other")
    
    if cfg.solver.optim == 'SGD':
        optimizer = jt.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = jt.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    elif cfg.solver.optim == 'AdamW':
        optimizer = jt.optim.AdamW(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    else:
        raise NotImplementedError(f"Optimizer {cfg.solver.optim} not implemented")
    
    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})
    
    del g_bnw, g_w, g_b
    return optimizer


def build_lr_scheduler_v2(args, cfg, optimizer, epochs):
    """构建学习率调度器的增强版本"""
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
        scheduler = JittorLambdaLR(optimizer, lr_lambda=lf)
    elif cfg.solver.lr_scheduler == 'Linear':
        lf = lambda x: (1 - x / epochs) * (1.0 - cfg.solver.lrf) + cfg.solver.lrf
        scheduler = JittorLambdaLR(optimizer, lr_lambda=lf)
    elif cfg.solver.lr_scheduler == 'Step':
        scheduler = JittorStepLR(optimizer, step_size=cfg.solver.step_size, gamma=cfg.solver.gamma)
        lf = None
    elif cfg.solver.lr_scheduler == 'CosineAnnealing':
        scheduler = JittorCosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg.solver.lr0 * cfg.solver.lrf)
        lf = None
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
        scheduler = JittorLambdaLR(optimizer, lr_lambda=lf)
    else:
        LOGGER.error(f'Unknown lr scheduler {cfg.solver.lr_scheduler}, use Cosine defaulted')
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
        scheduler = JittorLambdaLR(optimizer, lr_lambda=lf)
    
    return scheduler, lf
