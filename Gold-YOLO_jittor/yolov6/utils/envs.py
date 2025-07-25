#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 环境设置模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
import random
import numpy as np
import jittor as jt
from yolov6.utils.events import LOGGER


def get_envs():
    """Get Jittor needed environments from system environments."""
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return local_rank, rank, world_size


def select_device(device):
    """Set devices' information to the program.
    Args:
        device: a string, like 'cpu' or '1,2,3,4'
    Returns:
        device string for Jittor
    """
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        jt.flags.use_cuda = 0
        LOGGER.info('Using CPU for training... ')
        return 'cpu'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        if jt.has_cuda:
            jt.flags.use_cuda = 1
            nd = len(device.strip().split(','))
            LOGGER.info(f'Using {nd} GPU for training... ')
            return f'cuda:{device.split(",")[0]}'
        else:
            LOGGER.warning('CUDA not available, using CPU instead')
            jt.flags.use_cuda = 0
            return 'cpu'
    
    # 默认设备选择
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        LOGGER.info('Using GPU for training... ')
        return 'cuda:0'
    else:
        jt.flags.use_cuda = 0
        LOGGER.info('Using CPU for training... ')
        return 'cpu'


def set_random_seed(seed, deterministic=False):
    """ Set random state to random library, numpy, and jittor.
    Args:
        seed: int value.
        deterministic: bool value.
    """
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    
    if deterministic:
        # Jittor的确定性设置
        jt.flags.use_cuda_random = False
        LOGGER.info(f'Set random seed to {seed} with deterministic mode')
    else:
        LOGGER.info(f'Set random seed to {seed}')


def init_seeds(seed=0):
    """Initialize random seeds for reproducibility."""
    set_random_seed(seed, deterministic=True)
