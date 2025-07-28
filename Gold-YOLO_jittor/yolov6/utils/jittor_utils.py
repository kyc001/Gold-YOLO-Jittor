#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - Jittor工具函数
从PyTorch版本迁移到Jittor框架
"""

import time
from contextlib import contextmanager
from copy import deepcopy
import jittor as jt
import jittor.nn as nn
from yolov6.utils.events import LOGGER

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


@contextmanager
def jittor_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    Note: Jittor handles distributed training differently than PyTorch
    """
    # Jittor的分布式训练处理方式不同，这里简化实现
    yield


def time_sync():
    '''Waits for all kernels to complete if CUDA is available.'''
    # Jittor会自动同步，这里简化实现
    jt.sync_all()
    return time.time()


def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # Jittor的Conv2d默认初始化通常已经足够好
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            # Jittor的激活函数不需要设置inplace参数
            pass


def fuse_conv_and_bn(conv, bn):
    '''Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/.'''
    fusedconv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
    )
    
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = jt.diag(bn.weight.div(jt.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight = jt.matmul(w_bn, w_conv).view(fusedconv.weight.shape)
    
    # prepare spatial bias
    b_conv = (
            jt.zeros(conv.weight.size(0))
            if conv.bias is None
            else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            jt.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias = jt.matmul(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    
    return fusedconv


def fuse_model(model):
    '''Fuse convolution and batchnorm layers of the model.'''
    from yolov6.layers.common import Conv, SimConv, Conv_C3
    
    for m in model.modules():
        if (type(m) is Conv or type(m) is SimConv or type(m) is Conv_C3) and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.execute = m.execute_fuse  # update forward (Jittor uses execute instead of forward)
    return model


def get_model_info(model, img_size=640):
    """Get model Params and GFlops.
    Code base on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/model_utils.py
    """
    try:
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        
        # 计算参数数量
        n_p = sum(x.numel() for x in model.parameters())  # number parameters
        n_g = sum(x.numel() for x in model.parameters() if len(x.shape) > 1)  # number gradients
        
        # 尝试计算FLOPs
        try:
            if thop:
                flops = thop.profile(model, inputs=(jt.zeros(1, 3, *img_size),), verbose=False)[0] / 1E9 * 2
            else:
                flops = 0
        except:
            flops = 0
        
        LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients, {flops:.1f} GFLOPs")
        
    except Exception as e:
        LOGGER.warning(f"Model info calculation failed: {e}")


def model_info(model, verbose=False, img_size=640):
    """Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]"""
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if len(x.shape) > 1)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        if thop:
            img = jt.zeros((1, 3, img_size, img_size))
            flops = thop.profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        else:
            flops = 0
    except:
        flops = 0

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients, {flops:.1f} GFLOPs")
    return n_p, flops


def intersect_dicts(da, db, exclude=()):
    """Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values"""
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """Scales img(bs,3,y,x) by ratio constrained to gs-multiple"""
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = nn.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x / gs) * gs for x in (h, w))
        return nn.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447, mode='constant')  # value = imagenet mean
