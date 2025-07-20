#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Common layers for YOLOv6 Jittor implementation
"""

import jittor as jt
from jittor import nn
import math


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def execute_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution"""
    
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, d=d, g=math.gcd(c1, c2), act=act)


class ConvWrapper(nn.Module):
    """Wrapper for convolution with different activations"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, act=act)
    
    def execute(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def execute(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    
    def execute(self, x):
        return self.cv3(jt.concat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher"""
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def execute(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(jt.concat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    """Focus wh information into c-space"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
    
    def execute(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(jt.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Contract(nn.Module):
    """Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)"""
    
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain
    
    def execute(self, x):
        b, c, h, w = x.shape
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):
    """Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)"""
    
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain
    
    def execute(self, x):
        b, c, h, w = x.shape
        s = self.gain
        x = x.view(b, c // (s * s), s, s, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.view(b, c // (s * s), h * s, w * s)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension"""
    
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def execute(self, x):
        return jt.concat(x, self.d)


class DetectBackend(nn.Module):
    """YOLOv6 MultiBackend class for python inference on various backends"""
    
    def __init__(self, weights='yolov6s.pt', device=None, dnn=False, data=None, fp16=False):
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        self.pt = True  # PyTorch/Jittor
        
        # Load model
        model = jt.load(w) if w.endswith('.pkl') else None
        if model:
            model = model['model'] if 'model' in model else model
            self.model = model
            self.names = model.names if hasattr(model, 'names') else None
            self.stride = model.stride if hasattr(model, 'stride') else jt.array([32.])
        else:
            self.model = None
            self.names = None
            self.stride = jt.array([32.])
    
    def execute(self, im, augment=False, visualize=False):
        if self.model:
            return self.model(im)
        else:
            return jt.zeros((1, 25200, 85))  # dummy output


def autopad(k, p=None, d=1):
    """Auto-pad to 'same' shape outputs"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor"""
    if isinstance(divisor, jt.Var):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_anchor_order(m):
    """Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary"""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def initialize_weights(model):
    """Initialize model weights to random values"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
