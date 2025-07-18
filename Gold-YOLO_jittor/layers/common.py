# Jittor implementation of common layers for Gold-YOLO
# Migrated from PyTorch version

import jittor as jt
from jittor import nn
import numpy as np
from .activations import SiLU, ReLU6


class Conv(nn.Module):
    """Standard convolution with SiLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU()
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def execute_fuse(self, x):
        return self.act(self.conv(x))


class SimConv(nn.Module):
    """Simple convolution with ReLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def execute_fuse(self, x):
        return self.act(self.conv(x))


class ConvWrapper(nn.Module):
    """Wrapper for normal Conv with SiLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = Conv(in_channels, out_channels, kernel_size, stride, groups, bias)
    
    def execute(self, x):
        return self.block(x)


class SimConvWrapper(nn.Module):
    """Wrapper for normal Conv with ReLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = SimConv(in_channels, out_channels, kernel_size, stride, groups, bias)
    
    def execute(self, x):
        return self.block(x)


class SimSPPF(nn.Module):
    """Simplified SPPF with ReLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def execute(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(jt.concat([x, y1, y2, self.m(y2)], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def execute(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(jt.concat([x, y1, y2, self.m(y2)], dim=1))


class RepVGGBlock(nn.Module):
    """RepVGG Block for efficient inference"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.act = nn.ReLU()

    def execute(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def switch_to_deploy(self):
        """Switch to deployment mode"""
        if hasattr(self, 'rbr_reparam'):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Delete original branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

    def get_equivalent_kernel_bias(self):
        """Get equivalent kernel and bias for deployment"""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 kernel to 3x3"""
        if kernel1x1 is None:
            return 0
        else:
            return jt.nn.pad(kernel1x1, (1, 1, 1, 1))

    def _fuse_bn_tensor(self, branch):
        """Fuse conv and bn"""
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = jt.zeros((self.in_channels, input_dim, 3, 3))
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepBlock(nn.Module):
    """RepBlock with multiple RepVGGBlocks"""

    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock):
        super().__init__()
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else nn.Identity()

    def execute(self, x):
        x = self.conv1(x)
        x = self.block(x)
        return x


class BepC3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = block(in_channels, c_, 1, 1)
        self.cv2 = block(in_channels, c_, 1, 1)
        self.cv3 = block(2 * c_, out_channels, 1, 1)
        self.m = nn.Sequential(*(block(c_, c_) for _ in range(n)))

    def execute(self, x):
        return self.cv3(jt.concat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class BottleRep(nn.Module):
    """Bottleneck with RepVGGBlock"""

    def __init__(self, in_channels, out_channels, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = RepVGGBlock(in_channels, c_, 1, 1)
        self.cv2 = RepVGGBlock(c_, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels

    def execute(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SimCSPSPPF(nn.Module):
    """CSP SPPF with simple convolutions"""

    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super().__init__()
        c_ = int(in_channels * e)  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(in_channels, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(2 * c_, out_channels, 1, 1)

    def execute(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv6(jt.concat([y0, self.cv5(jt.concat([x1, y1, y2, y3], dim=1))], dim=1))


class CSPSPPF(nn.Module):
    """CSP SPPF with standard convolutions"""

    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super().__init__()
        c_ = int(in_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(2 * c_, out_channels, 1, 1)

    def execute(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv6(jt.concat([y0, self.cv5(jt.concat([x1, y1, y2, y3], dim=1))], dim=1))
