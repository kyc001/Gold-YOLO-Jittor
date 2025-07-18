# Jittor implementation of activation functions for Gold-YOLO
# Migrated from PyTorch version

import jittor as jt
from jittor import nn


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) activation function"""
    
    def execute(self, x):
        return x * jt.sigmoid(x)


class ReLU6(nn.Module):
    """ReLU6 activation function"""
    
    def execute(self, x):
        return jt.clamp(x, min_v=0, max_v=6)


# Alias for compatibility
Swish = SiLU
