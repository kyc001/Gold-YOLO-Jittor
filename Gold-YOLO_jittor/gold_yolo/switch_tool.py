# 2023.09.18-Changed for switch tool implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
GOLD-YOLO Jittor版本 - 模型切换工具
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import copy
import jittor as jt
import jittor.nn as nn


def switch_to_deploy(model):
    """
    切换模型到部署模式
    将训练时的多分支结构转换为单分支结构，提高推理效率
    """
    for layer in model.modules():
        if hasattr(layer, 'switch_to_deploy'):
            layer.switch_to_deploy()
    return model


def convert_checkpoint_False(model):
    """
    转换检查点，设置为非部署模式
    """
    for layer in model.modules():
        if hasattr(layer, 'deploy'):
            layer.deploy = False
    return model


def convert_checkpoint_True(model):
    """
    转换检查点，设置为部署模式
    """
    for layer in model.modules():
        if hasattr(layer, 'deploy'):
            layer.deploy = True
    return model


def fuse_conv_and_bn(conv, bn):
    """
    融合卷积层和BatchNorm层
    
    Args:
        conv: 卷积层
        bn: BatchNorm层
    
    Returns:
        融合后的卷积层
    """
    # 获取卷积层参数
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else jt.zeros_like(bn.running_mean)
    
    # 获取BatchNorm参数
    bn_w = bn.weight
    bn_b = bn.bias
    bn_rm = bn.running_mean
    bn_rv = bn.running_var
    bn_eps = bn.eps
    
    # 计算融合参数
    bn_var_rsqrt = jt.rsqrt(bn_rv + bn_eps)
    
    # 融合权重
    fused_weight = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    
    # 融合偏置
    fused_bias = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    
    # 创建新的卷积层
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True
    )
    
    # 设置融合后的参数
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    
    return fused_conv


def fuse_repvgg_block(block):
    """
    融合RepVGG块的多分支结构
    
    Args:
        block: RepVGG块
    
    Returns:
        融合后的卷积层
    """
    if hasattr(block, 'rbr_reparam'):
        # 已经融合过了
        return block.rbr_reparam
    
    # 获取各分支的等效卷积核
    kernel3x3, bias3x3 = block._get_equivalent_kernel_bias()
    
    # 创建融合后的卷积层
    fused_conv = nn.Conv2d(
        block.in_channels,
        block.out_channels,
        block.kernel_size,
        block.stride,
        block.padding,
        bias=True
    )
    
    fused_conv.weight.data = kernel3x3
    fused_conv.bias.data = bias3x3
    
    return fused_conv


class ModelFuser:
    """模型融合器，用于将训练模型转换为推理模型"""
    
    def __init__(self):
        self.fused_modules = {}
    
    def fuse_model(self, model):
        """
        融合整个模型
        
        Args:
            model: 要融合的模型
        
        Returns:
            融合后的模型
        """
        model_copy = copy.deepcopy(model)
        self._fuse_modules(model_copy)
        return model_copy
    
    def _fuse_modules(self, module):
        """递归融合模块"""
        for name, child in module.named_children():
            if self._is_fusable_conv_bn(child):
                # 融合Conv+BN
                fused = self._fuse_conv_bn_module(child)
                setattr(module, name, fused)
            elif hasattr(child, 'switch_to_deploy'):
                # RepVGG类型的块
                child.switch_to_deploy()
            else:
                # 递归处理子模块
                self._fuse_modules(child)
    
    def _is_fusable_conv_bn(self, module):
        """检查是否是可融合的Conv+BN结构"""
        if not isinstance(module, nn.Sequential):
            return False
        
        if len(module) != 2:
            return False
        
        return (isinstance(module[0], nn.Conv2d) and 
                isinstance(module[1], nn.BatchNorm2d))
    
    def _fuse_conv_bn_module(self, module):
        """融合Conv+BN模块"""
        conv = module[0]
        bn = module[1]
        return fuse_conv_and_bn(conv, bn)


def optimize_for_inference(model):
    """
    为推理优化模型
    
    Args:
        model: 要优化的模型
    
    Returns:
        优化后的模型
    """
    # 设置为评估模式
    model.eval()
    
    # 切换到部署模式
    model = switch_to_deploy(model)
    
    # 融合可融合的层
    fuser = ModelFuser()
    model = fuser.fuse_model(model)
    
    return model


def convert_weights(pytorch_weights_path, jittor_weights_path):
    """
    转换PyTorch权重到Jittor格式
    
    Args:
        pytorch_weights_path: PyTorch权重文件路径
        jittor_weights_path: Jittor权重保存路径
    """
    try:
        import torch
        # 加载PyTorch权重
        pytorch_state = torch.load(pytorch_weights_path, map_location='cpu')
        
        # 转换为Jittor格式
        jittor_state = {}
        for key, value in pytorch_state.items():
            if isinstance(value, torch.Tensor):
                jittor_state[key] = jt.array(value.numpy())
            else:
                jittor_state[key] = value
        
        # 保存Jittor权重
        jt.save(jittor_state, jittor_weights_path)
        print(f"Successfully converted weights from {pytorch_weights_path} to {jittor_weights_path}")
        
    except ImportError:
        print("PyTorch not available, cannot convert weights")
    except Exception as e:
        print(f"Error converting weights: {e}")


def load_pretrained_weights(model, weights_path, strict=True):
    """
    加载预训练权重
    
    Args:
        model: 要加载权重的模型
        weights_path: 权重文件路径
        strict: 是否严格匹配键名
    
    Returns:
        加载权重后的模型
    """
    try:
        state_dict = jt.load(weights_path)
        
        # 如果是完整的检查点，提取模型权重
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # 加载权重
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {weights_path}")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        if strict:
            raise e
    
    return model
