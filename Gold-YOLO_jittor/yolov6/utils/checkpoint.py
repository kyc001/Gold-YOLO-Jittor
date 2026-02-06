# 2023.09.18-Changed for checkpoint load implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 检查点保存和加载模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
import shutil
import jittor as jt
import os.path as osp
from yolov6.utils.events import LOGGER
from yolov6.utils.jittor_utils import fuse_model


def _is_state_dict_like(obj):
    return isinstance(obj, dict) and len(obj) > 0 and all(hasattr(v, "shape") for v in obj.values())


def _extract_state_dict_from_ckpt(ckpt):
    """Extract a plain state_dict from multiple checkpoint layouts."""
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            return ckpt['model_state_dict']
        if 'ema_state_dict' in ckpt:
            return ckpt['ema_state_dict']
        if 'state_dict' in ckpt:
            return ckpt['state_dict']
        if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
            return ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'float') else ckpt['model'].state_dict()
        if 'ema' in ckpt and hasattr(ckpt['ema'], 'state_dict'):
            return ckpt['ema'].float().state_dict() if hasattr(ckpt['ema'], 'float') else ckpt['ema'].state_dict()
        if 'ema' in ckpt and _is_state_dict_like(ckpt['ema']):
            return ckpt['ema']
        if 'model' in ckpt and _is_state_dict_like(ckpt['model']):
            return ckpt['model']
        if _is_state_dict_like(ckpt):
            return ckpt
    elif hasattr(ckpt, 'state_dict'):
        return ckpt.float().state_dict()
    raise ValueError("Unsupported checkpoint format: cannot extract state_dict.")


def _build_model_from_state_dict(state_dict, ckpt_meta=None, map_location=None):
    """Rebuild model from state_dict for checkpoints that do not serialize nn.Module."""
    from yolov6.utils.config import Config
    from yolov6.models.yolo import build_model

    cfg_path = './configs/gold_yolo-n.py'
    if isinstance(ckpt_meta, dict):
        cfg_path = ckpt_meta.get('config', ckpt_meta.get('cfg', cfg_path))

    if not osp.exists(cfg_path):
        cfg_path = './configs/gold_yolo-n.py'
    cfg = Config.fromfile(cfg_path)

    nc = 80
    if isinstance(ckpt_meta, dict):
        if 'nc' in ckpt_meta and ckpt_meta['nc'] is not None:
            nc = int(ckpt_meta['nc'])
        elif isinstance(ckpt_meta.get('names', None), list):
            nc = len(ckpt_meta['names'])

    device = map_location if map_location is not None else 'cpu'
    model = build_model(cfg, nc, device)
    model.load_state_dict(state_dict)

    if isinstance(ckpt_meta, dict) and isinstance(ckpt_meta.get('names', None), list):
        model.names = ckpt_meta['names']
    if isinstance(ckpt_meta, dict) and 'stride' in ckpt_meta:
        try:
            model.stride = ckpt_meta['stride']
        except Exception:
            pass
    return model


def load_state_dict(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    ckpt = jt.load(weights)  # Jittor使用jt.load
    state_dict = _extract_state_dict_from_ckpt(ckpt)
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict)  # Jittor不需要strict参数
    del ckpt, state_dict, model_state_dict
    return model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = jt.load(weights)  # load

    model = None
    if isinstance(ckpt, dict):
        model_key = 'ema' if ckpt.get('ema') is not None else 'model'
        if model_key in ckpt and hasattr(ckpt[model_key], 'float'):
            model = ckpt[model_key].float()
        elif model_key in ckpt and _is_state_dict_like(ckpt[model_key]):
            model = _build_model_from_state_dict(ckpt[model_key], ckpt_meta=ckpt, map_location=map_location)
        elif 'model' in ckpt and hasattr(ckpt['model'], 'float'):
            model = ckpt['model'].float()
        else:
            state_dict = _extract_state_dict_from_ckpt(ckpt)
            model = _build_model_from_state_dict(state_dict, ckpt_meta=ckpt, map_location=map_location)
    elif hasattr(ckpt, 'float'):
        model = ckpt.float()
    else:
        raise ValueError(f"Unsupported checkpoint format in {weights}")

    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def load_checkpoint_2(model, weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = jt.load(weights)
    state_dict = _extract_state_dict_from_ckpt(ckpt)
    model.load_state_dict(state_dict)
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + '.pkl')  # Jittor使用.pkl扩展名
    jt.save(ckpt, filename)  # 使用jt.save
    if is_best:
        best_filename = osp.join(save_dir, 'best_ckpt.pkl')
        shutil.copyfile(filename, best_filename)


def strip_optimizer(ckpt_dir, epoch, strip_last=False):
    """Delete optimizer states from checkpoints.

    By default only strips `best_ckpt.pkl` and keeps `last_ckpt.pkl`
    resumable with optimizer state.
    """
    targets = ['best'] + (['last'] if strip_last else [])
    for s in targets:
        ckpt_path = osp.join(ckpt_dir, '{}_ckpt.pkl'.format(s))  # 使用.pkl扩展名
        if not osp.exists(ckpt_path):
            continue
        ckpt = jt.load(ckpt_path)
        if ckpt.get('ema'):
            ckpt['model'] = ckpt['ema']  # replace model with ema
        for k in ['optimizer', 'ema', 'updates']:  # keys
            ckpt[k] = None
        ckpt['epoch'] = epoch
        if hasattr(ckpt.get('model', None), 'half'):
            ckpt['model'].half()  # to FP16
            for p in ckpt['model'].parameters():
                p.stop_grad()  # Jittor的停止梯度方法
        jt.save(ckpt, ckpt_path)


def save_model_state(model, filepath, epoch=None, optimizer=None, ema=None, **kwargs):
    """
    保存模型状态的通用函数
    
    Args:
        model: 要保存的模型
        filepath: 保存路径
        epoch: 当前epoch
        optimizer: 优化器状态
        ema: EMA模型状态
        **kwargs: 其他要保存的信息
    """
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    
    if ema is not None:
        state['ema'] = ema.state_dict() if hasattr(ema, 'state_dict') else ema
    
    # 添加其他信息
    state.update(kwargs)
    
    # 确保目录存在
    os.makedirs(osp.dirname(filepath), exist_ok=True)
    
    # 保存
    jt.save(state, filepath)
    LOGGER.info(f"Model saved to {filepath}")


def load_model_state(filepath, model=None, optimizer=None, ema=None, strict=True):
    """
    加载模型状态的通用函数
    
    Args:
        filepath: 检查点文件路径
        model: 要加载权重的模型
        optimizer: 要加载状态的优化器
        ema: 要加载状态的EMA
        strict: 是否严格匹配键名
    
    Returns:
        dict: 包含加载信息的字典
    """
    if not osp.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    LOGGER.info(f"Loading checkpoint from {filepath}")
    checkpoint = jt.load(filepath)
    
    result = {'epoch': checkpoint.get('epoch', 0)}
    
    # 加载模型权重
    if model is not None and 'model' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model'])
            LOGGER.info("Model weights loaded successfully")
        except Exception as e:
            if strict:
                raise e
            else:
                LOGGER.warning(f"Model loading warning: {e}")
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            LOGGER.info("Optimizer state loaded successfully")
        except Exception as e:
            LOGGER.warning(f"Optimizer loading warning: {e}")
    
    # 加载EMA状态
    if ema is not None and 'ema' in checkpoint:
        try:
            if hasattr(ema, 'load_state_dict'):
                ema.load_state_dict(checkpoint['ema'])
            else:
                # 如果ema是简单的模型对象
                ema = checkpoint['ema']
            LOGGER.info("EMA state loaded successfully")
        except Exception as e:
            LOGGER.warning(f"EMA loading warning: {e}")
    
    return result


class CheckpointManager:
    """检查点管理器，用于自动保存和管理检查点"""
    
    def __init__(self, save_dir, max_keep=5, save_interval=1):
        """
        Args:
            save_dir: 保存目录
            max_keep: 最多保留的检查点数量
            save_interval: 保存间隔（epoch）
        """
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.save_interval = save_interval
        self.best_metric = 0.0
        self.checkpoints = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, epoch, metric=None, optimizer=None, ema=None, is_best=False, **kwargs):
        """
        保存检查点
        
        Args:
            model: 模型
            epoch: 当前epoch
            metric: 当前指标值
            optimizer: 优化器
            ema: EMA模型
            is_best: 是否是最佳模型
            **kwargs: 其他信息
        """
        # 保存最新检查点
        latest_path = osp.join(self.save_dir, 'latest.pkl')
        save_model_state(model, latest_path, epoch, optimizer, ema, metric=metric, **kwargs)
        
        # 定期保存检查点
        if epoch % self.save_interval == 0:
            epoch_path = osp.join(self.save_dir, f'epoch_{epoch}.pkl')
            save_model_state(model, epoch_path, epoch, optimizer, ema, metric=metric, **kwargs)
            self.checkpoints.append((epoch, epoch_path, metric or 0.0))
            
            # 清理旧检查点
            self._cleanup_checkpoints()
        
        # 保存最佳模型
        if is_best or (metric is not None and metric > self.best_metric):
            self.best_metric = metric or self.best_metric
            best_path = osp.join(self.save_dir, 'best.pkl')
            save_model_state(model, best_path, epoch, optimizer, ema, metric=metric, **kwargs)
            LOGGER.info(f"New best model saved with metric: {self.best_metric}")
    
    def _cleanup_checkpoints(self):
        """清理旧的检查点文件"""
        if len(self.checkpoints) > self.max_keep:
            # 按指标排序，保留最好的几个
            self.checkpoints.sort(key=lambda x: x[2], reverse=True)
            
            # 删除多余的检查点
            for epoch, path, metric in self.checkpoints[self.max_keep:]:
                if osp.exists(path):
                    os.remove(path)
                    LOGGER.info(f"Removed checkpoint: {path}")
            
            self.checkpoints = self.checkpoints[:self.max_keep]
    
    def load_latest(self, model=None, optimizer=None, ema=None):
        """加载最新的检查点"""
        latest_path = osp.join(self.save_dir, 'latest.pkl')
        if osp.exists(latest_path):
            return load_model_state(latest_path, model, optimizer, ema)
        return None
    
    def load_best(self, model=None, optimizer=None, ema=None):
        """加载最佳检查点"""
        best_path = osp.join(self.save_dir, 'best.pkl')
        if osp.exists(best_path):
            return load_model_state(best_path, model, optimizer, ema)
        return None
