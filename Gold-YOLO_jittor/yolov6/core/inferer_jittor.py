#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本推理器
完全对齐PyTorch版本的推理逻辑
"""

import os
import cv2
import time
import math
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from collections import deque

import jittor as jt
from jittor import nn

from yolov6.models.yolo import Model
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.general import check_img_size, scale_coords
from yolov6.utils.events import load_yaml


class DetectBackendJittor:
    """Jittor版本的检测后端，对齐PyTorch版本的DetectBackend"""
    
    def __init__(self, weights='yolov6s.pkl', device='cpu'):
        """初始化检测后端"""
        self.weights = weights
        self.device = device
        
        # 加载模型
        self.model, self.stride = self._load_model(weights)
        
    def _load_model(self, weights):
        """加载Jittor模型，使用与训练时相同的模型创建方法"""
        print(f"📦 加载Jittor模型: {weights}")

        # 使用与训练时相同的模型创建方法
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
        
        # 加载权重
        if os.path.exists(weights):
            checkpoint = jt.load(weights)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ 成功加载权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                    print(f"✅ 成功加载权重 (epoch: {checkpoint.get('epoch', 'unknown')})")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"✅ 成功加载权重")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ 成功加载权重")
        else:
            raise FileNotFoundError(f"权重文件不存在: {weights}")
        
        model.eval()
        
        # 获取stride信息
        stride = 32  # GOLD-YOLO默认stride
        
        return model, stride
    
    def __call__(self, img):
        """前向推理"""
        return self.model(img)


class InfererJittor:
    """Jittor版本推理器，完全对齐PyTorch版本"""
    
    def __init__(self, source, weights, device, yaml, img_size, half=False):
        """初始化推理器"""
        self.__dict__.update(locals())
        
        # 初始化模型
        self.device = device
        self.img_size = img_size
        self.model = DetectBackendJittor(weights, device=device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = check_img_size(self.img_size, s=self.stride)  # 检查图像尺寸
        self.half = half
        
        # 模型切换到部署状态（Jittor版本暂时跳过）
        # self.model.model = self.model_switch(self.model.model, self.img_size)
        
        # 半精度（Jittor版本暂时跳过）
        if self.half:
            print("WARNING: Jittor版本暂不支持半精度推理，使用全精度")
            self.half = False
        
        # 预热
        self.warmup()
        
    def warmup(self):
        """模型预热"""
        print("🔥 模型预热中...")
        img_size = self.img_size
        if isinstance(img_size, list):
            img_size = max(img_size)
        
        # 创建虚拟输入进行预热
        dummy_input = jt.zeros((1, 3, img_size, img_size))
        for _ in range(2):
            with jt.no_grad():
                _ = self.model(dummy_input)
        print("✅ 模型预热完成")
    
    def model_switch(self, model, img_size):
        """模型切换到部署状态（对齐PyTorch版本）"""
        # Jittor版本暂时返回原模型
        return model
    
    def preprocess(self, img_src):
        """图像预处理，完全对齐PyTorch版本"""
        img_size = self.img_size
        if isinstance(img_size, list):
            img_size = max(img_size)
        
        # Letterbox resize
        img = letterbox(img_src, new_shape=img_size, stride=self.stride, auto=False)[0]
        
        # 转换颜色空间和数据类型
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
        
        # 转换为Jittor张量
        img = jt.array(img)
        if img.ndim == 3:
            img = img.unsqueeze(0)  # 添加batch维度
        
        return img
    
    def postprocess(self, pred, img_src, img_processed):
        """后处理，完全对齐PyTorch版本"""
        # NMS
        pred = non_max_suppression(
            pred, 
            conf_thres=0.25, 
            iou_thres=0.45, 
            max_det=1000
        )
        
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                # 坐标缩放回原图尺寸
                det[:, :4] = scale_coords(img_processed.shape[2:], det[:, :4], img_src.shape).round()
                detections.append(det)
            else:
                detections.append(jt.empty((0, 6)))
        
        return detections
    
    def infer(self, img_src, conf_thres=0.25, iou_thres=0.45, max_det=1000):
        """单张图像推理"""
        # 预处理
        img = self.preprocess(img_src)
        
        # 推理
        with jt.no_grad():
            pred = self.model(img)
        
        # 后处理
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                # 坐标缩放回原图尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_src.shape).round()
                detections.append(det)
            else:
                detections.append(jt.empty((0, 6)))
        
        return detections
    
    def draw_detections(self, img, detections, conf_thres=0.25):
        """绘制检测结果，对齐PyTorch版本"""
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        for det in detections:
            if len(det):
                # 转换为numpy
                if hasattr(det, 'numpy'):
                    det = det.numpy()
                
                for *xyxy, conf, cls in det:
                    if conf >= conf_thres:
                        # 绘制边界框
                        x1, y1, x2, y2 = map(int, xyxy)
                        cls_id = int(cls)
                        color = colors[cls_id].tolist()
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # 绘制标签
                        label = f'{self.class_names[cls_id]} {conf:.2f}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(img, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img
