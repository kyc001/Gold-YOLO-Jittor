#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
真实VOC数据加载器 - Jittor版本
加载真实的VOC2012子集数据进行训练
"""

import os
import cv2
import numpy as np
import jittor as jt
from PIL import Image
import random


class VOCDataset:
    """VOC数据集加载器"""
    
    def __init__(self, data_dir, img_size=640, batch_size=8, augment=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.augment = augment
        
        # 获取所有图像文件
        self.img_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        
        self.img_files = []
        self.label_files = []
        
        # 扫描所有图像文件
        for img_file in os.listdir(self.img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.img_dir, img_file)
                label_path = os.path.join(self.label_dir, img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
                
                if os.path.exists(label_path):
                    self.img_files.append(img_path)
                    self.label_files.append(label_path)
        
        print(f"✅ 加载VOC数据集: {len(self.img_files)}张图像")
        
        # 打乱数据
        combined = list(zip(self.img_files, self.label_files))
        random.shuffle(combined)
        self.img_files, self.label_files = zip(*combined)
    
    def __len__(self):
        return len(self.img_files) // self.batch_size
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration
        
        # 获取一个batch的数据
        start_idx = self.current_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.img_files))
        
        batch_imgs = []
        batch_targets = []
        
        for i in range(start_idx, end_idx):
            img, targets = self.load_item(i)
            batch_imgs.append(img)
            
            # 添加batch索引
            if len(targets) > 0:
                batch_idx = jt.full((len(targets), 1), i - start_idx)
                targets = jt.cat([batch_idx, targets], dim=1)
                batch_targets.append(targets)
        
        # 转换为tensor
        batch_imgs = jt.stack(batch_imgs)
        
        # 合并所有targets
        if batch_targets:
            batch_targets = jt.cat(batch_targets, dim=0)
        else:
            batch_targets = jt.zeros((0, 6))
        
        self.current_idx += 1
        return batch_imgs, batch_targets
    
    def load_item(self, idx):
        """加载单个数据项"""
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # 加载图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 调整图像尺寸
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        # 加载标签
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        targets.append([class_id, x_center, y_center, width, height])
        
        # 转换为tensor
        img = jt.array(img)
        if targets:
            targets = jt.array(targets)
        else:
            targets = jt.zeros((0, 5))
        
        return img, targets


def create_real_dataloader(data_dir, img_size=640, batch_size=8, augment=True):
    """创建真实数据加载器"""
    return VOCDataset(data_dir, img_size, batch_size, augment)


# 测试数据加载器
if __name__ == "__main__":
    data_dir = "/home/kyc/project/GOLD-YOLO/data/voc2012_subset"
    
    print("🧪 测试真实数据加载器...")
    dataloader = create_real_dataloader(data_dir, batch_size=4)
    
    print(f"数据集大小: {len(dataloader)} batches")
    
    # 测试加载一个batch
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  图像形状: {list(images.shape)}")
        print(f"  目标形状: {list(targets.shape)}")
        print(f"  目标数量: {len(targets)}")
        
        if len(targets) > 0:
            print(f"  目标示例: {targets[:3]}")
        
        if batch_idx >= 2:  # 只测试前3个batch
            break
    
    print("✅ 真实数据加载器测试完成！")
