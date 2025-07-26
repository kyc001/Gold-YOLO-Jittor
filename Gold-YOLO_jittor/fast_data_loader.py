#!/usr/bin/env python3
"""
高性能数据加载器 - 解决CPU高GPU低问题
"""

import os
import cv2
import numpy as np
import jittor as jt
from PIL import Image
import random
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time

class FastVOCDataset:
    """高性能VOC数据集加载器"""
    
    def __init__(self, data_dir, img_size=640, batch_size=8, num_workers=8, prefetch_factor=4):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
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
        
        print(f"✅ 高性能加载VOC数据集: {len(self.img_files)}张图像")
        print(f"   workers: {num_workers}, prefetch: {prefetch_factor}")
        
        # 打乱数据
        combined = list(zip(self.img_files, self.label_files))
        random.shuffle(combined)
        self.img_files, self.label_files = zip(*combined)
        
        # 预处理缓存
        self.cache = {}
        self.cache_size = min(1000, len(self.img_files))  # 缓存前1000张图像
        
        # 预加载数据
        self._preload_data()
    
    def _preload_data(self):
        """预加载部分数据到内存"""
        print(f"🚀 预加载 {self.cache_size} 张图像到内存...")
        
        def load_single(idx):
            return idx, self._load_and_process(idx)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(load_single, i) for i in range(min(self.cache_size, len(self.img_files)))]
            
            for future in futures:
                idx, (img, targets) = future.result()
                self.cache[idx] = (img, targets)
        
        print(f"✅ 预加载完成: {len(self.cache)} 张图像")
    
    def _load_and_process(self, idx):
        """加载和预处理单个数据项"""
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # 加载图像 - 使用PIL更快
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
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
        
        # 转换为numpy数组
        if targets:
            targets = np.array(targets, dtype=np.float32)
        else:
            targets = np.zeros((0, 5), dtype=np.float32)
        
        return img, targets
    
    def __len__(self):
        return len(self.img_files) // self.batch_size
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration
        
        return self._get_batch()
    
    def _get_batch(self):
        """快速获取batch数据"""
        batch_imgs = []
        batch_targets = []
        
        start_idx = self.current_idx * self.batch_size
        
        for i in range(self.batch_size):
            idx = start_idx + i
            if idx >= len(self.img_files):
                break
            
            # 从缓存获取或实时加载
            if idx in self.cache:
                img, targets = self.cache[idx]
            else:
                img, targets = self._load_and_process(idx)
            
            batch_imgs.append(img)
            
            # 添加batch索引
            if len(targets) > 0:
                batch_indices = np.full((len(targets), 1), i, dtype=np.float32)
                targets_with_batch = np.concatenate([batch_indices, targets], axis=1)
                batch_targets.append(targets_with_batch)
        
        # 转换为jittor tensor
        if batch_imgs:
            batch_imgs = jt.array(np.stack(batch_imgs))
        else:
            batch_imgs = jt.zeros((self.batch_size, 3, self.img_size, self.img_size))
        
        # 合并所有targets
        if batch_targets:
            batch_targets = jt.array(np.concatenate(batch_targets, axis=0))
        else:
            batch_targets = jt.zeros((0, 6))
        
        self.current_idx += 1
        return batch_imgs, batch_targets

class AsyncDataLoader:
    """异步数据加载器 - 进一步提升性能"""
    
    def __init__(self, dataset, prefetch_factor=4):
        self.dataset = dataset
        self.prefetch_factor = prefetch_factor
        self.queue = queue.Queue(maxsize=prefetch_factor)
        self.stop_event = threading.Event()
        
        # 启动后台加载线程
        self.loader_thread = threading.Thread(target=self._background_loader)
        self.loader_thread.start()
    
    def _background_loader(self):
        """后台加载数据"""
        dataset_iter = iter(self.dataset)
        
        while not self.stop_event.is_set():
            try:
                batch = next(dataset_iter)
                self.queue.put(batch, timeout=1)
            except StopIteration:
                # 重新开始迭代
                dataset_iter = iter(self.dataset)
            except queue.Full:
                continue
            except Exception as e:
                print(f"⚠️ 后台加载错误: {e}")
                break
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return self.queue.get(timeout=5)
        except queue.Empty:
            raise StopIteration
    
    def __len__(self):
        return len(self.dataset)
    
    def stop(self):
        """停止后台加载"""
        self.stop_event.set()
        self.loader_thread.join()

def create_fast_dataloader(data_config, batch_size, is_train=True, num_workers=8):
    """创建高性能数据加载器"""
    
    # 数据路径
    data_path = data_config['path']
    split = 'train' if is_train else 'val'
    data_dir = os.path.join(data_path, split)
    
    if not os.path.exists(data_dir):
        print(f"⚠️ 数据目录不存在: {data_dir}")
        return None
    
    # 创建高性能数据集
    dataset = FastVOCDataset(
        data_dir=data_dir,
        img_size=640,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4
    )
    
    # 创建异步数据加载器
    async_loader = AsyncDataLoader(dataset, prefetch_factor=4)
    
    return async_loader

def benchmark_dataloader():
    """数据加载器性能测试"""
    print("🚀 数据加载器性能测试")
    
    data_config = {
        'path': '/home/kyc/project/GOLD-YOLO/data/voc2012_subset',
        'nc': 20
    }
    
    # 测试原始数据加载器
    print("\n📊 测试原始数据加载器...")
    from real_data_loader import VOCDataset

    start_time = time.time()
    old_dataset = VOCDataset('/home/kyc/project/GOLD-YOLO/data/voc2012_subset/train', batch_size=8)
    old_loader = iter(old_dataset)
    
    for i, (images, targets) in enumerate(old_loader):
        if i >= 10:  # 只测试10个batch
            break
    
    old_time = time.time() - start_time
    print(f"   原始加载器: {old_time:.2f}秒 (10 batches)")
    
    # 测试高性能数据加载器
    print("\n📊 测试高性能数据加载器...")
    
    start_time = time.time()
    fast_loader = create_fast_dataloader(data_config, 8, is_train=True, num_workers=8)
    
    for i, (images, targets) in enumerate(fast_loader):
        if i >= 10:  # 只测试10个batch
            break
    
    fast_time = time.time() - start_time
    print(f"   高性能加载器: {fast_time:.2f}秒 (10 batches)")
    
    speedup = old_time / fast_time
    print(f"\n🚀 性能提升: {speedup:.2f}x")
    
    fast_loader.stop()

if __name__ == "__main__":
    benchmark_dataloader()
