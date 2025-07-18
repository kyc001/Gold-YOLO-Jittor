#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本 - 为对齐实验准备少量数据集
支持COCO和VOC格式，确保PyTorch和Jittor使用完全相同的数据
"""

import os
import json
import shutil
import random
from pathlib import Path
import argparse
from typing import List, Dict, Any


class DatasetPreparer:
    """数据集准备器"""
    
    def __init__(self, source_path: str, target_path: str, seed: int = 42):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.seed = seed
        random.seed(seed)
        
    def prepare_coco_subset(self, num_images: int = 1000, selected_classes: List[str] = None):
        """准备COCO子集用于对齐实验"""
        if selected_classes is None:
            selected_classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                'cat', 'dog', 'bottle', 'chair'
            ]
        
        print(f"🐾 准备COCO子集: {num_images}张图片, {len(selected_classes)}个类别")
        
        # 创建目标目录
        target_images = self.target_path / "images"
        target_labels = self.target_path / "labels"
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        
        # 读取COCO annotations
        ann_file = self.source_path / "annotations" / "instances_train2017.json"
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 创建类别映射
        class_mapping = {}
        new_categories = []
        for i, class_name in enumerate(selected_classes):
            for cat in coco_data['categories']:
                if cat['name'] == class_name:
                    class_mapping[cat['id']] = i
                    new_categories.append({
                        'id': i,
                        'name': class_name,
                        'supercategory': cat['supercategory']
                    })
                    break
        
        # 筛选图片和标注
        valid_images = []
        valid_annotations = []
        
        for img in coco_data['images']:
            img_annotations = [ann for ann in coco_data['annotations'] 
                             if ann['image_id'] == img['id'] and ann['category_id'] in class_mapping]
            if img_annotations:
                valid_images.append(img)
                valid_annotations.extend(img_annotations)
        
        # 随机选择指定数量的图片
        selected_images = random.sample(valid_images, min(num_images, len(valid_images)))
        selected_image_ids = {img['id'] for img in selected_images}
        
        # 复制图片和生成YOLO格式标签
        copied_count = 0
        for img in selected_images:
            # 复制图片
            src_img = self.source_path / "images" / "train2017" / img['file_name']
            dst_img = target_images / img['file_name']
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                
                # 生成YOLO格式标签
                img_annotations = [ann for ann in valid_annotations 
                                 if ann['image_id'] == img['id']]
                
                label_file = target_labels / (img['file_name'].replace('.jpg', '.txt'))
                with open(label_file, 'w') as f:
                    for ann in img_annotations:
                        if ann['category_id'] in class_mapping:
                            # 转换为YOLO格式
                            x, y, w, h = ann['bbox']
                            img_w, img_h = img['width'], img['height']
                            
                            # 归一化
                            x_center = (x + w/2) / img_w
                            y_center = (y + h/2) / img_h
                            norm_w = w / img_w
                            norm_h = h / img_h
                            
                            class_id = class_mapping[ann['category_id']]
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
                copied_count += 1
        
        # 保存数据集信息
        dataset_info = {
            'name': 'coco_subset_alignment',
            'num_images': copied_count,
            'num_classes': len(selected_classes),
            'classes': selected_classes,
            'class_mapping': class_mapping,
            'seed': self.seed,
            'source': str(self.source_path)
        }
        
        with open(self.target_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # 生成数据集配置文件
        self._generate_dataset_yaml(selected_classes)
        
        print(f"✅ 数据准备完成: {copied_count}张图片")
        return dataset_info
    
    def _generate_dataset_yaml(self, classes: List[str]):
        """生成数据集YAML配置文件"""
        yaml_content = f"""# Gold-YOLO对齐实验数据集配置
path: {self.target_path.absolute()}
train: images
val: images  # 使用相同数据进行验证（对齐实验）

# 类别数量
nc: {len(classes)}

# 类别名称
names: {classes}
"""
        
        with open(self.target_path / "dataset.yaml", 'w') as f:
            f.write(yaml_content)
    
    def split_train_val(self, val_ratio: float = 0.2):
        """划分训练集和验证集"""
        images_dir = self.target_path / "images"
        labels_dir = self.target_path / "labels"
        
        # 创建train/val目录
        for split in ['train', 'val']:
            (self.target_path / "images" / split).mkdir(exist_ok=True)
            (self.target_path / "labels" / split).mkdir(exist_ok=True)
        
        # 获取所有图片
        image_files = list(images_dir.glob("*.jpg"))
        random.shuffle(image_files)
        
        # 划分数据
        val_count = int(len(image_files) * val_ratio)
        val_files = image_files[:val_count]
        train_files = image_files[val_count:]
        
        # 移动文件
        for img_file in train_files:
            label_file = labels_dir / (img_file.stem + '.txt')
            shutil.move(img_file, self.target_path / "images" / "train" / img_file.name)
            if label_file.exists():
                shutil.move(label_file, self.target_path / "labels" / "train" / label_file.name)
        
        for img_file in val_files:
            label_file = labels_dir / (img_file.stem + '.txt')
            shutil.move(img_file, self.target_path / "images" / "val" / img_file.name)
            if label_file.exists():
                shutil.move(label_file, self.target_path / "labels" / "val" / label_file.name)
        
        # 删除原始目录
        if images_dir.exists() and not any(images_dir.iterdir()):
            images_dir.rmdir()
        if labels_dir.exists() and not any(labels_dir.iterdir()):
            labels_dir.rmdir()
        
        print(f"✅ 数据划分完成: 训练集{len(train_files)}张, 验证集{len(val_files)}张")


def main():
    parser = argparse.ArgumentParser(description='准备对齐实验数据集')
    parser.add_argument('--source', type=str, required=True, help='源数据集路径')
    parser.add_argument('--target', type=str, default='./data/alignment_dataset', help='目标数据集路径')
    parser.add_argument('--num_images', type=int, default=1000, help='图片数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--split', action='store_true', help='是否划分训练验证集')
    
    args = parser.parse_args()
    
    # 选择的类别（确保PyTorch和Jittor使用相同的类别）
    selected_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        'cat', 'dog', 'bottle', 'chair'
    ]
    
    preparer = DatasetPreparer(args.source, args.target, args.seed)
    dataset_info = preparer.prepare_coco_subset(args.num_images, selected_classes)
    
    if args.split:
        preparer.split_train_val()
    
    print(f"\n🎉 数据准备完成!")
    print(f"📁 数据路径: {args.target}")
    print(f"📊 数据信息: {dataset_info}")


if __name__ == "__main__":
    main()
