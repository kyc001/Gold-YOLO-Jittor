#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
创建规范的数据集划分
新芽第二阶段：训练集/测试集分离，统一评估
"""

import os
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, data_root, train_ratio=0.8, seed=42):
        self.data_root = Path(data_root)
        self.train_ratio = train_ratio
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        
        # 路径配置
        self.img_dir = self.data_root / "images"
        self.ann_file = self.data_root / "annotations" / "instances_val2017.json"
        
        # 输出路径
        self.output_dir = self.data_root / "splits"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 数据集划分器")
        print(f"   数据根目录: {self.data_root}")
        print(f"   训练比例: {self.train_ratio}")
        print(f"   随机种子: {self.seed}")
    
    def load_and_validate_data(self, max_images=1000):
        """加载并验证数据"""
        print(f"\n📊 加载COCO数据...")
        
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        print(f"✅ 原始数据: {len(coco_data['images'])}张图片, {len(coco_data['annotations'])}个标注")
        
        # 按图片ID组织标注
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # 筛选有效图片
        valid_images = []
        print("验证图片完整性...")
        
        for img_info in tqdm(coco_data['images'][:max_images*2]):  # 多取一些以防不够
            img_path = self.img_dir / img_info['file_name']
            img_id = img_info['id']
            
            # 检查文件存在且有标注
            if img_path.exists() and img_id in image_annotations:
                try:
                    # 验证图片可读
                    img = Image.open(img_path)
                    img.verify()
                    
                    valid_images.append({
                        'info': img_info,
                        'annotations': image_annotations[img_id],
                        'annotation_count': len(image_annotations[img_id])
                    })
                    
                    # 达到目标数量就停止
                    if len(valid_images) >= max_images:
                        break
                        
                except:
                    continue
        
        print(f"✅ 有效图片: {len(valid_images)}")
        
        # 按标注数量排序，优先选择标注丰富的图片
        valid_images.sort(key=lambda x: x['annotation_count'], reverse=True)
        
        return valid_images[:max_images], coco_data
    
    def split_dataset(self, valid_images, coco_data):
        """划分训练集和测试集"""
        print(f"\n🔄 划分数据集...")
        
        # 随机打乱
        random.shuffle(valid_images)
        
        # 计算划分点
        split_point = int(len(valid_images) * self.train_ratio)
        
        train_images = valid_images[:split_point]
        test_images = valid_images[split_point:]
        
        print(f"✅ 训练集: {len(train_images)}张图片")
        print(f"✅ 测试集: {len(test_images)}张图片")
        
        # 创建训练集和测试集数据
        train_data = self._create_subset_data(train_images, coco_data, "train")
        test_data = self._create_subset_data(test_images, coco_data, "test")
        
        return train_data, test_data
    
    def _create_subset_data(self, images, coco_data, split_name):
        """创建子集数据"""
        # 收集图片信息
        subset_images = [img['info'] for img in images]
        selected_img_ids = {img['info']['id'] for img in images}
        
        # 收集对应的标注
        subset_annotations = []
        for ann in coco_data['annotations']:
            if ann['image_id'] in selected_img_ids:
                subset_annotations.append(ann)
        
        # 创建子集数据结构
        subset_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'categories': coco_data['categories'],
            'images': subset_images,
            'annotations': subset_annotations
        }
        
        # 统计信息
        total_annotations = len(subset_annotations)
        avg_annotations = total_annotations / len(subset_images) if subset_images else 0
        
        # 统计类别分布
        category_counts = {}
        for ann in subset_annotations:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        print(f"   {split_name}集统计:")
        print(f"     图片数: {len(subset_images)}")
        print(f"     标注数: {total_annotations}")
        print(f"     平均标注/图片: {avg_annotations:.1f}")
        print(f"     使用类别数: {len(category_counts)}")
        
        return subset_data
    
    def save_splits(self, train_data, test_data):
        """保存数据集划分"""
        print(f"\n💾 保存数据集划分...")
        
        # 保存训练集
        train_file = self.output_dir / "train_annotations.json"
        with open(train_file, 'w') as f:
            json.dump(train_data, f)
        print(f"✅ 训练集已保存: {train_file}")
        
        # 保存测试集
        test_file = self.output_dir / "test_annotations.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        print(f"✅ 测试集已保存: {test_file}")
        
        # 保存数据集信息
        dataset_info = {
            'dataset_name': 'COCO2017_Val_Split',
            'total_images': len(train_data['images']) + len(test_data['images']),
            'train_images': len(train_data['images']),
            'test_images': len(test_data['images']),
            'train_annotations': len(train_data['annotations']),
            'test_annotations': len(test_data['annotations']),
            'train_ratio': self.train_ratio,
            'test_ratio': 1 - self.train_ratio,
            'random_seed': self.seed,
            'categories': len(train_data['categories']),
            'split_date': '2024-12-19'
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        print(f"✅ 数据集信息已保存: {info_file}")
        
        # 创建图片列表文件
        train_list_file = self.output_dir / "train_images.txt"
        with open(train_list_file, 'w') as f:
            for img in train_data['images']:
                f.write(f"{img['file_name']}\n")
        
        test_list_file = self.output_dir / "test_images.txt"
        with open(test_list_file, 'w') as f:
            for img in test_data['images']:
                f.write(f"{img['file_name']}\n")
        
        print(f"✅ 图片列表已保存: {train_list_file}, {test_list_file}")
        
        return {
            'train_file': train_file,
            'test_file': test_file,
            'info_file': info_file,
            'train_list': train_list_file,
            'test_list': test_list_file
        }
    
    def create_splits(self, max_images=1000):
        """创建完整的数据集划分"""
        print("🎯 开始创建数据集划分...")
        print("=" * 60)
        
        # 1. 加载和验证数据
        valid_images, coco_data = self.load_and_validate_data(max_images)
        
        if len(valid_images) < 100:
            print(f"❌ 有效图片太少: {len(valid_images)}")
            return None
        
        # 2. 划分数据集
        train_data, test_data = self.split_dataset(valid_images, coco_data)
        
        # 3. 保存数据集
        files = self.save_splits(train_data, test_data)
        
        print("=" * 60)
        print("✅ 数据集划分完成！")
        print(f"📁 输出目录: {self.output_dir}")
        
        return {
            'files': files,
            'train_data': train_data,
            'test_data': test_data,
            'stats': {
                'total_images': len(valid_images),
                'train_images': len(train_data['images']),
                'test_images': len(test_data['images']),
                'train_annotations': len(train_data['annotations']),
                'test_annotations': len(test_data['annotations'])
            }
        }


def main():
    """主函数"""
    # 数据集配置
    data_root = "/home/kyc/project/GOLD-YOLO/data/coco2017_val"
    max_images = 1000  # 总图片数
    train_ratio = 0.8  # 训练集比例
    
    print("🎯 Gold-YOLO 数据集划分")
    print("新芽第二阶段：规范的训练集/测试集分离")
    print("=" * 60)
    print(f"📊 配置:")
    print(f"   数据根目录: {data_root}")
    print(f"   总图片数: {max_images}")
    print(f"   训练集比例: {train_ratio}")
    print(f"   测试集比例: {1-train_ratio}")
    
    # 创建数据集划分器
    splitter = DatasetSplitter(
        data_root=data_root,
        train_ratio=train_ratio,
        seed=42
    )
    
    # 创建数据集划分
    result = splitter.create_splits(max_images)
    
    if result:
        # 显示最终统计
        stats = result['stats']
        
        print("\n📈 最终统计:")
        print(f"   总图片: {stats['total_images']}")
        print(f"   训练集: {stats['train_images']}张图片, {stats['train_annotations']}个标注")
        print(f"   测试集: {stats['test_images']}张图片, {stats['test_annotations']}个标注")
        
        print("\n🚀 数据集准备完成，可以开始训练和评估！")
        print("\n📋 使用方法:")
        print("   训练: python fixed_jittor_train.py --train-file splits/train_annotations.json")
        print("   测试: python evaluate_model.py --test-file splits/test_annotations.json")
    else:
        print("❌ 数据集划分失败")


if __name__ == "__main__":
    main()
