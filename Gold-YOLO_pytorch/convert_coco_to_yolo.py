#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
COCO to YOLO格式转换器
新芽第二阶段：为PyTorch训练准备数据
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir):
    """
    将COCO格式的标注转换为YOLO格式
    
    Args:
        coco_json_path: COCO JSON标注文件路径
        images_dir: 图片目录路径
        output_labels_dir: 输出标签目录路径
    """
    
    # 创建输出目录
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取COCO标注
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 创建图片ID到文件名的映射
    image_id_to_filename = {}
    image_id_to_size = {}
    
    for img in coco_data['images']:
        image_id_to_filename[img['id']] = img['file_name']
        image_id_to_size[img['id']] = (img['width'], img['height'])
    
    # 创建类别ID到索引的映射
    category_id_to_index = {}
    for i, cat in enumerate(coco_data['categories']):
        category_id_to_index[cat['id']] = i
    
    # 按图片分组标注
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    print(f"🔄 转换COCO标注到YOLO格式")
    print(f"   输入: {coco_json_path}")
    print(f"   输出: {output_labels_dir}")
    print(f"   图片数量: {len(image_id_to_filename)}")
    print(f"   标注数量: {len(coco_data['annotations'])}")
    
    # 转换每个图片的标注
    converted_count = 0
    for image_id, filename in tqdm(image_id_to_filename.items(), desc="转换标注"):
        # 检查图片是否存在
        image_path = Path(images_dir) / filename
        if not image_path.exists():
            continue
        
        # 获取图片尺寸
        img_width, img_height = image_id_to_size[image_id]
        
        # 创建对应的标签文件
        label_filename = Path(filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        # 转换该图片的所有标注
        yolo_lines = []
        if image_id in image_annotations:
            for ann in image_annotations[image_id]:
                # 获取边界框 [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # 转换为YOLO格式 (中心点坐标 + 归一化)
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # 获取类别索引
                category_id = ann['category_id']
                class_index = category_id_to_index[category_id]
                
                # 创建YOLO格式的行
                yolo_line = f"{class_index} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                yolo_lines.append(yolo_line)
        
        # 写入标签文件
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted_count += 1
    
    print(f"✅ 转换完成")
    print(f"   转换图片: {converted_count}")
    print(f"   标签目录: {output_labels_dir}")
    
    return converted_count

def main():
    """主函数"""
    print("🔄 COCO to YOLO格式转换器")
    print("新芽第二阶段：为PyTorch训练准备数据")
    print("=" * 60)
    
    # 数据路径
    data_root = Path("/home/kyc/project/GOLD-YOLO/data/coco2017_val")
    images_dir = data_root / "images"
    splits_dir = data_root / "splits"
    
    # 输入文件
    train_json = splits_dir / "instances_train.json"
    val_json = splits_dir / "instances_val.json"
    
    # 输出目录
    labels_dir = data_root / "labels"
    train_labels_dir = labels_dir / "train"
    val_labels_dir = labels_dir / "val"
    
    if not train_json.exists():
        print(f"❌ 找不到训练标注文件: {train_json}")
        return False
    
    if not val_json.exists():
        print(f"❌ 找不到验证标注文件: {val_json}")
        return False
    
    try:
        # 转换训练集
        print(f"\n📋 转换训练集标注")
        train_count = convert_coco_to_yolo(train_json, images_dir, train_labels_dir)
        
        # 转换验证集
        print(f"\n📋 转换验证集标注")
        val_count = convert_coco_to_yolo(val_json, images_dir, val_labels_dir)
        
        print(f"\n🎉 所有转换完成！")
        print(f"   训练集: {train_count} 个标签文件")
        print(f"   验证集: {val_count} 个标签文件")
        print(f"   标签目录: {labels_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ YOLO格式标签准备完成！")
        print(f"💡 现在可以开始PyTorch训练")
    else:
        print(f"\n❌ 标签转换失败！")
        print(f"💡 请检查数据路径和格式")
