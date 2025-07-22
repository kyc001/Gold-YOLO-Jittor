#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
使用完整YOLO解码器的推理测试
新芽第二阶段：满血推理实现，不简化不妥协
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

import jittor as jt
import jittor.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置Jittor
jt.flags.use_cuda = 1

# 导入完整模型和解码器
from full_pytorch_small_model import FullPyTorchGoldYOLOSmall
from full_yolo_decoder import FullYOLODecoder

class FullGoldYOLOInference:
    """使用完整解码器的Gold-YOLO推理器"""
    
    def __init__(self, model_path, conf_threshold=0.3, nms_threshold=0.5):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # 加载模型
        self.model = self._load_model()
        self.model.eval()
        
        # 创建完整解码器
        self.decoder = FullYOLODecoder(
            input_size=640,
            num_classes=80,
            strides=[8, 16, 32]
        )
        
        print(f"🎯 完整Gold-YOLO推理器初始化完成")
        print(f"   模型: {self.model_path}")
        print(f"   置信度阈值: {self.conf_threshold}")
        print(f"   NMS阈值: {self.nms_threshold}")
        print(f"   解码器anchor数: {sum(len(grid) for grid in self.decoder.anchor_grids)}")
    
    def _load_model(self):
        """加载训练好的模型"""
        model = FullPyTorchGoldYOLOSmall(num_classes=80)
        
        if self.model_path.exists():
            try:
                checkpoint = jt.load(str(self.model_path))
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                    print(f"✅ 成功加载模型权重: {self.model_path}")
                    if 'training_info' in checkpoint:
                        print(f"   训练轮次: {checkpoint['training_info'].get('epoch', 'unknown')}")
                        print(f"   最佳损失: {checkpoint['training_info'].get('best_loss', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"✅ 成功加载模型权重: {self.model_path}")
            except Exception as e:
                print(f"⚠️ 加载权重失败: {e}")
                print(f"使用随机初始化权重进行推理测试")
        else:
            print(f"⚠️ 模型文件不存在: {self.model_path}")
            print(f"使用随机初始化权重进行推理测试")
        
        return model
    
    def preprocess_image(self, image_path, target_size=640):
        """预处理图片"""
        # 读取图片
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 等比例缩放到目标尺寸
        ratio = min(target_size / original_size[0], target_size / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image_resized = image.resize(new_size, Image.LANCZOS)
        
        # 创建目标尺寸的画布并居中放置
        canvas = Image.new('RGB', (target_size, target_size), (114, 114, 114))
        paste_x = (target_size - new_size[0]) // 2
        paste_y = (target_size - new_size[1]) // 2
        canvas.paste(image_resized, (paste_x, paste_y))
        
        # 转换为tensor
        image_array = np.array(canvas).astype(np.float32) / 255.0
        image_tensor = jt.array(image_array.transpose(2, 0, 1)).unsqueeze(0)
        
        return image_tensor, image, ratio, (paste_x, paste_y)
    
    def postprocess_detections(self, detections, ratio, offset, original_size):
        """后处理检测结果，转换回原图坐标"""
        paste_x, paste_y = offset
        
        processed_detections = []
        for detection in detections:
            # 获取640x640图像中的坐标
            x1, y1, x2, y2 = detection['bbox']
            
            # 转换回原图坐标
            # 1. 减去padding偏移
            x1 -= paste_x
            y1 -= paste_y
            x2 -= paste_x
            y2 -= paste_y
            
            # 2. 缩放回原图尺寸
            x1 /= ratio
            y1 /= ratio
            x2 /= ratio
            y2 /= ratio
            
            # 3. 限制在原图范围内
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
            # 过滤无效框
            if x2 > x1 and y2 > y1:
                processed_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': detection['confidence'],
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name']
                })
        
        return processed_detections
    
    def visualize_detections(self, image, detections, output_path):
        """可视化检测结果"""
        # 创建matplotlib图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # 左侧：原图
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=16)
        ax1.axis('off')
        
        # 右侧：检测结果
        ax2.imshow(image)
        ax2.set_title(f'Full YOLO Decoder Results ({len(detections)} objects)', fontsize=16)
        ax2.axis('off')
        
        # 颜色列表
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(detections), 1)))
        
        # 绘制检测框
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 计算框的坐标和尺寸
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # 绘制边界框
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=3, edgecolor=colors[i], 
                                   facecolor='none', alpha=0.8)
            ax2.add_patch(rect)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.3f}"
            ax2.text(x1, y1-5, label, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8),
                    color='black', weight='bold')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 完整解码可视化结果已保存: {output_path}")
    
    def inference_single_image(self, image_path, output_dir=None):
        """对单张图片进行完整推理"""
        image_path = Path(image_path)
        
        if output_dir is None:
            output_dir = Path("runs/full_inference_results")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔍 完整推理: {image_path}")
        
        # 预处理
        start_time = time.time()
        image_tensor, original_image, ratio, offset = self.preprocess_image(image_path)
        preprocess_time = time.time() - start_time
        
        # 模型推理
        start_time = time.time()
        with jt.no_grad():
            features, cls_pred, reg_pred = self.model(image_tensor)
        model_inference_time = time.time() - start_time
        
        # 完整解码
        start_time = time.time()
        batch_detections = self.decoder.decode_predictions(
            cls_pred, reg_pred,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            max_detections=100
        )
        decode_time = time.time() - start_time
        
        # 后处理（坐标转换）
        start_time = time.time()
        detections = self.postprocess_detections(
            batch_detections[0], ratio, offset, original_image.size
        )
        postprocess_time = time.time() - start_time
        
        # 可视化
        start_time = time.time()
        vis_output_path = output_dir / f"full_detection_{image_path.stem}.png"
        self.visualize_detections(original_image, detections, vis_output_path)
        visualization_time = time.time() - start_time
        
        # 打印结果
        total_time = preprocess_time + model_inference_time + decode_time + postprocess_time + visualization_time
        print(f"⏱️ 完整推理时间统计:")
        print(f"   预处理: {preprocess_time*1000:.2f} ms")
        print(f"   模型推理: {model_inference_time*1000:.2f} ms")
        print(f"   完整解码: {decode_time*1000:.2f} ms")
        print(f"   后处理: {postprocess_time*1000:.2f} ms")
        print(f"   可视化: {visualization_time*1000:.2f} ms")
        print(f"   总时间: {total_time*1000:.2f} ms")
        print(f"   FPS: {1/total_time:.1f}")
        
        print(f"🎯 完整解码检测结果: {len(detections)}个目标")
        for i, det in enumerate(detections):
            print(f"   {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        
        # 保存详细结果
        results = {
            'image_path': str(image_path),
            'detections': detections,
            'timing': {
                'preprocess_ms': preprocess_time * 1000,
                'model_inference_ms': model_inference_time * 1000,
                'decode_ms': decode_time * 1000,
                'postprocess_ms': postprocess_time * 1000,
                'visualization_ms': visualization_time * 1000,
                'total_ms': total_time * 1000,
                'fps': 1 / total_time
            },
            'model_info': {
                'input_shape': list(image_tensor.shape),
                'cls_output_shape': list(cls_pred.shape),
                'reg_output_shape': list(reg_pred.shape),
                'num_features': len(features),
                'decoder_anchors': sum(len(grid) for grid in self.decoder.anchor_grids)
            }
        }
        
        results_file = output_dir / f"full_results_{image_path.stem}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return detections, results
    
    def batch_full_inference(self, image_dir, max_images=3):
        """批量完整推理"""
        image_dir = Path(image_dir)
        output_dir = Path("runs/batch_full_inference")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        image_files = image_files[:max_images]
        
        print(f"\n🎯 批量完整推理")
        print(f"   图片目录: {image_dir}")
        print(f"   找到图片: {len(image_files)}张")
        print(f"   输出目录: {output_dir}")
        
        if not image_files:
            print("❌ 未找到图片文件")
            return
        
        all_results = []
        total_detections = 0
        
        for i, image_file in enumerate(image_files):
            print(f"\n--- 完整推理 {i+1}/{len(image_files)} ---")
            detections, results = self.inference_single_image(image_file, output_dir)
            all_results.append(results)
            total_detections += len(detections)
        
        # 创建汇总报告
        summary = {
            'total_images': len(image_files),
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(image_files),
            'avg_model_inference_time_ms': np.mean([r['timing']['model_inference_ms'] for r in all_results]),
            'avg_decode_time_ms': np.mean([r['timing']['decode_ms'] for r in all_results]),
            'avg_total_time_ms': np.mean([r['timing']['total_ms'] for r in all_results]),
            'avg_fps': np.mean([r['timing']['fps'] for r in all_results]),
            'results': all_results
        }
        
        summary_file = output_dir / "full_batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 批量完整推理汇总:")
        print(f"   处理图片: {len(image_files)}张")
        print(f"   总检测数: {total_detections}个")
        print(f"   平均每张: {total_detections/len(image_files):.1f}个")
        print(f"   平均模型推理: {summary['avg_model_inference_time_ms']:.2f} ms")
        print(f"   平均完整解码: {summary['avg_decode_time_ms']:.2f} ms")
        print(f"   平均总时间: {summary['avg_total_time_ms']:.2f} ms")
        print(f"   平均FPS: {summary['avg_fps']:.1f}")
        print(f"✅ 完整推理汇总报告已保存: {summary_file}")


def main():
    """主函数"""
    print("🎯 Gold-YOLO 完整解码推理测试")
    print("新芽第二阶段：满血YOLO解码，不简化不妥协")
    print("=" * 60)
    
    # 模型路径 (使用修复后的模型)
    model_path = "runs/validated_full_pytorch_small/fixed_best_model.pkl"
    
    # 创建完整推理器
    inferencer = FullGoldYOLOInference(
        model_path=model_path, 
        conf_threshold=0.3, 
        nms_threshold=0.5
    )
    
    # 使用测试集中的图片进行完整推理
    test_image_dir = "/home/kyc/project/GOLD-YOLO/data/coco2017_val/images"
    if Path(test_image_dir).exists():
        print(f"使用测试集图片进行完整推理: {test_image_dir}")
        inferencer.batch_full_inference(test_image_dir, max_images=3)
        
        print(f"\n🎉 完整解码推理测试完成！")
        print(f"📁 结果保存在: runs/batch_full_inference/")
        print(f"📊 包含:")
        print(f"   - 使用完整YOLO解码器的检测结果")
        print(f"   - 真实的anchor生成和NMS处理")
        print(f"   - 完整的边界框解码算法")
        print(f"   - 详细的性能分析报告")
        
    else:
        print("❌ 测试集图片目录不存在")
        print("请确保数据集已正确下载和配置")


if __name__ == "__main__":
    main()
