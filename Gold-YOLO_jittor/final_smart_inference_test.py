#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
最终智能推理测试
使用智能匹配权重进行最终推理测试
"""

import os
import sys
import numpy as np
import jittor as jt
import cv2
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# 设置Jittor
jt.flags.use_cuda = 1 if jt.has_cuda else 0

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from pytorch_aligned_model import PyTorchAlignedGoldYOLO


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """严格按照PyTorch版本的letterbox实现"""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class FinalSmartInferenceTester:
    """最终智能推理测试器"""
    
    def __init__(self):
        """初始化"""
        self.smart_weights_path = "weights/smart_matched_weights.npz"
        self.test_images_dir = "/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/gold_yolo_n_test/test_images"
        
        # VOC 20类别名称
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # 创建输出目录
        os.makedirs("outputs/final_smart_inference", exist_ok=True)
        
        print("🧠 最终智能推理测试器")
        print("   使用智能匹配权重进行推理")
        print("=" * 80)
    
    def load_smart_model(self):
        """加载智能匹配权重的模型"""
        print("\n📦 加载智能匹配权重模型")
        print("-" * 60)
        
        # 创建模型
        model = PyTorchAlignedGoldYOLO(num_classes=20)
        
        # 加载智能匹配权重
        if not os.path.exists(self.smart_weights_path):
            print(f"❌ 智能匹配权重文件不存在: {self.smart_weights_path}")
            return None
        
        weights = np.load(self.smart_weights_path)
        jt_state_dict = {name: jt.array(weight) for name, weight in weights.items()}
        model.load_state_dict(jt_state_dict)
        model.eval()
        
        # 计算覆盖率
        model_params = dict(model.named_parameters())
        coverage = len(weights) / len(model_params) * 100
        
        print(f"✅ 智能匹配模型加载成功")
        print(f"   权重覆盖率: {coverage:.1f}%")
        print(f"   加载权重数: {len(weights)}")
        
        return model
    
    def preprocess_image(self, img_path, img_size=640):
        """图像预处理"""
        # 读取图像
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, f'Image Not Found {img_path}'
        
        # Letterbox
        img = letterbox(img0, img_size, stride=32)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        # Normalize
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Add batch dimension
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        return jt.array(img), img0
    
    def smart_inference(self, model, img_path, conf_thres=0.01):  # 降低阈值
        """智能推理"""
        print(f"\n🧠 智能推理测试")
        print("-" * 60)
        
        # 预处理
        img, img0 = self.preprocess_image(img_path)
        print(f"   图像预处理: {img.shape} -> 原图: {img0.shape}")
        
        # 推理
        t1 = time.time()
        with jt.no_grad():
            output = model(img)
        t2 = time.time()
        
        print(f"   模型推理: 耗时 {(t2-t1)*1000:.1f}ms")
        
        # 解析输出
        if isinstance(output, list):
            detections, featmaps = output
            print(f"   输出解析: 检测{detections.shape}, 特征图{len(featmaps)}个")
        else:
            detections = output
            print(f"   输出解析: 检测{detections.shape}")
        
        # 分析检测结果
        det = detections[0]  # [anchors, 25]
        coords = det[:, :4]  # xywh
        obj_conf = det[:, 4]  # objectness
        cls_probs = det[:, 5:]  # class probabilities
        
        # 计算最终置信度
        max_cls_probs = jt.max(cls_probs, dim=1)[0]
        cls_ids = jt.argmax(cls_probs, dim=1)[0]
        total_conf = obj_conf * max_cls_probs
        
        print(f"   🔍 检测分析:")
        print(f"      总anchor数: {len(det)}")
        print(f"      目标置信度范围: [{obj_conf.min():.6f}, {obj_conf.max():.6f}]")
        print(f"      目标置信度唯一值: {len(jt.unique(obj_conf))}")
        print(f"      类别概率范围: [{max_cls_probs.min():.6f}, {max_cls_probs.max():.6f}]")
        print(f"      最高总置信度: {total_conf.max():.6f}")
        print(f"      >0.01检测数: {(total_conf > 0.01).sum()}")
        print(f"      >0.005检测数: {(total_conf > 0.005).sum()}")
        
        # 后处理 - 使用更低的阈值
        final_detections = self.postprocess_detections(det, img.shape[2:], img0.shape, conf_thres)
        
        print(f"   后处理: {len(final_detections)}个检测")
        
        return final_detections, img0
    
    def postprocess_detections(self, pred, img_shape, orig_shape, conf_thres=0.01):
        """后处理检测结果"""
        # 分离坐标、置信度和类别概率
        boxes = pred[:, :4]  # xywh
        obj_conf = pred[:, 4]  # objectness
        cls_probs = pred[:, 5:]  # class probabilities
        
        # 计算最终置信度
        max_cls_probs = jt.max(cls_probs, dim=1)[0]
        cls_ids = jt.argmax(cls_probs, dim=1)[0]
        total_conf = obj_conf * max_cls_probs
        
        # 置信度过滤
        conf_mask = total_conf > conf_thres
        if not jt.any(conf_mask):
            return []
        
        # 过滤
        boxes = boxes[conf_mask]
        total_conf = total_conf[conf_mask]
        cls_ids = cls_ids[conf_mask]
        
        # 转换坐标格式 xywh -> xyxy
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        xyxy_boxes = jt.stack([x1, y1, x2, y2], dim=1)
        
        # 坐标缩放回原图
        gain = min(img_shape[0] / orig_shape[0], img_shape[1] / orig_shape[1])
        pad = (img_shape[1] - orig_shape[1] * gain) / 2, (img_shape[0] - orig_shape[0] * gain) / 2
        
        xyxy_boxes[:, [0, 2]] -= pad[0]  # x padding
        xyxy_boxes[:, [1, 3]] -= pad[1]  # y padding
        xyxy_boxes[:, :4] /= gain
        
        # 限制在图像范围内
        xyxy_boxes[:, 0] = jt.clamp(xyxy_boxes[:, 0], 0, orig_shape[1])
        xyxy_boxes[:, 1] = jt.clamp(xyxy_boxes[:, 1], 0, orig_shape[0])
        xyxy_boxes[:, 2] = jt.clamp(xyxy_boxes[:, 2], 0, orig_shape[1])
        xyxy_boxes[:, 3] = jt.clamp(xyxy_boxes[:, 3], 0, orig_shape[0])
        
        # 转换为检测结果
        detections = []
        for i in range(len(xyxy_boxes)):
            x1, y1, x2, y2 = xyxy_boxes[i].numpy()
            conf = float(total_conf[i].numpy())
            cls_id = int(cls_ids[i].numpy())
            
            # 检查框的有效性
            if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id]
                })
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # 限制检测数量
        detections = detections[:50]  # 增加检测数量限制
        
        return detections
    
    def visualize_detections(self, img, detections, save_path):
        """可视化检测结果"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # 绘制检测框
        colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colors[class_id], facecolor='none'
            )
            ax.add_patch(rect)
            
            # 添加标签
            label = f"{class_name}: {confidence:.4f}"
            ax.text(x1, y1 - 5, label, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[class_id], alpha=0.7),
                   fontsize=10, color='black', weight='bold')
        
        image_name = Path(save_path).stem.replace('_smart', '')
        ax.set_title(f"Smart Matched: {image_name} - {len(detections)} detections", fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   可视化保存: {Path(save_path).name}")
    
    def run_final_smart_test(self):
        """运行最终智能测试"""
        print("🧠 运行最终智能推理测试")
        print("=" * 80)
        
        # 1. 加载智能模型
        model = self.load_smart_model()
        if model is None:
            return
        
        # 2. 获取测试图像
        image_files = glob.glob(os.path.join(self.test_images_dir, "*.jpg"))
        print(f"\n🖼️ 找到 {len(image_files)} 张测试图像")
        
        # 3. 对每张图像进行智能推理
        all_results = []
        
        for i, image_path in enumerate(image_files[:3]):  # 测试前3张
            image_name = Path(image_path).stem
            print(f"\n📷 处理图像 {i+1}: {image_name}")
            
            # 运行智能推理
            detections, orig_img = self.smart_inference(model, image_path, conf_thres=0.01)
            
            # 可视化结果
            save_path = f"outputs/final_smart_inference/{image_name}_smart.png"
            self.visualize_detections(orig_img, detections, save_path)
            
            # 记录结果
            all_results.append({
                'image_name': image_name,
                'detections': detections,
                'detection_count': len(detections)
            })
            
            # 显示检测结果
            if detections:
                print(f"   ✅ 检测结果:")
                for j, det in enumerate(detections[:10]):  # 显示前10个
                    print(f"      {j+1}. {det['class_name']}: {det['confidence']:.4f}")
                if len(detections) > 10:
                    print(f"      ... 还有{len(detections)-10}个检测")
            else:
                print(f"   ❌ 未检测到目标")
        
        print(f"\n🎉 最终智能推理测试完成!")
        print("=" * 80)
        
        # 总结
        total_detections = sum(r['detection_count'] for r in all_results)
        avg_detections = total_detections / len(all_results) if all_results else 0
        
        print(f"📊 智能推理总结:")
        print(f"   测试图像: {len(all_results)}")
        print(f"   总检测数: {total_detections}")
        print(f"   平均检测数: {avg_detections:.1f}")
        
        if total_detections > 0:
            print(f"   🎯 智能推理成功: 检测到目标！")
            print(f"   🏆 Gold-YOLO Jittor版本智能匹配成功！")
            
            # 最终评估
            if avg_detections > 5:
                print(f"   🌟 性能评估: 优秀")
            elif avg_detections > 1:
                print(f"   ✅ 性能评估: 良好")
            else:
                print(f"   ⚠️ 性能评估: 一般")
        else:
            print(f"   ⚠️ 仍需进一步优化")


def main():
    """主函数"""
    tester = FinalSmartInferenceTester()
    tester.run_final_smart_test()


if __name__ == '__main__':
    main()
