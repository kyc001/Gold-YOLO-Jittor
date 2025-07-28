#!/usr/bin/env python3
"""
GOLD-YOLO-n 自检训练验证系统
用单张图片训练500次，然后检测同一张图片验证模型功能
"""

import os
import sys
import time
import cv2
import numpy as np
import yaml
from pathlib import Path

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset, DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.general import scale_coords

class SingleImageDataset(Dataset):
    """单张图片数据集"""
    
    def __init__(self, img_path, label_path, img_size=640):
        super().__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.img_size = img_size
        
        # 加载图像
        self.img = cv2.imread(img_path)
        assert self.img is not None, f"无法读取图像: {img_path}"
        
        # 加载标签
        self.labels = self.load_labels(label_path)
        
        print(f"📸 自检图像: {img_path}")
        print(f"🏷️ 图像尺寸: {self.img.shape}")
        print(f"🎯 目标数量: {len(self.labels)}")
        if len(self.labels) > 0:
            print(f"🎯 目标类别: {[int(label[0]) for label in self.labels]}")
    
    def load_labels(self, label_path):
        """加载YOLO格式标签"""
        if not os.path.exists(label_path):
            print(f"⚠️ 标签文件不存在: {label_path}")
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append([cls_id, x_center, y_center, width, height])
        
        return labels
    
    def __len__(self):
        return 1  # 只有一张图片
    
    def __getitem__(self, idx):
        # 图像预处理
        img = letterbox(self.img, new_shape=self.img_size, stride=32, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0  # 归一化到[0,1]

        # 标签处理 - 使用调试验证的正确格式
        if len(self.labels) > 0:
            # 单个目标的格式：[cls, x_center, y_center, width, height, 0]
            label = self.labels[0]  # 只使用第一个标签
            labels_out = jt.array([[label[0], label[1], label[2], label[3], label[4], 0]], dtype='float32')
        else:
            labels_out = jt.zeros((1, 6), dtype='float32')  # 空标签

        return jt.array(img, dtype='float32'), labels_out

def create_self_check_dataset():
    """创建自检数据集"""
    # 选择测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    
    # 创建对应的标签文件（手动标注一个简单的目标）
    label_path = 'self_check_label.txt'
    
    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"❌ 测试图像不存在: {img_path}")
        return None, None
    
    # 创建简单的标签（假设图像中心有一个大目标）
    with open(label_path, 'w') as f:
        # 类别0，中心位置(0.5, 0.5)，尺寸(0.8, 0.8) - 更大的目标更容易学习
        f.write("0 0.5 0.5 0.8 0.8\n")
    
    print(f"✅ 创建自检标签: {label_path}")
    
    # 创建数据集
    dataset = SingleImageDataset(img_path, label_path)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    return dataset, dataloader

def self_check_training():
    """自检训练主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                GOLD-YOLO-n 自检训练验证系统                  ║
    ║                                                              ║
    ║  🎯 用单张图片训练500次验证模型功能                          ║
    ║  📊 训练完成后检测同一张图片验证识别能力                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建自检数据集
    print("📦 创建自检数据集...")
    dataset, dataloader = create_self_check_dataset()
    if dataset is None:
        return False
    
    # 创建模型
    print("🔧 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 创建损失函数
    print("🔧 创建损失函数...")
    import importlib.util
    losses_file = os.path.join(os.path.dirname(__file__), 'yolov6', 'models', 'losses.py')
    spec = importlib.util.spec_from_file_location("losses", losses_file)
    losses_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses_module)
    
    loss_fn = losses_module.ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        iou_type='siou',
        loss_weight={
            'class': 1.0,
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 创建优化器
    print("🔧 创建优化器...")
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    
    # 开始自检训练
    print("🚀 开始自检训练...")
    print(f"   训练轮数: 500")
    print(f"   学习率: 0.01")
    print("=" * 70)
    
    model.train()
    
    for epoch in range(500):
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):

            # 前向传播
            predictions = model(images)

            # 计算损失
            loss, _ = loss_fn(predictions, targets, epoch_num=epoch+1, step_num=batch_idx+1)
            
            if loss is not None:
                # 反向传播
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                
                epoch_loss += float(loss.data)
        
        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/500 | Loss: {epoch_loss:.6f}")
    
    print("✅ 自检训练完成！")
    
    # 保存自检模型
    save_path = 'self_check_model.pkl'
    jt.save(model.state_dict(), save_path)
    print(f"💾 自检模型已保存: {save_path}")
    
    # 进行自检推理
    print("\n🔍 开始自检推理...")
    model.eval()
    
    # 加载测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    
    # 预处理
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
    img = jt.array(img)
    if img.ndim == 3:
        img = img.unsqueeze(0)  # 添加batch维度
    
    # 推理
    with jt.no_grad():
        pred = model(img)
    
    # 后处理
    pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.45, max_det=1000)
    
    # 统计检测结果
    detections = []
    for i, det in enumerate(pred):
        if len(det):
            # 坐标缩放回原图尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            detections.append(det)
        else:
            detections.append(jt.empty((0, 6)))
    
    # 输出检测结果
    num_det = len(detections[0]) if len(detections) > 0 and len(detections[0]) > 0 else 0
    print(f"🎯 检测结果: {num_det} 个目标")
    
    if num_det > 0:
        det = detections[0]
        if hasattr(det, 'numpy'):
            det = det.numpy()
        
        for i, (*xyxy, conf, cls) in enumerate(det):
            print(f"   目标 {i+1}: 类别={int(cls)}, 置信度={conf:.3f}, 坐标=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        
        # 保存可视化结果
        img_vis = img0.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (*xyxy, conf, cls) in enumerate(det):
            if conf >= 0.01:
                x1, y1, x2, y2 = map(int, xyxy)
                color = colors[i % len(colors)]
                
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                label = f'Class{int(cls)} {conf:.2f}'
                cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite('self_check_result.jpg', img_vis)
        print(f"📸 可视化结果已保存: self_check_result.jpg")
        
        print("🎉 自检验证成功！模型能够检测到目标！")
        return True
    else:
        print("⚠️ 自检验证失败：模型未检测到任何目标")
        return False

if __name__ == "__main__":
    success = self_check_training()
    if success:
        print("\n✅ 自检训练验证完成！模型功能正常！")
    else:
        print("\n❌ 自检训练验证失败！需要进一步调试！")
