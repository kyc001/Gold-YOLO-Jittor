#!/usr/bin/env python3
"""
监控版自检训练 - 深入监控损失函数内部
不简化任何步骤，完整500轮训练，详细监控每个环节
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset, DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

class MonitoredDataset(Dataset):
    """监控版数据集"""
    
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
        
        print(f"📸 监控图像: {img_path}")
        print(f"🏷️ 图像尺寸: {self.img.shape}")
        print(f"🎯 目标数量: {len(self.labels)}")
        if len(self.labels) > 0:
            print(f"🎯 目标详情: {self.labels}")
    
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
        return 1
    
    def __getitem__(self, idx):
        # 图像预处理
        img = letterbox(self.img, new_shape=self.img_size, stride=32, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        
        # 标签处理
        if len(self.labels) > 0:
            label = self.labels[0]
            labels_out = jt.array([[label[0], label[1], label[2], label[3], label[4], 0]], dtype='float32')
        else:
            labels_out = jt.zeros((1, 6), dtype='float32')
        
        return jt.array(img, dtype='float32'), labels_out

def create_monitored_loss_function():
    """创建监控版损失函数"""
    import importlib.util
    losses_file = os.path.join(os.path.dirname(__file__), 'yolov6', 'models', 'losses.py')
    spec = importlib.util.spec_from_file_location("losses", losses_file)
    losses_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses_module)
    
    # 创建原始损失函数
    original_loss_fn = losses_module.ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        iou_type='siou',
        loss_weight={
            'class': 5.0,  # 增加分类损失权重
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 包装损失函数以添加监控
    class MonitoredLoss:
        def __init__(self, loss_fn):
            self.loss_fn = loss_fn
            self.call_count = 0
        
        def __call__(self, predictions, targets, epoch_num, step_num):
            self.call_count += 1
            
            # 详细监控输入
            if self.call_count <= 5 or self.call_count % 100 == 0:
                print(f"\n🔍 损失函数调用 #{self.call_count} (Epoch {epoch_num}):")
                print(f"   输入targets形状: {targets.shape}")
                print(f"   输入targets内容: {targets.numpy()}")
                
                if isinstance(predictions, (list, tuple)):
                    print(f"   预测输出数量: {len(predictions)}")
                    for i, pred in enumerate(predictions):
                        if hasattr(pred, 'shape'):
                            print(f"     预测{i}形状: {pred.shape}")
                else:
                    print(f"   预测输出形状: {predictions.shape}")
            
            # 调用原始损失函数
            try:
                loss, loss_items = self.loss_fn(predictions, targets, epoch_num, step_num)
                
                # 监控输出
                if self.call_count <= 5 or self.call_count % 100 == 0:
                    if loss is not None:
                        print(f"   ✅ 损失计算成功: {float(loss.numpy()):.6f}")
                        if loss_items is not None:
                            print(f"   损失详情: {loss_items}")
                    else:
                        print(f"   ❌ 损失计算返回None")
                
                return loss, loss_items
                
            except Exception as e:
                print(f"   ❌ 损失计算异常: {e}")
                if self.call_count <= 5:
                    import traceback
                    traceback.print_exc()
                return None, None
    
    return MonitoredLoss(original_loss_fn)

def monitored_self_check_training():
    """监控版自检训练"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              GOLD-YOLO-n 监控版自检训练系统                  ║
    ║                                                              ║
    ║  🔍 深入监控损失函数内部                                     ║
    ║  🎯 完整500轮训练，不简化任何步骤                            ║
    ║  📊 详细记录每个训练环节                                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建数据集
    print("📦 创建监控数据集...")
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    label_path = 'monitored_self_check_label.txt'
    
    if not os.path.exists(img_path):
        print(f"❌ 测试图像不存在: {img_path}")
        return False
    
    # 创建标签文件
    with open(label_path, 'w') as f:
        f.write("0 0.5 0.5 0.8 0.8\n")
    print(f"✅ 创建监控标签: {label_path}")
    
    dataset = MonitoredDataset(img_path, label_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 创建模型
    print("🔧 创建监控模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            print(f"   重新初始化: {name}")
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                jt.init.constant_(module.bias, 0.01)
    
    # 创建监控版损失函数
    print("🔧 创建监控版损失函数...")
    loss_fn = create_monitored_loss_function()
    
    # 创建优化器
    print("🔧 创建优化器...")
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    
    # 开始监控训练
    print("🚀 开始监控版自检训练...")
    print(f"   训练轮数: 500")
    print(f"   学习率: 0.01")
    print(f"   分类损失权重: 5.0")
    print("=" * 70)
    
    model.train()
    
    # 训练统计
    loss_history = []
    valid_loss_count = 0
    
    for epoch in range(500):
        epoch_loss = 0.0
        epoch_valid_loss = False
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # 前向传播
            predictions = model(images)
            
            # 计算损失（带监控）
            loss, loss_items = loss_fn(predictions, targets, epoch_num=epoch+1, step_num=batch_idx+1)
            
            if loss is not None:
                # 反向传播
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                
                epoch_loss += float(loss.numpy())
                epoch_valid_loss = True
                valid_loss_count += 1
            else:
                if epoch < 10:  # 前10轮详细记录
                    print(f"   Epoch {epoch+1}: 损失为None")
        
        # 记录损失
        loss_history.append(epoch_loss)
        
        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/500 | Loss: {epoch_loss:.6f} | 有效损失计算: {valid_loss_count}")
            
            # 每100轮详细检查
            if (epoch + 1) % 100 == 0:
                print(f"   详细检查 Epoch {epoch+1}:")
                model.eval()
                with jt.no_grad():
                    test_pred = model(images)
                    
                    if isinstance(test_pred, (list, tuple)) and len(test_pred) >= 3:
                        # 训练模式输出：[features, cls_pred, reg_pred]
                        cls_pred = test_pred[1]  # [1, 8400, 20]
                        reg_pred = test_pred[2]  # [1, 8400, 4]
                        
                        cls_min = float(cls_pred.min().numpy())
                        cls_max = float(cls_pred.max().numpy())
                        cls_mean = float(cls_pred.mean().numpy())
                        cls_range = cls_max - cls_min
                        
                        reg_min = float(reg_pred.min().numpy())
                        reg_max = float(reg_pred.max().numpy())
                        
                        print(f"     分类输出: 范围[{cls_min:.6f}, {cls_max:.6f}], 均值{cls_mean:.6f}, 变化范围{cls_range:.6f}")
                        print(f"     回归输出: 范围[{reg_min:.6f}, {reg_max:.6f}]")
                    else:
                        # 推理模式输出：[1, 8400, 25]
                        pred = test_pred[0] if isinstance(test_pred, (list, tuple)) else test_pred
                        cls_conf = pred[:, :, 5:]  # 类别置信度
                        
                        cls_min = float(cls_conf.min().numpy())
                        cls_max = float(cls_conf.max().numpy())
                        cls_range = cls_max - cls_min
                        
                        print(f"     推理模式类别置信度: 范围[{cls_min:.6f}, {cls_max:.6f}], 变化范围{cls_range:.6f}")
                
                model.train()
    
    print("✅ 监控版自检训练完成！")
    
    # 训练总结
    print(f"\n📊 训练总结:")
    print(f"   总轮数: 500")
    print(f"   有效损失计算次数: {valid_loss_count}")
    print(f"   有效损失比例: {valid_loss_count/500*100:.1f}%")
    print(f"   初始损失: {loss_history[0]:.6f}")
    print(f"   最终损失: {loss_history[-1]:.6f}")
    
    if valid_loss_count > 0:
        print(f"   损失下降: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%")
    
    # 保存模型
    save_path = 'monitored_self_check_model.pkl'
    jt.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': 500,
        'loss_history': loss_history,
        'valid_loss_count': valid_loss_count
    }, save_path)
    print(f"💾 监控版模型已保存: {save_path}")
    
    return model, loss_history, valid_loss_count

if __name__ == "__main__":
    print("🚀 开始GOLD-YOLO监控版自检训练...")
    
    # 监控训练
    result = monitored_self_check_training()
    
    if result:
        model, loss_history, valid_loss_count = result
        
        print("\n" + "="*70)
        print("🎉 GOLD-YOLO监控版自检训练完成！")
        print("="*70)
        
        if valid_loss_count > 400:  # 80%以上的损失计算有效
            print("✅ 训练过程正常，损失函数工作良好")
        elif valid_loss_count > 200:  # 40%以上有效
            print("⚠️ 训练过程部分正常，需要进一步优化")
        else:
            print("❌ 训练过程异常，损失函数存在问题")
        
        print(f"📊 最终统计:")
        print(f"   有效损失计算: {valid_loss_count}/500")
        print(f"   模型保存: monitored_self_check_model.pkl")
    else:
        print("❌ 监控版自检训练失败！")
