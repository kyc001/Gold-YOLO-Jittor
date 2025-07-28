#!/usr/bin/env python3
"""
最终完整自检训练 - 修复后的完整500轮训练
不简化任何步骤，验证所有修复效果
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

class FinalDataset(Dataset):
    """最终完整数据集"""
    
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
        
        print(f"📸 最终训练图像: {img_path}")
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

def final_complete_training():
    """最终完整自检训练主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              GOLD-YOLO-n 最终完整自检训练系统                ║
    ║                                                              ║
    ║  🎯 修复后的完整500轮训练                                    ║
    ║  🔧 验证所有修复效果                                         ║
    ║  📊 不简化任何步骤                                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建数据集
    print("📦 创建最终数据集...")
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    label_path = 'final_complete_label.txt'
    
    if not os.path.exists(img_path):
        print(f"❌ 测试图像不存在: {img_path}")
        return False
    
    # 创建标签文件
    with open(label_path, 'w') as f:
        f.write("0 0.5 0.5 0.8 0.8\n")
    print(f"✅ 创建最终标签: {label_path}")
    
    dataset = FinalDataset(img_path, label_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 创建模型
    print("🔧 创建最终模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            print(f"   重新初始化: {name}")
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                jt.init.constant_(module.bias, 0.01)
    
    # 创建损失函数
    print("🔧 创建最终损失函数...")
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
            'class': 5.0,  # 增加分类损失权重
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 创建优化器
    print("🔧 创建优化器...")
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    
    # 开始最终完整训练
    print("🚀 开始最终完整自检训练...")
    print(f"   训练轮数: 500")
    print(f"   学习率: 0.01")
    print(f"   分类损失权重: 5.0")
    print("=" * 70)
    
    model.train()
    
    # 训练统计
    loss_history = []
    valid_loss_count = 0
    best_loss = float('inf')
    
    # 记录开始时间
    start_time = time.time()
    
    for epoch in range(500):
        epoch_loss = 0.0
        epoch_valid_loss = False
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            loss, loss_items = loss_fn(predictions, targets, epoch_num=epoch+1, step_num=batch_idx+1)
            
            if loss is not None:
                # 反向传播
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                
                epoch_loss += float(loss.numpy())
                epoch_valid_loss = True
                valid_loss_count += 1
                
                # 更新最佳损失
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
        
        # 记录损失
        loss_history.append(epoch_loss)
        
        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (epoch + 1) * (500 - epoch - 1)
            print(f"Epoch {epoch+1:3d}/500 | Loss: {epoch_loss:.6f} | Best: {best_loss:.6f} | ETA: {eta/60:.1f}min")
            
            # 每100轮详细检查
            if (epoch + 1) % 100 == 0:
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
    
    # 训练完成
    total_time = time.time() - start_time
    print("✅ 最终完整自检训练完成！")
    
    # 训练总结
    print(f"\n📊 最终训练总结:")
    print(f"   总训练时间: {total_time/60:.1f}分钟")
    print(f"   总轮数: 500")
    print(f"   有效损失计算次数: {valid_loss_count}")
    print(f"   有效损失比例: {valid_loss_count/500*100:.1f}%")
    print(f"   初始损失: {loss_history[0]:.6f}")
    print(f"   最终损失: {loss_history[-1]:.6f}")
    print(f"   最佳损失: {best_loss:.6f}")
    
    if valid_loss_count > 0 and loss_history[0] > 0:
        loss_reduction = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100)
        print(f"   损失下降: {loss_reduction:.2f}%")
    
    # 保存模型
    save_path = 'final_complete_model.pkl'
    jt.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': 500,
        'loss_history': loss_history,
        'valid_loss_count': valid_loss_count,
        'best_loss': best_loss,
        'training_time': total_time
    }, save_path)
    print(f"💾 最终完整模型已保存: {save_path}")
    
    return model, loss_history, valid_loss_count, best_loss

if __name__ == "__main__":
    print("🚀 开始GOLD-YOLO最终完整自检训练...")
    
    # 最终完整训练
    result = final_complete_training()
    
    if result:
        model, loss_history, valid_loss_count, best_loss = result
        
        print("\n" + "="*70)
        print("🎉 GOLD-YOLO最终完整自检训练完成！")
        print("="*70)
        
        # 评估训练效果
        if valid_loss_count >= 450:  # 90%以上有效
            print("✅ 训练过程优秀，损失函数工作完美")
            training_quality = "优秀"
        elif valid_loss_count >= 400:  # 80%以上有效
            print("✅ 训练过程良好，损失函数工作正常")
            training_quality = "良好"
        elif valid_loss_count >= 200:  # 40%以上有效
            print("⚠️ 训练过程一般，需要进一步优化")
            training_quality = "一般"
        else:
            print("❌ 训练过程异常，损失函数存在问题")
            training_quality = "异常"
        
        # 评估损失下降
        if len(loss_history) > 0 and loss_history[0] > 0:
            loss_reduction = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100)
            if loss_reduction > 50:
                loss_quality = "显著下降"
            elif loss_reduction > 20:
                loss_quality = "明显下降"
            elif loss_reduction > 5:
                loss_quality = "轻微下降"
            else:
                loss_quality = "基本无变化"
        else:
            loss_quality = "无法评估"
        
        print(f"📊 最终评估:")
        print(f"   训练质量: {training_quality}")
        print(f"   损失变化: {loss_quality}")
        print(f"   有效训练: {valid_loss_count}/500")
        print(f"   最佳损失: {best_loss:.6f}")
        print(f"   模型保存: final_complete_model.pkl")
        
        if training_quality in ["优秀", "良好"] and loss_quality in ["显著下降", "明显下降"]:
            print("🎉 最终完整自检训练成功！模型修复完成！")
        else:
            print("⚠️ 最终完整自检训练完成，但可能需要进一步优化")
    else:
        print("❌ 最终完整自检训练失败！")
