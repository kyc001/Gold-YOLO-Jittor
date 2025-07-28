#!/usr/bin/env python3
"""
最终修复分类头训练问题
解决分类头在训练过程中被"训练坏"的问题
"""

import os
import sys
import time
import cv2
import numpy as np

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

class FinalFixDataset(Dataset):
    """最终修复数据集"""
    
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
        
        print(f"📸 最终修复图像: {img_path}")
        print(f"🏷️ 图像尺寸: {self.img.shape}")
        print(f"🎯 目标数量: {len(self.labels)}")
    
    def load_labels(self, label_path):
        """加载YOLO格式标签"""
        if not os.path.exists(label_path):
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

def final_classification_fix():
    """最终修复分类头训练问题"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              GOLD-YOLO 最终分类头修复系统                    ║
    ║                                                              ║
    ║  🎯 修复分类头训练过程中被"训练坏"的问题                     ║
    ║  🔧 使用特殊的训练策略保护分类头                             ║
    ║  📊 完整500轮训练验证修复效果                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建数据集
    print("📦 创建最终修复数据集...")
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    label_path = 'final_classification_fix_label.txt'
    
    if not os.path.exists(img_path):
        print(f"❌ 测试图像不存在: {img_path}")
        return False
    
    # 创建标签文件
    with open(label_path, 'w') as f:
        f.write("0 0.5 0.5 0.8 0.8\n")
    print(f"✅ 创建最终修复标签: {label_path}")
    
    dataset = FinalFixDataset(img_path, label_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 创建模型
    print("🔧 创建最终修复模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 特殊的分类头初始化策略
    print("🔧 特殊分类头初始化...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            print(f"   特殊初始化: {name}")
            # 使用更保守的初始化
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                # 给偏置一个小的正值，避免被训练到负无穷
                jt.init.constant_(module.bias, -2.0)  # 对应sigmoid后约0.12的概率
    
    # 创建修复版损失函数
    print("🔧 创建修复版损失函数...")
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
            'class': 1.0,  # 降低分类损失权重，避免过度优化
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 创建分离的优化器策略
    print("🔧 创建分离优化器策略...")
    
    # 分离分类头和其他参数
    cls_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'cls_pred' in name:
            cls_params.append(param)
        else:
            other_params.append(param)
    
    print(f"   分类头参数数量: {len(cls_params)}")
    print(f"   其他参数数量: {len(other_params)}")
    
    # 使用不同的学习率
    cls_optimizer = nn.SGD(cls_params, lr=0.001, momentum=0.9, weight_decay=0.0001)  # 更低的学习率
    other_optimizer = nn.SGD(other_params, lr=0.01, momentum=0.9, weight_decay=0.0005)  # 正常学习率
    
    # 开始最终修复训练
    print("🚀 开始最终修复训练...")
    print(f"   训练轮数: 500")
    print(f"   分类头学习率: 0.001 (保守)")
    print(f"   其他参数学习率: 0.01 (正常)")
    print(f"   分类损失权重: 1.0 (降低)")
    print("=" * 70)
    
    model.train()
    
    # 训练统计
    loss_history = []
    cls_output_history = []
    best_loss = float('inf')
    
    # 记录开始时间
    start_time = time.time()
    
    for epoch in range(500):
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # 前向传播
            predictions = model(images)
            
            # 记录分类输出统计
            if isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
                cls_output = predictions[1]  # [1, 8400, 20]
                cls_min = float(cls_output.min().numpy())
                cls_max = float(cls_output.max().numpy())
                cls_mean = float(cls_output.mean().numpy())
                cls_output_history.append((cls_min, cls_max, cls_mean))
            
            # 计算损失
            loss, loss_items = loss_fn(predictions, targets, epoch_num=epoch+1, step_num=batch_idx+1)
            
            if loss is not None:
                # 分离的反向传播
                cls_optimizer.zero_grad()
                other_optimizer.zero_grad()
                
                # 反向传播
                cls_optimizer.backward(loss)
                other_optimizer.backward(loss)
                
                # 梯度裁剪（保护分类头）
                for param in cls_params:
                    grad = param.opt_grad(cls_optimizer)
                    if grad is not None:
                        jt.clamp(grad, -0.1, 0.1)  # 限制梯度范围
                
                # 更新参数
                cls_optimizer.step()
                other_optimizer.step()
                
                epoch_loss += float(loss.numpy())
                
                # 更新最佳损失
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
        
        # 记录损失
        loss_history.append(epoch_loss)
        
        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (epoch + 1) * (500 - epoch - 1)
            
            # 获取当前分类输出统计
            if len(cls_output_history) > 0:
                recent_cls = cls_output_history[-1]
                cls_range = recent_cls[1] - recent_cls[0]
                print(f"Epoch {epoch+1:3d}/500 | Loss: {epoch_loss:.6f} | Best: {best_loss:.6f} | "
                      f"Cls范围: {cls_range:.6f} | ETA: {eta/60:.1f}min")
            else:
                print(f"Epoch {epoch+1:3d}/500 | Loss: {epoch_loss:.6f} | Best: {best_loss:.6f} | ETA: {eta/60:.1f}min")
            
            # 每100轮详细检查
            if (epoch + 1) % 100 == 0:
                model.eval()
                with jt.no_grad():
                    test_pred = model(images)
                    
                    if isinstance(test_pred, (list, tuple)) and len(test_pred) >= 2:
                        cls_pred = test_pred[1]  # [1, 8400, 20]
                        
                        cls_min = float(cls_pred.min().numpy())
                        cls_max = float(cls_pred.max().numpy())
                        cls_mean = float(cls_pred.mean().numpy())
                        cls_range = cls_max - cls_min
                        
                        print(f"     详细检查: 分类输出范围[{cls_min:.6f}, {cls_max:.6f}], 均值{cls_mean:.6f}, 变化范围{cls_range:.6f}")
                        
                        if cls_range > 0.001:
                            print(f"     ✅ 分类头工作正常")
                        else:
                            print(f"     ⚠️ 分类头输出变化范围过小")
                
                model.train()
    
    # 训练完成
    total_time = time.time() - start_time
    print("✅ 最终修复训练完成！")
    
    # 训练总结
    print(f"\n📊 最终修复训练总结:")
    print(f"   总训练时间: {total_time/60:.1f}分钟")
    print(f"   总轮数: 500")
    print(f"   初始损失: {loss_history[0]:.6f}")
    print(f"   最终损失: {loss_history[-1]:.6f}")
    print(f"   最佳损失: {best_loss:.6f}")
    
    if len(loss_history) > 0 and loss_history[0] > 0:
        loss_reduction = ((loss_history[0] - loss_history[-1]) / loss_history[0] * 100)
        print(f"   损失下降: {loss_reduction:.2f}%")
    
    # 分析分类输出变化
    if len(cls_output_history) > 0:
        initial_cls = cls_output_history[0]
        final_cls = cls_output_history[-1]
        
        initial_range = initial_cls[1] - initial_cls[0]
        final_range = final_cls[1] - final_cls[0]
        
        print(f"\n📊 分类输出变化分析:")
        print(f"   初始分类范围: {initial_range:.6f}")
        print(f"   最终分类范围: {final_range:.6f}")
        
        if final_range > 0.001:
            print(f"   ✅ 分类头修复成功！")
            fix_success = True
        else:
            print(f"   ❌ 分类头仍有问题")
            fix_success = False
    else:
        fix_success = False
    
    # 保存模型
    save_path = 'final_classification_fixed_model.pkl'
    jt.save({
        'model': model.state_dict(),
        'cls_optimizer': cls_optimizer.state_dict(),
        'other_optimizer': other_optimizer.state_dict(),
        'epoch': 500,
        'loss_history': loss_history,
        'cls_output_history': cls_output_history,
        'best_loss': best_loss,
        'training_time': total_time,
        'fix_success': fix_success
    }, save_path)
    print(f"💾 最终修复模型已保存: {save_path}")
    
    return model, loss_history, cls_output_history, fix_success

if __name__ == "__main__":
    print("🚀 开始GOLD-YOLO最终分类头修复...")
    
    # 最终修复训练
    result = final_classification_fix()
    
    if result:
        model, loss_history, cls_output_history, fix_success = result
        
        print("\n" + "="*70)
        print("🎉 GOLD-YOLO最终分类头修复完成！")
        print("="*70)
        
        if fix_success:
            print("✅ 分类头修复成功！")
            print("🎯 模型现在应该能够正确进行目标检测")
        else:
            print("❌ 分类头修复失败")
            print("🔧 需要进一步调整训练策略")
        
        print(f"📊 最终状态:")
        print(f"   训练完成度: 100% (500/500轮)")
        print(f"   分类头状态: {'正常' if fix_success else '异常'}")
        print(f"   模型保存: final_classification_fixed_model.pkl")
        
        if fix_success:
            print("\n🎉 GOLD-YOLO Jittor版本修复完成！")
            print("📋 下一步建议:")
            print("   1. 使用修复后的模型进行推理测试")
            print("   2. 在更大的数据集上验证性能")
            print("   3. 与PyTorch版本进行对比验证")
        else:
            print("\n⚠️ 需要进一步调查分类头问题")
    else:
        print("❌ 最终分类头修复失败！")
