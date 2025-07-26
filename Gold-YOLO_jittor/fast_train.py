#!/usr/bin/env python3
"""
快速训练脚本 - 优化Jittor性能
"""

import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# 设置最优性能环境变量
os.environ['JT_SYNC'] = '0'  # 异步执行
os.environ['JT_CUDA_MEMORY_POOL'] = '1'  # 内存池
os.environ['JT_ENABLE_TUNER'] = '1'  # 自动调优
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU

import jittor as jt

# 强制GPU模式和性能优化
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0  # 禁用懒执行

print(f"🚀 Jittor快速训练模式")
print(f"   版本: {jt.__version__}")
print(f"   GPU: {jt.has_cuda}")
print(f"   use_cuda: {jt.flags.use_cuda}")

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description='GOLD-YOLO Fast Training')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        from yolov6.models.losses.loss import ComputeLoss
        from data_loader import create_real_dataloader
        
        print(f"\n🏗️ 创建模型...")
        model = create_perfect_gold_yolo_model('gold_yolo-n', 20)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # 创建优化器
        print(f"\n🔧 创建优化器...")
        optimizer = jt.optim.SGD(model.parameters(), lr=args.lr, momentum=0.937, weight_decay=0.0005)
        
        # 创建损失函数
        print(f"\n📈 创建损失函数...")
        loss_fn = ComputeLoss(num_classes=20)
        
        # 创建数据配置
        data_config = {
            'nc': 20,
            'path': '/home/kyc/project/GOLD-YOLO/data/voc2012_subset',
            'train': 'train',
            'val': 'val',
            'names': [f'class_{i}' for i in range(20)]
        }
        
        # 创建数据加载器
        print(f"\n📦 创建数据加载器...")
        train_dataloader = create_real_dataloader(data_config, args.batch_size, is_train=True)
        
        print(f"\n🚀 开始快速训练 {args.epochs} 轮...")
        print("=" * 80)
        
        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            num_batches = len(train_dataloader)
            
            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # 前向传播
                images = images.float32()
                outputs = model(images)
                
                # 计算损失
                result = loss_fn(outputs, targets, epoch, batch_idx)
                
                if isinstance(result, (list, tuple)) and len(result) == 2:
                    loss, loss_items = result
                    loss_value = float(loss)
                else:
                    loss = result
                    loss_value = float(loss)
                    loss_items = [loss_value, 0, 0]
                
                # 反向传播
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                
                # 更新统计
                total_loss += loss_value
                avg_loss = total_loss / (batch_idx + 1)
                
                # 更新进度条
                if hasattr(loss_items, 'shape') and len(loss_items) >= 3:
                    iou_loss = float(loss_items[0])
                    dfl_loss = float(loss_items[1])
                    cls_loss = float(loss_items[2])
                    actual_iou = max(0.0, min(1.0, 1.0 - iou_loss)) if iou_loss <= 2.0 else 0.0
                    
                    pbar.set_postfix({
                        'Loss': f'{loss_value:.6f}',
                        'Avg': f'{avg_loss:.6f}',
                        'IoU': f'{actual_iou:.4f}',
                        'DFL': f'{dfl_loss:.4f}',
                        'Cls': f'{cls_loss:.4f}'
                    })
                else:
                    pbar.set_postfix({
                        'Loss': f'{loss_value:.6f}',
                        'Avg': f'{avg_loss:.6f}'
                    })
                
                # 每50个batch显示一次速度
                if batch_idx % 50 == 0 and batch_idx > 0:
                    print(f"\n   批次 {batch_idx}: 平均损失 {avg_loss:.6f}")
            
            print(f"\nEpoch {epoch+1} 完成 - 平均损失: {avg_loss:.6f}")
            
            # 每10个epoch保存一次模型
            if (epoch + 1) % 2 == 0:
                save_path = f"runs/train/fast_train/epoch_{epoch+1}.pkl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                model.save(save_path)
                print(f"   模型已保存: {save_path}")
        
        print(f"\n🎉 训练完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
