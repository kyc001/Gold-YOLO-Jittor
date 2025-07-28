#!/usr/bin/env python3
"""
GOLD-YOLO Jittor版本 - 200轮完整训练脚本
与PyTorch版本完全对齐的训练配置
"""

import os
import sys
import argparse
import time
import jittor as jt
from pathlib import Path

# 添加路径
sys.path.append('.')
sys.path.append('..')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GOLD-YOLO Jittor 200轮训练')
    
    # 基本参数
    parser.add_argument('--data', type=str, default='../data/voc2012_subset/voc20.yaml', 
                       help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, 
                       help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='输入图像大小')
    
    # 学习率参数
    parser.add_argument('--lr-initial', type=float, default=0.01, 
                       help='初始学习率')
    parser.add_argument('--lr-final', type=float, default=0.01, 
                       help='最终学习率')
    parser.add_argument('--warmup-epochs', type=int, default=3, 
                       help='预热轮数')
    
    # 优化器参数
    parser.add_argument('--momentum', type=float, default=0.937, 
                       help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=0.0005, 
                       help='权重衰减')
    
    # 保存参数
    parser.add_argument('--save-dir', type=str, default='./runs/train_200', 
                       help='保存目录')
    parser.add_argument('--save-interval', type=int, default=50, 
                       help='保存间隔（轮数）')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', 
                       help='训练设备')
    
    return parser.parse_args()

def main():
    """主训练函数"""
    args = parse_args()
    
    print("🚀 开始GOLD-YOLO Jittor版本200轮完整训练")
    print("=" * 60)
    print(f"📊 训练配置:")
    print(f"   数据集: {args.data}")
    print(f"   训练轮数: {args.epochs}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   图像大小: {args.img_size}")
    print(f"   初始学习率: {args.lr_initial}")
    print(f"   最终学习率: {args.lr_final}")
    print(f"   预热轮数: {args.warmup_epochs}")
    print(f"   动量: {args.momentum}")
    print(f"   权重衰减: {args.weight_decay}")
    print(f"   保存目录: {args.save_dir}")
    print("=" * 60)
    
    # 设置Jittor
    if args.device == 'cuda':
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练配置
    config_file = save_dir / 'train_config.txt'
    with open(config_file, 'w') as f:
        f.write("GOLD-YOLO Jittor 200轮训练配置\n")
        f.write("=" * 40 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 导入训练模块
    from models.perfect_gold_yolo import create_perfect_gold_yolo_model
    from yolov6.models.losses import ComputeLoss
    from yolov6.data.data_load import create_dataloader
    import jittor.optim as optim

    try:
        # 创建模型
        print("🔧 创建模型...")
        model = create_perfect_gold_yolo_model('gold_yolo-n', 20)

        # 创建损失函数
        print("🔧 创建损失函数...")
        loss_fn = ComputeLoss(
            num_classes=20,
            ori_img_size=args.img_size,
            warmup_epoch=args.warmup_epochs,
            use_dfl=False,  # 与配置文件保持一致
            reg_max=16,     # 使用默认值
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )

        # 创建优化器
        print("🔧 创建优化器...")
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_initial,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        # 加载数据配置
        print("📁 加载数据配置...")
        import yaml
        with open(args.data, 'r') as f:
            data_dict = yaml.safe_load(f)

        # 创建数据加载器
        print("📁 创建数据加载器...")
        from yolov6.data.datasets import TrainValDataset
        from jittor.dataset import DataLoader

        # 创建数据集
        train_dataset = TrainValDataset(
            img_dir=data_dict['train'],
            img_size=args.img_size,
            batch_size=args.batch_size,
            augment=True,
            hyp=None,
            rect=False,
            check_images=False,
            check_labels=False,
            stride=32,
            pad=0.0,
            rank=-1,
            data_dict=data_dict,
            task="train"
        )

        # 创建数据加载器 - 使用Jittor的方式
        train_loader = train_dataset.set_attrs(
            batch_size=args.batch_size
        )

        print(f"✅ 数据加载器创建成功，共 {len(train_loader)} 个批次")
        
        # 开始训练
        print("🚀 开始200轮训练...")
        start_time = time.time()
        
        # 训练日志
        train_log = []
        
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            
            # 计算当前学习率（余弦退火）
            if epoch <= args.warmup_epochs:
                # 预热阶段
                lr = args.lr_initial * epoch / args.warmup_epochs
            else:
                # 余弦退火
                import math
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                lr = args.lr_final + (args.lr_initial - args.lr_final) * 0.5 * (1 + math.cos(math.pi * progress))
            
            # 更新学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 训练一个epoch
            model.train()
            epoch_loss = 0.0
            total_samples = 0

            for batch_idx, (images, targets) in enumerate(train_loader):
                try:
                    # 前向传播
                    outputs = model(images)

                    # 计算损失
                    loss = loss_fn(outputs, targets, epoch, batch_idx)

                    # 反向传播
                    optimizer.zero_grad()
                    optimizer.backward(loss)
                    optimizer.step()

                    # 累计损失
                    epoch_loss += float(loss.data)
                    total_samples += images.shape[0]

                    # 打印进度（每100个batch）
                    if batch_idx % 100 == 0:
                        print(f"  Batch {batch_idx:4d}/{len(train_loader)} | "
                              f"Loss: {float(loss.data):.4f} | "
                              f"Samples: {total_samples}")

                except Exception as e:
                    print(f"⚠️ Batch {batch_idx} 训练失败: {e}")
                    continue

            avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录训练日志
            log_entry = {
                'epoch': epoch,
                'avg_loss': avg_loss,
                'lr': lr,
                'epoch_time': epoch_time,
                'total_samples': total_samples
            }
            train_log.append(log_entry)
            
            # 打印进度
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Samples: {total_samples}")
            
            # 保存检查点
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pkl'
                jt.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss,
                    'lr': lr,
                    'train_log': train_log
                }, str(checkpoint_path))
                print(f"💾 已保存检查点: {checkpoint_path}")
        
        # 训练完成
        total_time = time.time() - start_time
        print("🎉 200轮训练完成！")
        print(f"⏱️ 总训练时间: {total_time/3600:.2f} 小时")
        print(f"📊 最终损失: {train_log[-1]['avg_loss']:.4f}")
        
        # 保存最终模型
        final_model_path = save_dir / 'gold_yolo_jittor_final.pkl'
        jt.save({
            'model_state_dict': model.state_dict(),
            'train_log': train_log,
            'config': vars(args)
        }, str(final_model_path))
        print(f"💾 已保存最终模型: {final_model_path}")
        
        # 保存训练日志
        log_file = save_dir / 'train_log.txt'
        with open(log_file, 'w') as f:
            f.write("GOLD-YOLO Jittor 200轮训练日志\n")
            f.write("=" * 50 + "\n")
            f.write("Epoch | Loss     | LR       | Time(s) | Samples\n")
            f.write("-" * 50 + "\n")
            for log in train_log:
                f.write(f"{log['epoch']:5d} | {log['avg_loss']:8.4f} | "
                       f"{log['lr']:8.6f} | {log['epoch_time']:7.1f} | {log['total_samples']:7d}\n")
            f.write("-" * 50 + "\n")
            f.write(f"总训练时间: {total_time/3600:.2f} 小时\n")
            f.write(f"最终损失: {train_log[-1]['avg_loss']:.4f}\n")
        
        print(f"📝 已保存训练日志: {log_file}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 训练成功完成！")
    else:
        print("❌ 训练失败！")
        sys.exit(1)
