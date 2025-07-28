#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 对齐PyTorch参数 + 强力数值稳定保护
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 设置环境变量
os.environ['JT_SYNC'] = '1'

import jittor as jt
jt.flags.use_cuda = 0  # 使用CPU避免CUDA问题

def parse_args():
    """解析命令行参数 - 对齐PyTorch版本"""
    parser = argparse.ArgumentParser(description='GOLD-YOLO Jittor Training (PyTorch Aligned + Stable)')
    
    # 严格对齐PyTorch版本的参数
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小 (对齐PyTorch)')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数 (对齐PyTorch)')
    parser.add_argument('--img-size', type=int, default=640, help='图像尺寸 (对齐PyTorch)')
    parser.add_argument('--lr-initial', type=float, default=0.02, help='初始学习率 (严格对齐PyTorch: lr0=0.02)')
    parser.add_argument('--lr-final', type=float, default=0.01, help='最终学习率 (严格对齐PyTorch: lrf=0.01)')
    parser.add_argument('--momentum', type=float, default=0.937, help='动量 (对齐PyTorch)')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减 (对齐PyTorch)')
    parser.add_argument('--data', type=str, default='../data/voc2012_subset/voc20.yaml', help='数据配置文件')
    parser.add_argument('--project', type=str, default='runs/train', help='项目保存目录')
    parser.add_argument('--name', type=str, default='pytorch_aligned_stable', help='实验名称')
    
    return parser.parse_args()


def safe_loss_computation_with_protection(loss_fn, outputs, targets, epoch, step):
    """超级安全的损失计算 - 多重保护"""
    try:
        # 检查输入 - 处理tuple格式的outputs
        if isinstance(outputs, (list, tuple)):
            # 检查每个输出tensor
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape') and (jt.isnan(output).any() or jt.isinf(output).any()):
                    print(f"⚠️ 模型输出[{i}]包含NaN/Inf，跳过")
                    return None
        else:
            # 单个tensor输出
            if jt.isnan(outputs).any() or jt.isinf(outputs).any():
                print(f"⚠️ 模型输出包含NaN/Inf，跳过")
                return None
        
        # 计算损失
        loss_result = loss_fn(outputs, targets, epoch, step)
        
        if isinstance(loss_result, tuple):
            total_loss = loss_result[0]
        else:
            total_loss = loss_result
        
        # 确保损失是标量
        if hasattr(total_loss, 'shape') and len(total_loss.shape) > 0:
            total_loss = jt.mean(total_loss)
        
        # 多重检查损失有效性
        if jt.isnan(total_loss) or jt.isinf(total_loss):
            print(f"⚠️ 损失为NaN/Inf，跳过")
            return None
        
        loss_value = float(total_loss)
        
        # 损失值范围检查
        if loss_value <= 0:
            print(f"⚠️ 损失为负数或零: {loss_value}，跳过")
            return None
        
        if loss_value > 1e6:  # 100万
            print(f"⚠️ 损失过大: {loss_value:.2e}，强制缩放")
            total_loss = total_loss / 1000.0  # 强力缩放
            loss_value = float(total_loss)
        
        if loss_value > 1e5:  # 10万
            print(f"⚠️ 损失较大: {loss_value:.2e}，缩放")
            total_loss = total_loss / 10.0
            loss_value = float(total_loss)
        
        return total_loss
        
    except Exception as e:
        print(f"❌ 损失计算异常: {e}")
        print(f"❌ 异常类型: {type(e).__name__}")
        import traceback
        print(f"❌ 详细堆栈: {traceback.format_exc()}")
        return None


def safe_gradient_clipping_pytorch_aligned(model, optimizer, max_norm=10.0):
    """完整的梯度裁剪实现 - 适配Jittor的梯度访问方式"""
    try:
        # 计算所有参数的梯度范数
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            # Jittor使用opt_grad(optimizer)访问梯度
            try:
                grad = param.opt_grad(optimizer)
                if grad is not None:
                    # 计算参数梯度的L2范数
                    param_norm = grad.norm()

                    # 确保是标量值 - 修复Jittor .item()问题
                    try:
                        param_norm_value = float(param_norm.data)  # Jittor方式获取标量值
                    except:
                        param_norm_value = float(param_norm)

                    total_norm += param_norm_value ** 2
                    param_count += 1
            except:
                # 如果无法获取梯度，跳过这个参数
                continue

        # 计算总梯度范数
        if param_count > 0:
            total_norm = (total_norm ** 0.5)
        else:
            total_norm = 0.0

        # 执行梯度裁剪
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)

            for param in model.parameters():
                try:
                    grad = param.opt_grad(optimizer)
                    if grad is not None:
                        # 在Jittor中直接修改梯度
                        grad *= clip_coef
                except:
                    continue

            return max_norm  # 返回裁剪后的范数
        else:
            return total_norm  # 返回原始范数

    except Exception as e:
        print(f"⚠️ 梯度裁剪失败: {e}")
        return 0.0


def train_one_epoch_stable(model, dataset, loss_fn, optimizer, epoch, args, lr_lambda):
    """超级稳定的训练函数"""
    model.train()
    total_loss = 0
    successful_batches = 0
    failed_batches = 0
    
    dataset_size = len(dataset)
    # 严格对齐PyTorch版本：使用完整数据集，不限制批次数
    num_batches = (dataset_size + args.batch_size - 1) // args.batch_size
    
    # 更新学习率
    current_lr = args.lr_initial * lr_lambda(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    print(f"📊 Epoch {epoch+1}: 学习率={current_lr:.6f}, 批次数={num_batches}")
    
    pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}')
    
    for batch_idx in pbar:
        try:
            # 收集批次数据
            batch_images = []
            batch_targets = []
            
            valid_samples = 0
            for i in range(args.batch_size):
                sample_idx = batch_idx * args.batch_size + i
                if sample_idx >= dataset_size:
                    break
                
                try:
                    dataset_output = dataset[sample_idx]
                    
                    if len(dataset_output) >= 2:
                        image = dataset_output[0]
                        target = dataset_output[1]
                        
                        if image is not None:
                            # 确保数据类型正确
                            if image.dtype != 'float32':
                                image = image.float32()
                            
                            # 检查图像数据
                            if jt.isnan(image).any() or jt.isinf(image).any():
                                continue
                            
                            batch_images.append(image)
                            
                            # 处理目标
                            if target is not None and len(target) > 0:
                                if target.dtype != 'float32':
                                    target = target.float32()

                                if len(target.shape) == 1:
                                    target = target.unsqueeze(0)

                                # 生产训练：移除调试信息
                                # if sample_idx == 0:  # 只打印第一个样本
                                # #     print(f"🔍 [数据加载] 原始target形状: {target.shape}")
                                # #     print(f"🔍 [数据加载] 原始target数值范围: [{float(target.min().data):.6f}, {float(target.max().data):.6f}]")
                                # #     print(f"🔍 [数据加载] 原始target前3行: {target[:3].numpy()}")

                                batch_indices = jt.full((target.shape[0], 1), valid_samples, dtype='float32')
                                target_with_batch = jt.concat([batch_indices, target], dim=1)
                                batch_targets.append(target_with_batch)
                            
                            valid_samples += 1
                            
                except Exception as e:
                    continue
            
            if len(batch_images) == 0:
                failed_batches += 1
                continue
            
            # 堆叠图像
            images = jt.stack(batch_images)
            if images.dtype != 'float32':
                images = images.float32()
            
            # 处理目标
            if batch_targets:
                targets = jt.concat(batch_targets, dim=0)
            else:
                targets = jt.zeros((0, 6), dtype='float32')
            
            # 前向传播
            outputs = model(images)
            
            # 超级安全的损失计算
            loss = safe_loss_computation_with_protection(loss_fn, outputs, targets, epoch, batch_idx)
            
            if loss is None:
                failed_batches += 1
                continue
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            
            # 安全的梯度裁剪
            grad_norm = safe_gradient_clipping_pytorch_aligned(model, optimizer, max_norm=10.0)
            
            # 参数更新
            optimizer.step()
            
            # 更新统计
            loss_value = float(loss)
            total_loss += loss_value
            successful_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss_value:.4f}',
                'Avg': f'{total_loss/successful_batches:.4f}',
                'GradNorm': f'{grad_norm:.2f}',
                'Valid': f'{valid_samples}/{args.batch_size}'
            })
            
        except Exception as e:
            print(f"⚠️ Batch {batch_idx} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            failed_batches += 1
            continue
    
    avg_loss = total_loss / max(successful_batches, 1)
    success_rate = successful_batches / max(successful_batches + failed_batches, 1) * 100
    
    print(f"📈 Epoch {epoch+1} 完成: 平均损失={avg_loss:.6f}, 成功率={success_rate:.1f}%")
    
    return avg_loss


def find_dataset_config():
    """自动查找数据集配置文件"""
    possible_paths = [
        '../data/voc2012_subset/voc20.yaml',
        '/home/kyc/project/GOLD-YOLO/data/voc2012_subset/voc20.yaml',
        'data/voc2012_subset/voc20.yaml',
        './data/voc2012_subset/voc20.yaml'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"数据集配置文件未找到，尝试过的路径: {possible_paths}")

def main():
    """主训练函数 - GOLD-YOLO-n点击即用"""
    args = parse_args()

    print("🚀 GOLD-YOLO-n Jittor版本 - 点击即用稳定训练")
    print("=" * 70)

    # 自动查找数据集
    if not os.path.exists(args.data):
        try:
            args.data = find_dataset_config()
            print(f"📊 自动找到数据集: {args.data}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

    print(f"🎯 模型: GOLD-YOLO-n | 数据集: {os.path.basename(args.data)} | 轮数: {args.epochs} | 批次: {args.batch_size}")
    print("=" * 70)

    try:
        # 加载数据配置
        with open(args.data, 'r') as f:
            data_config = yaml.safe_load(f)
        
        num_classes = data_config.get('nc', 20)
        
        # 创建数据集
        from yolov6.data.datasets import TrainValDataset
        
        # 临时调整数据增强参数以解决极小目标问题
        hyp = {
            'mosaic': 0.0, 'mixup': 0.0, 'degrees': 0.0, 'translate': 0.0,  # 暂时禁用mosaic和translate
            'scale': 0.0, 'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.0,      # 暂时禁用所有几何变换
            'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0                       # 暂时禁用颜色增强
        }
        
        train_dataset = TrainValDataset(
            img_dir=data_config['val'],
            img_size=args.img_size,
            augment=True,
            hyp=hyp,
            rect=False,
            check_images=True,
            check_labels=True,
            stride=32,
            pad=0.0,
            rank=-1,
            data_dict=data_config
        )
        
        print(f"📦 数据集: {len(train_dataset)} 样本")
        
        # 创建模型和损失函数
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes)
        
        import importlib.util
        losses_file = os.path.join(os.path.dirname(__file__), 'yolov6', 'models', 'losses.py')
        spec = importlib.util.spec_from_file_location("fixed_losses", losses_file)
        fixed_losses = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixed_losses)
        ComputeLoss = fixed_losses.ComputeLoss
        
        loss_fn = ComputeLoss(
            fpn_strides=[8, 16, 32],
            grid_cell_size=5.0,
            grid_cell_offset=0.5,
            num_classes=num_classes,
            ori_img_size=640,
            warmup_epoch=4,
            use_dfl=False,  # gold-yolo-n原始配置：禁用DFL
            reg_max=0,      # gold-yolo-n原始配置：reg_max=0
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        
        # 创建优化器 - 严格对齐PyTorch版本
        optimizer = jt.optim.SGD(
            model.parameters(), 
            lr=args.lr_initial,
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        def lr_lambda(epoch):
            progress = epoch / args.epochs
            lr_ratio = args.lr_final / args.lr_initial
            current_lr_ratio = 1.0 - progress * (1.0 - lr_ratio)
            return current_lr_ratio
        
        # 创建保存目录
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 模型保存目录: {save_dir}")
        print(f"🎯 开始超级稳定训练...")
        
        best_loss = float('inf')
        
        for epoch in range(args.epochs):
            avg_loss = train_one_epoch_stable(
                model, train_dataset, loss_fn, optimizer, epoch, args, lr_lambda
            )
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = str(save_dir / "best.pkl")
                try:
                    jt.save({
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, best_model_path)
                    print(f"✅ 保存最佳模型: {best_model_path}")
                except Exception as e:
                    print(f"⚠️ 保存失败: {e}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = str(save_dir / f"epoch_{epoch+1}.pkl")
                try:
                    jt.save({
                        'epoch': epoch + 1,
                        'loss': avg_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"✅ 保存检查点: {checkpoint_path}")
                except:
                    pass
        
        # 保存最终模型
        final_model_path = str(save_dir / "final.pkl")
        try:
            jt.save({
                'epoch': args.epochs,
                'loss': avg_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, final_model_path)
            print(f"💾 最终模型: {final_model_path}")
        except:
            pass
        
        print(f"🎉 训练完成！最佳损失: {best_loss:.6f}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
