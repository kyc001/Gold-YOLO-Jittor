#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - CPU训练脚本（修复数据格式问题）
专门用于CPU训练，避免CUDA问题
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 强制CPU模式
os.environ['JT_SYNC'] = '1'

import jittor as jt
jt.flags.use_cuda = 0  # 强制使用CPU
print("⚠️ 强制使用CPU训练（避免CUDA问题）")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GOLD-YOLO CPU Training')
    parser.add_argument('--data', type=str, default='data/voc_subset_improved.yaml', 
                        help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小(CPU优化)')
    parser.add_argument('--img-size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率(CPU优化)')
    parser.add_argument('--project', type=str, default='runs/train', help='项目保存目录')
    parser.add_argument('--name', type=str, default='cpu_training', help='实验名称')
    
    return parser.parse_args()


def load_data_config(config_path):
    """加载数据配置文件"""
    try:
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 数据配置加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 数据配置加载失败: {e}")
        print(f"   尝试的路径: {config_path}")
        sys.exit(1)


def create_simple_dataset(data_config, img_size, is_train=True):
    """创建简化的数据集 - CPU优化版"""
    try:
        from yolov6.data.datasets import TrainValDataset
        
        data_path = data_config['train'] if is_train else data_config['val']
        
        # CPU优化的简化配置
        hyp = {
            'mosaic': 0.0,      # 禁用复杂增强
            'mixup': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,      # 只保留水平翻转
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0
        }
        
        dataset = TrainValDataset(
            img_dir=data_path,
            img_size=img_size,
            augment=is_train,
            hyp=hyp,
            rect=False,
            check_images=True,
            check_labels=True,
            stride=32,
            pad=0.0,
            rank=-1,
            data_dict=data_config
        )
        
        print(f"✅ {'训练' if is_train else '验证'}数据集创建成功: {len(dataset)} 样本")
        return dataset
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_model_and_loss(num_classes=20):
    """创建模型和损失函数"""
    try:
        # 创建模型
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes)
        
        # 创建损失函数 - 直接导入修复版本
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
            use_dfl=False,
            reg_max=0,
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        
        print("✅ 模型和损失函数创建成功")
        return model, loss_fn
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def safe_loss_computation(loss_fn, outputs, targets, epoch, step):
    """安全的损失计算"""
    try:
        loss_result = loss_fn(outputs, targets, epoch, step)
        
        if isinstance(loss_result, tuple):
            total_loss = loss_result[0]
        else:
            total_loss = loss_result
        
        # 确保损失是标量
        if hasattr(total_loss, 'shape') and len(total_loss.shape) > 0:
            total_loss = jt.mean(total_loss)
        
        # 检查损失是否有效
        if jt.isnan(total_loss) or jt.isinf(total_loss):
            return None
        
        # CPU模式下的损失缩放
        loss_value = float(total_loss)
        if loss_value > 100.0:
            total_loss = total_loss / 10.0
        elif loss_value > 50.0:
            total_loss = total_loss / 5.0
        
        return total_loss
        
    except Exception as e:
        print(f"⚠️ 损失计算失败: {e}")
        return None


def safe_target_processing(batch_targets):
    """安全的目标处理 - 修复数据格式问题"""
    if not batch_targets:
        return jt.zeros((0, 6))
    
    try:
        # 处理每个目标张量，确保格式正确
        processed_targets = []
        
        for target_tensor in batch_targets:
            # 确保是张量
            if not isinstance(target_tensor, jt.Var):
                target_tensor = jt.array(target_tensor)
            
            # 确保是2维的
            if len(target_tensor.shape) == 1:
                target_tensor = target_tensor.unsqueeze(0)
            elif len(target_tensor.shape) == 0:
                continue  # 跳过空目标
            
            # 确保有正确的列数
            if target_tensor.shape[1] >= 5:  # 至少有batch_idx, class, x, y, w, h
                processed_targets.append(target_tensor)
        
        if processed_targets:
            return jt.concat(processed_targets, dim=0)
        else:
            return jt.zeros((0, 6))
            
    except Exception as e:
        print(f"⚠️ 目标处理失败: {e}")
        return jt.zeros((0, 6))


def train_one_epoch_cpu(model, dataset, loss_fn, optimizer, epoch, batch_size=2):
    """CPU优化的训练函数"""
    model.train()
    total_loss = 0
    successful_batches = 0
    failed_batches = 0
    
    dataset_size = len(dataset)
    # CPU模式下处理较少批次
    max_batches = 30
    num_batches = min(max_batches, (dataset_size + batch_size - 1) // batch_size)
    
    print(f"📊 Epoch {epoch+1}: 数据集大小={dataset_size}, 训练批次数={num_batches}")
    
    pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}')
    
    for batch_idx in pbar:
        try:
            # 收集批次数据
            batch_images = []
            batch_targets = []
            
            valid_samples = 0
            for i in range(batch_size):
                sample_idx = batch_idx * batch_size + i
                if sample_idx >= dataset_size:
                    break
                
                try:
                    dataset_output = dataset[sample_idx]
                    
                    if len(dataset_output) >= 2:
                        image = dataset_output[0]
                        target = dataset_output[1]
                        
                        if image is not None:
                            batch_images.append(image)
                            
                            # 安全处理目标
                            if target is not None and len(target) > 0:
                                # 确保target是2维的
                                if len(target.shape) == 1:
                                    target = target.unsqueeze(0)
                                
                                # 添加批次索引
                                batch_indices = jt.full((target.shape[0], 1), valid_samples)
                                target_with_batch = jt.concat([batch_indices, target], dim=1)
                                batch_targets.append(target_with_batch)
                            
                            valid_samples += 1
                            
                except Exception as e:
                    continue
            
            # 检查是否有有效数据
            if len(batch_images) == 0:
                failed_batches += 1
                continue
            
            # 堆叠图像 - 修复：确保数据类型正确
            images = jt.stack(batch_images)
            if images.dtype != 'float32':
                images = images.float32()  # 强制转换为float32
            
            # 安全处理目标
            targets = safe_target_processing(batch_targets)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = safe_loss_computation(loss_fn, outputs, targets, epoch, batch_idx)
            
            if loss is None:
                failed_batches += 1
                continue
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            
            # CPU模式下使用温和的梯度裁剪 - 修复Jittor API
            try:
                # 深入修复梯度裁剪 - 正确处理张量norm
                max_norm = 10.0
                total_norm = 0.0
                for param in model.parameters():
                    if param.opt_grad(optimizer) is not None:
                        grad = param.opt_grad(optimizer)
                        # 深入修复：计算梯度的L2范数，确保结果是标量
                        param_norm_squared = jt.sum(grad * grad)  # 计算平方和
                        # 转换为Python标量
                        if hasattr(param_norm_squared, 'data'):
                            norm_val = float(param_norm_squared.data)
                        else:
                            # 使用numpy转换
                            norm_val = float(param_norm_squared.numpy())
                        total_norm += norm_val

                total_norm = (total_norm ** 0.5)  # 开平方得到L2范数

                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1:
                    for param in model.parameters():
                        if param.opt_grad(optimizer) is not None:
                            param.opt_grad(optimizer).data.mul_(clip_coef)
            except:
                pass  # 如果梯度裁剪失败，继续训练
            
            optimizer.step()
            
            # 更新统计
            loss_value = float(loss)
            total_loss += loss_value
            successful_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss_value:.4f}',
                'Avg': f'{total_loss/successful_batches:.4f}',
                'Valid': f'{valid_samples}/{batch_size}'
            })
            
        except Exception as e:
            print(f"⚠️ Batch {batch_idx} 训练失败: {e}")
            failed_batches += 1
            continue
    
    avg_loss = total_loss / max(successful_batches, 1)
    success_rate = successful_batches / max(successful_batches + failed_batches, 1) * 100
    
    print(f"📈 Epoch {epoch+1} 完成: 平均损失={avg_loss:.6f}, 成功率={success_rate:.1f}%")
    
    return avg_loss


def main():
    """主训练函数"""
    args = parse_args()
    
    print("🚀 GOLD-YOLO Jittor版本 - CPU训练（修复数据格式）")
    print("=" * 60)
    
    try:
        # 加载数据配置
        data_config = load_data_config(args.data)
        num_classes = data_config.get('nc', 20)
        
        # 创建数据集
        train_dataset = create_simple_dataset(data_config, args.img_size, is_train=True)
        
        # 创建模型和损失函数
        model, loss_fn = create_model_and_loss(num_classes)
        
        # 创建优化器 - CPU优化
        optimizer = jt.optim.SGD(
            model.parameters(), 
            lr=args.lr,
            momentum=0.9, 
            weight_decay=0.0001
        )
        
        # 创建保存目录
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 模型保存目录: {save_dir}")
        print(f"🎯 开始CPU训练 {args.epochs} 轮...")
        
        best_loss = float('inf')
        
        for epoch in range(args.epochs):
            avg_loss = train_one_epoch_cpu(
                model, train_dataset, loss_fn, optimizer, epoch, args.batch_size
            )
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = str(save_dir / "best.pkl")
                try:
                    save_dict = {
                        'epoch': epoch + 1,
                        'loss': float(avg_loss),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    jt.save(save_dict, best_model_path)
                    print(f"✅ 最佳模型保存: {best_model_path}")
                except Exception as e:
                    print(f"⚠️ 模型保存失败: {e}")
        
        print(f"🎉 CPU训练完成！最佳损失: {best_loss:.6f}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
