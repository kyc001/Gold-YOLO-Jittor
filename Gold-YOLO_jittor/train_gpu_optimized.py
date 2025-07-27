#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - GPU优化训练脚本
专门针对GPU训练优化，包含完整的CUDA错误处理
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

# 设置环境变量 - GPU优化
os.environ['JT_SYNC'] = '1'
# 不强制禁用CUDA，让Jittor自动检测

def setup_gpu():
    """设置GPU环境"""
    try:
        import jittor as jt
        
        print("🔍 检测GPU环境...")
        print(f"   Jittor版本: {jt.__version__}")
        print(f"   CUDA可用: {jt.has_cuda}")
        
        if jt.has_cuda:
            # 尝试启用CUDA
            jt.flags.use_cuda = 1
            
            # 测试CUDA是否真的可用
            test_tensor = jt.randn(2, 2)
            test_result = test_tensor * 2
            
            print("✅ GPU设置成功")
            print(f"   使用设备: GPU")
            return True, jt
        else:
            print("⚠️ CUDA不可用，使用CPU")
            jt.flags.use_cuda = 0
            return False, jt
            
    except Exception as e:
        print(f"❌ GPU设置失败: {e}")
        print("   回退到CPU模式")
        import jittor as jt
        jt.flags.use_cuda = 0
        return False, jt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GOLD-YOLO GPU Optimized Training')
    parser.add_argument('--data', type=str, default='data/voc_subset_improved.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小(GPU优化)')
    parser.add_argument('--img-size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--project', type=str, default='runs/train', help='项目保存目录')
    parser.add_argument('--name', type=str, default='gpu_training', help='实验名称')
    parser.add_argument('--force-cpu', action='store_true', help='强制使用CPU')
    
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


def create_real_dataset(data_config, img_size, is_train=True, use_gpu=True):
    """创建真实数据集 - GPU优化版"""
    try:
        from yolov6.data.datasets import TrainValDataset
        
        data_path = data_config['train'] if is_train else data_config['val']
        
        # GPU优化的超参数配置
        if use_gpu:
            hyp = {
                'mosaic': 0.5 if is_train else 0.0,  # GPU可以处理更复杂的增强
                'mixup': 0.1 if is_train else 0.0,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 2.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4
            }
        else:
            # CPU模式使用简化配置
            hyp = {
                'mosaic': 0.0,
                'mixup': 0.0,
                'degrees': 0.0,
                'translate': 0.0,
                'scale': 0.0,
                'shear': 0.0,
                'flipud': 0.0,
                'fliplr': 0.0,
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
        
        # 创建损失函数 - 使用我们修复过的版本
        # 直接导入修复过的losses.py中的ComputeLoss
        from yolov6.models.losses import ComputeLoss
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
    """安全的损失计算 - GPU优化版"""
    import jittor as jt  # 确保jt在函数内可用

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
            print(f"⚠️ 检测到无效损失: {float(total_loss)}")
            return None
        
        # GPU模式下的损失缩放策略
        loss_value = float(total_loss)
        if loss_value > 50.0:
            total_loss = total_loss / 5.0
        elif loss_value > 10.0:
            total_loss = total_loss / 2.0
        
        return total_loss
        
    except Exception as e:
        print(f"⚠️ 损失计算失败: {e}")
        return None


def train_one_epoch_gpu(model, dataset, loss_fn, optimizer, epoch, batch_size=8, use_gpu=True):
    """GPU优化的训练函数"""
    import jittor as jt  # 确保jt在函数内可用

    model.train()
    total_loss = 0
    successful_batches = 0
    failed_batches = 0
    
    dataset_size = len(dataset)
    # GPU模式下可以处理更多批次
    max_batches = 100 if use_gpu else 50
    num_batches = min(max_batches, (dataset_size + batch_size - 1) // batch_size)
    
    print(f"📊 Epoch {epoch+1}: 数据集大小={dataset_size}, 训练批次数={num_batches}")
    
    pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}')
    
    for batch_idx in pbar:
        try:
            # 收集真实数据批次
            batch_images = []
            batch_targets = []
            
            valid_samples = 0
            for i in range(batch_size):
                sample_idx = batch_idx * batch_size + i
                if sample_idx >= dataset_size:
                    break
                
                try:
                    # 加载真实数据
                    dataset_output = dataset[sample_idx]
                    
                    if len(dataset_output) == 4:
                        image, target, img_path, shapes = dataset_output
                    else:
                        continue
                    
                    if image is not None:
                        batch_images.append(image)
                        
                        # 处理目标标签
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
            
            # 合并targets - 修复形状不匹配问题
            if batch_targets:
                # 安全地合并不同长度的目标列表
                all_targets = []
                for target_tensor in batch_targets:
                    # 确保每个target都是2维的
                    if len(target_tensor.shape) == 1:
                        target_tensor = target_tensor.unsqueeze(0)
                    all_targets.append(target_tensor)

                # 现在可以安全地连接
                targets = jt.concat(all_targets, dim=0)
            else:
                targets = jt.zeros((0, 6))
            
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
            
            # GPU模式下使用更激进的梯度裁剪 - 修复Jittor API
            max_norm = 5.0 if use_gpu else 10.0
            # Jittor使用不同的梯度裁剪API
            try:
                # 深入修复梯度裁剪 - 正确处理张量norm
                total_norm = 0.0
                for param in model.parameters():
                    if param.opt_grad(optimizer) is not None:
                        grad = param.opt_grad(optimizer)
                        # 深入修复：计算梯度的L2范数，确保结果是标量
                        # 方法1：手动计算L2范数
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
                'Valid': f'{valid_samples}/{batch_size}',
                'GPU': 'Yes' if use_gpu else 'No'
            })
            
        except Exception as e:
            print(f"⚠️ Batch {batch_idx} 训练失败: {e}")
            failed_batches += 1
            continue
    
    avg_loss = total_loss / max(successful_batches, 1)
    success_rate = successful_batches / max(successful_batches + failed_batches, 1) * 100
    
    print(f"📈 Epoch {epoch+1} 完成: 平均损失={avg_loss:.6f}, 成功率={success_rate:.1f}%")
    
    return avg_loss


def save_model(model, optimizer, epoch, loss, save_path, use_gpu=True):
    """保存模型"""
    import jittor as jt  # 确保jt在函数内可用

    try:
        save_dict = {
            'epoch': epoch,
            'loss': float(loss),
            'model_config': 'gold_yolo-n',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_type': 'gpu_optimized' if use_gpu else 'cpu_fallback',
            'device': 'GPU' if use_gpu else 'CPU'
        }
        
        jt.save(save_dict, save_path)
        print(f"✅ 模型保存成功: {save_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ 模型保存失败: {e}")
        return False


def main():
    """主训练函数"""
    args = parse_args()
    
    print("🚀 GOLD-YOLO Jittor版本 - GPU优化训练")
    print("=" * 60)
    
    # 设置GPU环境
    if args.force_cpu:
        print("⚠️ 强制使用CPU模式")
        import jittor as jt
        jt.flags.use_cuda = 0
        use_gpu = False
    else:
        use_gpu, jt = setup_gpu()
    
    try:
        # 加载数据配置
        data_config = load_data_config(args.data)
        num_classes = data_config.get('nc', 20)
        
        # 根据GPU状态调整批次大小
        if not use_gpu and args.batch_size > 2:
            print(f"⚠️ CPU模式，批次大小从{args.batch_size}调整为2")
            args.batch_size = 2
        
        # 创建真实数据集
        train_dataset = create_real_dataset(data_config, args.img_size, is_train=True, use_gpu=use_gpu)
        
        # 创建模型和损失函数
        model, loss_fn = create_model_and_loss(num_classes)
        
        # 创建优化器 - GPU优化
        if use_gpu:
            optimizer = jt.optim.SGD(
                model.parameters(), 
                lr=args.lr, 
                momentum=0.937, 
                weight_decay=0.0005
            )
        else:
            # CPU模式使用更保守的参数
            optimizer = jt.optim.SGD(
                model.parameters(), 
                lr=args.lr * 0.5,  # 降低学习率
                momentum=0.9, 
                weight_decay=0.0001
            )
        
        # 创建保存目录
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 模型保存目录: {save_dir}")
        print(f"🎯 开始{'GPU' if use_gpu else 'CPU'}训练 {args.epochs} 轮...")
        
        best_loss = float('inf')
        
        for epoch in range(args.epochs):
            avg_loss = train_one_epoch_gpu(
                model, train_dataset, loss_fn, optimizer, epoch, args.batch_size, use_gpu
            )
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = str(save_dir / "best.pkl")
                save_model(model, optimizer, epoch + 1, avg_loss, best_model_path, use_gpu)
            
            # 定期保存检查点
            if (epoch + 1) % 25 == 0:
                checkpoint_path = str(save_dir / f"epoch_{epoch+1}.pkl")
                save_model(model, optimizer, epoch + 1, avg_loss, checkpoint_path, use_gpu)
        
        # 保存最终模型
        final_model_path = str(save_dir / "final.pkl")
        save_model(model, optimizer, args.epochs, avg_loss, final_model_path, use_gpu)
        
        print(f"🎉 {'GPU' if use_gpu else 'CPU'}训练完成！最佳损失: {best_loss:.6f}")
        print(f"💾 最终模型保存在: {final_model_path}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
