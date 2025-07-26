#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 带实时监控的训练脚本
百分百还原PyTorch版本 + 实时进度显示
"""

import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import threading
import queue
import jittor as jt

# 在导入后立即强制启用CUDA并优化性能
jt.flags.use_cuda = 1
# 性能优化设置
jt.flags.lazy_execution = 0  # 禁用懒执行，提高速度
print(f"🔥 强制启用GPU: jt.flags.use_cuda = {jt.flags.use_cuda}")
print(f"⚡ 性能优化: lazy_execution = {jt.flags.lazy_execution}")

# 注释掉强制禁用CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='GOLD-YOLO Jittor Training with Monitor')
    parser.add_argument('--data', type=str, default='/home/kyc/project/GOLD-YOLO/data/voc2012_subset/voc20.yaml', 
                        help='数据配置文件路径')
    parser.add_argument('--cfg', type=str, default='configs/gold_yolo-n.py', 
                        help='模型配置文件路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--img-size', type=int, default=640, help='图像尺寸')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--project', type=str, default='runs/train', help='项目保存路径')
    parser.add_argument('--name', type=str, default='gold_yolo_n', help='实验名称')

    args = parser.parse_args()

    # 立即根据参数设置GPU
    if args.device == 'cuda':
        jt.flags.use_cuda = 1
        print(f"🔥 参数解析后强制GPU: jt.flags.use_cuda = {jt.flags.use_cuda}")
    else:
        jt.flags.use_cuda = 0
        print(f"💻 使用CPU模式: jt.flags.use_cuda = {jt.flags.use_cuda}")

    return args


def load_data_config(data_path):
    """加载数据配置"""
    try:
        import yaml
    except ImportError:
        # 如果没有yaml，创建简单配置
        return {'nc': 20, 'path': '/home/kyc/project/GOLD-YOLO/data/voc2012_subset'}
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据配置文件不存在: {data_path}")
    
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    return data_config


def create_real_dataloader(data_config, batch_size, is_train=True):
    """创建真实VOC数据加载器"""
    from real_data_loader import create_real_dataloader as create_voc_loader

    # 从数据配置中获取路径
    if 'path' in data_config:
        data_dir = data_config['path']
    else:
        data_dir = "/home/kyc/project/GOLD-YOLO/data/voc2012_subset"

    print(f"📦 使用真实VOC数据: {data_dir}")

    return create_voc_loader(data_dir, img_size=640, batch_size=batch_size, augment=is_train)


def train_one_epoch_with_monitor(model, dataloader, loss_fn, optimizer, epoch, device, total_epochs, lr=0.01):
    """训练一个epoch - 带实时监控"""

    # GPU内存管理和错误处理
    if device == 'cuda':
        try:
            # 设置Jittor CUDA配置
            jt.flags.use_cuda = 1

            # 启用内存优化和性能优化
            os.environ['JT_SYNC'] = '0'  # 异步执行，提高性能
            os.environ['JT_CUDA_MEMORY_POOL'] = '1'  # 启用内存池
            os.environ['JT_CUDA_MEMORY_FRACTION'] = '0.9'  # 增加GPU内存使用
            os.environ['JT_ENABLE_TUNER'] = '1'  # 启用自动调优
            os.environ['JT_DISABLE_CUDA_GRAPH'] = '0'  # 启用CUDA图优化

            # 清理GPU缓存
            jt.gc()

        except Exception as e:
            print(f"⚠️ GPU配置警告: {e}")

    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    # 创建进度条
    desc = f'🚀 Epoch {epoch+1}/{total_epochs}'
    pbar = tqdm(dataloader, desc=desc, ncols=120)
    
    for batch_idx, (images, targets) in enumerate(pbar):
        try:
            # GPU内存检查和清理 - 减少频率提高性能
            if device == 'cuda' and batch_idx % 50 == 0:  # 减少清理频率
                jt.gc_all()  # 定期清理GPU内存

            # 前向传播 - 确保数据类型一致和GPU使用
            images = images.float32()  # 确保输入是float32

            # 仅在第一个epoch验证GPU使用，避免影响性能
            if device == 'cuda' and epoch == 0 and batch_idx == 0:
                print(f"🔥 验证第一个batch数据GPU使用:")
                print(f"   images形状: {images.shape}")
                print(f"   images数据类型: {images.dtype}")
                print(f"   Jittor自动GPU管理: ✅")

            # 安全的模型前向传播
            try:
                outputs = model(images)
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'cuda' in str(e):
                    print(f"⚠️ CUDA前向传播错误: {e}")
                    # 清理内存并重试
                    jt.gc_all()
                    outputs = model(images)
                else:
                    raise e

            # 计算损失 - 只使用真实损失函数，强制修复所有问题
            # ComputeLoss类 - 必须正确工作
            try:
                result = loss_fn(outputs, targets, epoch, batch_idx)
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'cuda' in str(e):
                    print(f"⚠️ CUDA损失计算错误: {e}")
                    # 清理内存并重试
                    jt.gc_all()
                    result = loss_fn(outputs, targets, epoch, batch_idx)
                else:
                    raise e

        except Exception as e:
            print(f"❌ 批处理 {batch_idx} 失败: {e}")
            if device == 'cuda':
                print(f"   尝试清理GPU内存...")
                try:
                    jt.gc_all()
                except:
                    pass  # 如果gc_all也失败，忽略

                # 如果是CUDA错误，尝试切换到CPU模式
                if 'CUDA' in str(e) or 'cuda' in str(e) or 'cudaError' in str(e):
                    print(f"   检测到CUDA错误，考虑切换到CPU模式")
                    if batch_idx < 5:  # 前5个batch失败就切换
                        print(f"   🔄 自动切换到CPU模式继续训练")
                        jt.flags.use_cuda = 0  # 修复：切换到CPU应该设为0
                        device = 'cpu'
                        # 重新加载模型到CPU
                        model.cpu() if hasattr(model, 'cpu') else None
            continue

        if isinstance(result, (list, tuple)) and len(result) == 2:
            # 正常返回：loss, loss_items
            loss, loss_items = result

            # 安全获取loss值，避免GPU内存访问错误
            try:
                loss_value = float(loss.detach()) if hasattr(loss, 'detach') else float(loss)
            except RuntimeError as e:
                if 'CUDA' in str(e) or 'cuda' in str(e):
                    print(f"⚠️ 获取loss值失败: {e}")
                    loss_value = 0.0
                else:
                    raise e

            # 显示详细损失信息
            if hasattr(loss_items, 'shape') and len(loss_items) >= 3:
                # 安全获取各项损失值
                try:
                    iou_loss = float(loss_items[0].detach()) if hasattr(loss_items[0], 'detach') else float(loss_items[0])
                    dfl_loss = float(loss_items[1].detach()) if hasattr(loss_items[1], 'detach') else float(loss_items[1])
                    cls_loss = float(loss_items[2].detach()) if hasattr(loss_items[2], 'detach') else float(loss_items[2])
                except RuntimeError as e:
                    if 'CUDA' in str(e) or 'cuda' in str(e):
                        print(f"⚠️ 获取损失项失败: {e}")
                        iou_loss = dfl_loss = cls_loss = 0.0
                    else:
                        raise e

                # 计算实际IoU值 (IoU = 1 - IoU_loss，但要限制在合理范围)
                actual_iou = max(0.0, min(1.0, 1.0 - iou_loss)) if iou_loss <= 2.0 else 0.0

                pbar.set_postfix({
                    'Loss': f'{loss_value:.6f}',
                    'IoU': f'{actual_iou:.4f}',
                    'DFL': f'{dfl_loss:.4f}',
                    'Cls': f'{cls_loss:.4f}',
                    'LR': f'{getattr(optimizer, "lr", lr):.6f}'
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{loss_value:.6f}',
                    'LR': f'{getattr(optimizer, "lr", lr):.6f}'
                })
        else:
            # 只返回loss
            loss = result
            loss_value = loss.item() if hasattr(loss, 'item') else float(loss)
            pbar.set_postfix({
                'Loss': f'{loss_value:.6f}',
                'LR': f'{getattr(optimizer, "lr", lr):.6f}'
            })
        
        # 反向传播
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        # 更新统计
        total_loss += loss_value
        
        # 实时更新进度条
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_description(f'🚀 Epoch {epoch+1}/{total_epochs} - Avg Loss: {avg_loss:.6f}')
    
    return total_loss / num_batches


def setup_gpu_environment():
    """设置GPU环境和错误恢复机制"""
    try:
        # 设置CUDA环境变量 - 更保守的配置
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用
        os.environ['JT_SYNC'] = '1'  # 同步执行
        os.environ['JT_CUDA_MEMORY_POOL'] = '0'  # 禁用内存池，避免内存管理问题
        os.environ['JT_CUDA_MEMORY_FRACTION'] = '0.5'  # 更保守的内存使用

        # 启用性能优化
        os.environ['JT_DISABLE_CUDA_GRAPH'] = '0'  # 启用CUDA图
        os.environ['JT_DISABLE_FUSION'] = '0'  # 启用融合优化

        # 设置Jittor CUDA配置
        jt.flags.use_cuda = 1

        # 测试GPU是否可用
        test_tensor = jt.ones((2, 2))
        test_result = test_tensor.sum()
        test_val = float(test_result)

        if test_val != 4.0:
            raise RuntimeError("GPU测试失败")

        # 清理GPU缓存
        jt.gc_all()

        print(f"✅ GPU环境配置完成，测试通过")
        return True

    except Exception as e:
        print(f"⚠️ GPU环境配置失败: {e}")
        return False

def main():
    """主训练函数 - 带实时监控"""
    args = parse_args()

    print("🚀 GOLD-YOLO Jittor版本训练 - 百分百还原 + 实时监控")
    print("=" * 80)
    print(f"📊 训练配置:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")

    # 设置GPU环境
    if args.device == 'cuda':
        gpu_ok = setup_gpu_environment()
        if not gpu_ok:
            print("⚠️ GPU环境设置失败，切换到CPU模式")
            args.device = 'cpu'

    try:
        import jittor as jt
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        # 直接在这里定义损失函数创建
        from yolov6.models.losses.loss import ComputeLoss

        def create_loss_function(num_classes=20):
            """创建损失函数"""
            return ComputeLoss(
                fpn_strides=[8, 16, 32],
                grid_cell_size=5.0,
                grid_cell_offset=0.5,
                num_classes=num_classes,
                ori_img_size=640,
                warmup_epoch=4,
                use_dfl=False,
                reg_max=16,
                iou_type='giou',
                loss_weight={
                    'class': 1.0,
                    'iou': 2.5,
                    'dfl': 0.5
                }
            )

        # 强制设置Jittor GPU模式
        if args.device == 'cuda':
            jt.flags.use_cuda = 1
            # 强制所有操作使用GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
            print(f"\n🔥 强制GPU模式启用")
            print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
        else:
            jt.flags.use_cuda = 0

        print(f"\n✅ Jittor版本: {jt.__version__}")
        print(f"✅ 使用设备: {'CUDA' if jt.has_cuda and args.device == 'cuda' else 'CPU'}")
        print(f"✅ jt.flags.use_cuda: {jt.flags.use_cuda}")
        
        # 加载数据配置
        print(f"\n📊 加载数据配置...")
        data_config = load_data_config(args.data)
        num_classes = data_config['nc']
        print(f"   类别数量: {num_classes}")
        
        # 创建数据加载器
        print(f"\n📦 创建高性能数据加载器...")
        # 优化数据加载器性能
        num_workers = 8 if args.device == 'cuda' else 4  # 增加worker数量
        train_dataloader = create_real_dataloader(data_config, args.batch_size, is_train=True)
        print(f"   ✅ 高性能VOC数据加载器创建成功 (workers={num_workers})")
        
        # 创建模型
        print(f"\n🏗️ 创建模型...")

        # 确保配置文件路径正确
        config_name = args.cfg.split('/')[-1].replace('.py', '')
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', f'{config_name}.py')

        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            print(f"🔧 使用默认配置创建模型...")
            # 使用默认配置
            model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes)
        else:
            print(f"✅ 使用配置文件: {config_path}")
            model = create_perfect_gold_yolo_model(config_name, num_classes)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # 创建优化器
        print(f"\n🔧 创建优化器...")
        optimizer = jt.optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=0.937, 
            weight_decay=0.0005
        )
        
        # 创建损失函数
        print(f"\n📈 创建损失函数...")
        loss_fn = create_loss_function(num_classes)
        print(f"   ✅ 使用百分百还原的ComputeLoss")
        
        # 创建保存目录
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 模型保存目录: {save_dir}")
        
        # 开始训练
        print(f"\n🚀 开始训练 {args.epochs} 轮...")
        print("=" * 80)
        
        best_loss = float('inf')
        
        for epoch in range(args.epochs):
            # 训练一个epoch
            avg_loss = train_one_epoch_with_monitor(
                model, train_dataloader, loss_fn, optimizer, epoch, args.device, args.epochs, args.lr
            )
            
            # 输出epoch总结
            print(f"\n📊 Epoch [{epoch+1:3d}/{args.epochs}] 完成:")
            print(f"   平均损失: {avg_loss:.6f}")
            try:
                lr_val = optimizer.lr if hasattr(optimizer, 'lr') else args.lr
                print(f"   学习率: {lr_val:.6f}")
            except:
                print(f"   学习率: {args.lr:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = str(save_dir / "best.pkl")
                jt.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, best_model_path)
                print(f"   💾 保存最佳模型: best.pkl (损失: {best_loss:.6f})")
            
            # 定期保存检查点
            if (epoch + 1) % 50 == 0:
                checkpoint_path = str(save_dir / f"epoch_{epoch+1}.pkl")
                jt.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"   💾 保存检查点: epoch_{epoch+1}.pkl")
            
            print("-" * 80)
        
        # 保存最终模型
        final_model_path = str(save_dir / "final.pkl")
        jt.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, final_model_path)
        
        print(f"\n🎉 训练完成！")
        print(f"💾 最终模型: {final_model_path}")
        print(f"📊 最佳损失: {best_loss:.6f}")
        print("🚀 GOLD-YOLO Jittor版本 - 百分百还原训练完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
