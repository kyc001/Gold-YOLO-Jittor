#!/usr/bin/env python3
"""
单张图片过拟合训练自检
这是验证模型是否成功的最可靠方法
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import time
import matplotlib.pyplot as plt

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.pytorch_aligned_losses import ComputeLoss

def pytorch_exact_initialization(model):
    """完全照抄PyTorch版本的初始化"""
    for module in model.modules():
        if hasattr(module, 'initialize_biases'):
            module.initialize_biases()
            break
    return model

def single_image_overfit_test():
    """单张图片过拟合训练自检"""
    print(f"🔥 单张图片过拟合训练自检")
    print("=" * 80)
    print(f"这是验证模型是否成功的最可靠方法！")
    print("=" * 80)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    print(f"📁 数据文件:")
    print(f"   图像: {img_path}")
    print(f"   标注: {label_file}")
    
    # 读取标注
    annotations = []
    with open(label_file, 'r') as f:
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
                    annotations.append([cls_id, x_center, y_center, width, height])
    
    print(f"   标注数量: {len(annotations)}个目标")
    
    # 读取图像
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    # 准备标签
    targets = []
    for ann in annotations:
        cls_id, x_center, y_center, width, height = ann
        targets.append([0, cls_id, x_center, y_center, width, height])
    targets_tensor = jt.array(targets, dtype=jt.float32).unsqueeze(0)
    
    print(f"📊 数据准备:")
    print(f"   图像张量: {img_tensor.shape}")
    print(f"   标签张量: {targets_tensor.shape}")
    
    # 创建模型
    print(f"\n🎯 创建模型:")
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    model.train()  # 训练模式
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 创建损失函数 - 100%对齐PyTorch版本
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=500,
        warmup_epoch=0,  # 不使用warmup
        use_dfl=False,   # 对齐PyTorch版本
        reg_max=0,       # 对齐PyTorch版本
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    
    # 创建优化器 - 使用较大学习率进行过拟合
    optimizer = jt.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    
    print(f"\n🔥 开始过拟合训练:")
    print(f"   学习率: 0.1 (较大，便于快速过拟合)")
    print(f"   优化器: SGD")
    print(f"   目标: 损失快速下降到接近0")
    
    # 记录训练过程
    loss_history = []
    loss_items_history = []
    
    # 训练循环
    num_epochs = 100
    print_interval = 10
    
    print(f"\n📈 训练进度:")
    print(f"   总轮数: {num_epochs}")
    print(f"   打印间隔: 每{print_interval}轮")
    print("-" * 80)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
        
        # 反向传播
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        # 记录损失
        loss_value = float(loss.data.item())
        loss_items_values = [float(item.data.item()) for item in loss_items]
        
        loss_history.append(loss_value)
        loss_items_history.append(loss_items_values)
        
        # 打印进度
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"   轮次 {epoch+1:3d}/{num_epochs}: 损失={loss_value:.6f}, "
                  f"损失项={[f'{x:.4f}' for x in loss_items_values]}, "
                  f"用时={elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    print("-" * 80)
    print(f"✅ 训练完成! 总用时: {total_time:.1f}s")
    
    # 分析训练结果
    print(f"\n📊 训练结果分析:")
    
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"   初始损失: {initial_loss:.6f}")
    print(f"   最终损失: {final_loss:.6f}")
    print(f"   损失下降: {loss_reduction:.1f}%")
    
    # 判断过拟合是否成功
    success_criteria = [
        ("损失下降超过90%", loss_reduction > 90),
        ("最终损失小于0.1", final_loss < 0.1),
        ("损失持续下降", loss_history[-1] < loss_history[len(loss_history)//2]),
    ]
    
    print(f"\n🎯 过拟合成功标准:")
    success_count = 0
    for criterion, passed in success_criteria:
        status = "✅" if passed else "❌"
        print(f"   {status} {criterion}")
        if passed:
            success_count += 1
    
    overall_success = success_count >= 2
    print(f"\n{'🎉' if overall_success else '❌'} 总体评估: "
          f"{'过拟合成功！模型工作正常' if overall_success else '过拟合失败，模型可能有问题'}")
    
    # 保存训练曲线
    save_dir = Path("runs/single_image_overfit")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(loss_history)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 分别绘制各项损失
    loss_items_array = np.array(loss_items_history)
    loss_names = ['IoU Loss', 'DFL Loss', 'Class Loss']
    
    for i in range(min(3, loss_items_array.shape[1])):
        plt.subplot(2, 2, i+2)
        plt.plot(loss_items_array[:, i])
        plt.title(loss_names[i])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    curve_path = save_dir / 'loss_curves.png'
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 训练曲线已保存: {curve_path}")
    
    # 保存训练日志
    log_path = save_dir / 'training_log.txt'
    with open(log_path, 'w') as f:
        f.write("# 单张图片过拟合训练日志\n\n")
        f.write(f"模型参数: {total_params:,} ({total_params/1e6:.2f}M)\n")
        f.write(f"训练轮数: {num_epochs}\n")
        f.write(f"学习率: 0.1\n")
        f.write(f"初始损失: {initial_loss:.6f}\n")
        f.write(f"最终损失: {final_loss:.6f}\n")
        f.write(f"损失下降: {loss_reduction:.1f}%\n")
        f.write(f"训练时间: {total_time:.1f}s\n")
        f.write(f"过拟合成功: {'是' if overall_success else '否'}\n\n")
        
        f.write("详细损失记录:\n")
        for i, (loss_val, loss_items_val) in enumerate(zip(loss_history, loss_items_history)):
            f.write(f"轮次{i+1:3d}: {loss_val:.6f} {loss_items_val}\n")
    
    print(f"📝 训练日志已保存: {log_path}")
    
    return {
        'success': overall_success,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'loss_history': loss_history,
        'total_time': total_time
    }

def main():
    print("🔥 单张图片过拟合训练自检")
    print("=" * 80)
    
    try:
        result = single_image_overfit_test()
        
        if result and result['success']:
            print(f"\n🎉🎉🎉 模型验证成功！")
            print(f"   损失从 {result['initial_loss']:.6f} 下降到 {result['final_loss']:.6f}")
            print(f"   下降幅度: {result['loss_reduction']:.1f}%")
            print(f"   训练时间: {result['total_time']:.1f}s")
            print(f"\n✅ 模型工作正常，可以进行正式训练！")
        else:
            print(f"\n❌ 模型验证失败！")
            print(f"   需要检查模型结构或损失函数")
            
    except Exception as e:
        print(f"\n❌ 验证过程异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
