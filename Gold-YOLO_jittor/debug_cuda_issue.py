#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
深度调试CUDA问题，找出具体在哪个步骤出错
"""

import os
import sys

# 设置环境变量
os.environ['JT_SYNC'] = '1'

import jittor as jt

def test_basic_cuda():
    """测试基本CUDA功能"""
    print("🔍 测试基本CUDA功能...")
    
    try:
        print(f"   Jittor版本: {jt.__version__}")
        print(f"   CUDA可用: {jt.has_cuda}")
        
        # 启用CUDA
        jt.flags.use_cuda = 1
        print("   ✅ CUDA启用成功")
        
        # 创建简单张量
        x = jt.randn(2, 3)
        print(f"   ✅ 创建张量成功: {x.shape}")
        
        # 简单运算
        y = x * 2
        print(f"   ✅ 张量运算成功: {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 基本CUDA测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        
        print("   正在创建模型...")
        model = create_perfect_gold_yolo_model('gold_yolo-n', 20)
        print("   ✅ 模型创建成功")
        
        # 测试前向传播
        print("   测试前向传播...")
        x = jt.randn(1, 3, 640, 640)
        print(f"   输入张量: {x.shape}")
        
        with jt.no_grad():
            output = model(x)
        print(f"   ✅ 前向传播成功: {output.shape}")
        
        return True, model
        
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loss_function():
    """测试损失函数"""
    print("\n🔍 测试损失函数...")
    
    try:
        from yolov6.models.losses import ComputeLoss
        
        print("   创建损失函数...")
        loss_fn = ComputeLoss(
            fpn_strides=[8, 16, 32],
            grid_cell_size=5.0,
            grid_cell_offset=0.5,
            num_classes=20,
            ori_img_size=640,
            warmup_epoch=4,
            use_dfl=False,
            reg_max=0,
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        print("   ✅ 损失函数创建成功")
        
        # 测试损失计算
        print("   测试损失计算...")
        outputs = jt.randn(2, 8400, 25)
        targets = jt.array([
            [0, 5, 0.5, 0.5, 0.2, 0.2],
            [1, 3, 0.3, 0.3, 0.15, 0.15]
        ]).float32()
        
        loss_result = loss_fn(outputs, targets, 0, 0)
        print(f"   ✅ 损失计算成功: {loss_result}")
        
        return True, loss_fn
        
    except Exception as e:
        print(f"   ❌ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        import yaml
        from yolov6.data.datasets import TrainValDataset
        
        # 加载数据配置
        with open('data/voc_subset_improved.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        
        print("   创建数据集...")
        
        # 创建基本的超参数配置
        hyp = {
            'mosaic': 0.0,  # 禁用mosaic避免复杂性
            'mixup': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4
        }
        
        dataset = TrainValDataset(
            img_dir=data_config['val'],
            img_size=640,
            augment=False,  # 禁用增强
            hyp=hyp,
            rect=False,
            check_images=True,
            check_labels=True,
            stride=32,
            pad=0.0,
            rank=-1,
            data_dict=data_config
        )
        
        print(f"   ✅ 数据集创建成功: {len(dataset)} 样本")
        
        # 测试加载一个样本
        print("   测试样本加载...")
        sample = dataset[0]
        
        if len(sample) == 4:
            image, target, img_path, shapes = sample
            print(f"   ✅ 样本加载成功: 图像{image.shape}, 目标{len(target)}个")
        else:
            print(f"   ⚠️ 样本格式异常: {len(sample)} 个值")
        
        return True, dataset
        
    except Exception as e:
        print(f"   ❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_step(model, loss_fn, dataset):
    """测试完整训练步骤"""
    print("\n🔍 测试完整训练步骤...")
    
    try:
        # 创建优化器
        print("   创建优化器...")
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
        print("   ✅ 优化器创建成功")
        
        # 设置训练模式
        model.train()
        
        # 准备一个batch的数据
        print("   准备训练数据...")
        batch_images = []
        batch_targets = []
        
        for i in range(2):  # 小batch
            try:
                sample = dataset[i]
                if len(sample) == 4:
                    image, target, img_path, shapes = sample
                    
                    if image is not None:
                        batch_images.append(image)
                        
                        if target is not None and len(target) > 0:
                            target_with_batch = jt.concat([
                                jt.full((target.shape[0], 1), len(batch_images)-1),
                                target
                            ], dim=1)
                            batch_targets.append(target_with_batch)
            except Exception as e:
                print(f"     样本{i}加载失败: {e}")
                continue
        
        if not batch_images:
            print("   ❌ 没有有效的训练数据")
            return False
        
        # 堆叠数据
        images = jt.stack(batch_images)
        targets = jt.concat(batch_targets, dim=0) if batch_targets else jt.zeros((0, 6))
        
        print(f"   数据准备完成: 图像{images.shape}, 目标{targets.shape}")
        
        # 前向传播
        print("   执行前向传播...")
        outputs = model(images)
        print(f"   ✅ 前向传播成功: {outputs.shape}")
        
        # 计算损失
        print("   计算损失...")
        loss_result = loss_fn(outputs, targets, 0, 0)
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
        else:
            loss = loss_result
        print(f"   ✅ 损失计算成功: {float(loss)}")
        
        # 反向传播
        print("   执行反向传播...")
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        print("   ✅ 反向传播成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主调试函数"""
    print("🚀 深度调试CUDA问题")
    print("=" * 60)
    
    # 测试1: 基本CUDA功能
    print("测试1: 基本CUDA功能")
    print("-" * 30)
    cuda_ok = test_basic_cuda()
    
    if not cuda_ok:
        print("❌ 基本CUDA功能失败，停止测试")
        return
    
    # 测试2: 模型创建
    print("\n测试2: 模型创建")
    print("-" * 30)
    model_ok, model = test_model_creation()
    
    if not model_ok:
        print("❌ 模型创建失败，停止测试")
        return
    
    # 测试3: 损失函数
    print("\n测试3: 损失函数")
    print("-" * 30)
    loss_ok, loss_fn = test_loss_function()
    
    if not loss_ok:
        print("❌ 损失函数失败，停止测试")
        return
    
    # 测试4: 数据加载
    print("\n测试4: 数据加载")
    print("-" * 30)
    data_ok, dataset = test_data_loading()
    
    if not data_ok:
        print("❌ 数据加载失败，停止测试")
        return
    
    # 测试5: 完整训练步骤
    print("\n测试5: 完整训练步骤")
    print("-" * 30)
    train_ok = test_training_step(model, loss_fn, dataset)
    
    # 总结
    print("\n" + "=" * 60)
    print("🎯 调试结果总结")
    print("=" * 60)
    print(f"   基本CUDA功能: {'✅' if cuda_ok else '❌'}")
    print(f"   模型创建: {'✅' if model_ok else '❌'}")
    print(f"   损失函数: {'✅' if loss_ok else '❌'}")
    print(f"   数据加载: {'✅' if data_ok else '❌'}")
    print(f"   完整训练步骤: {'✅' if train_ok else '❌'}")
    
    if all([cuda_ok, model_ok, loss_ok, data_ok, train_ok]):
        print("\n🎉 所有测试都通过了！训练应该可以正常进行。")
    else:
        print("\n🚨 发现问题，需要进一步调试。")


if __name__ == "__main__":
    main()
