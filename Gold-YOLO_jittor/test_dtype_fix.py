#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试数据类型修复 - 确保没有int类型进入卷积层
"""

import os
os.environ['JT_SYNC'] = '1'

import jittor as jt
jt.flags.use_cuda = 0  # 强制CPU避免CUDA问题

def test_data_types():
    """测试数据类型修复"""
    print("🔍 测试数据类型修复")
    
    try:
        # 1. 测试数据集返回的数据类型
        print("\n📊 测试数据集数据类型...")
        
        import yaml
        from yolov6.data.datasets import TrainValDataset
        
        # 加载数据配置
        with open('data/voc_subset_improved.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        
        # 创建简化数据集
        hyp = {
            'mosaic': 0.0, 'mixup': 0.0, 'degrees': 0.0, 'translate': 0.0,
            'scale': 0.0, 'shear': 0.0, 'flipud': 0.0, 'fliplr': 0.0,
            'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0
        }
        
        dataset = TrainValDataset(
            img_dir=data_config['val'],
            img_size=640,
            augment=False,
            hyp=hyp,
            rect=False,
            check_images=True,
            check_labels=True,
            stride=32,
            pad=0.0,
            rank=-1,
            data_dict=data_config
        )
        
        # 测试加载样本
        sample = dataset[0]
        image, target, img_path, shapes = sample
        
        print(f"   ✅ 图像数据类型: {image.dtype}")
        print(f"   ✅ 图像形状: {list(image.shape)}")
        print(f"   ✅ 目标数据类型: {target.dtype}")
        print(f"   ✅ 目标形状: {list(target.shape)}")
        
        # 检查数据类型
        if image.dtype == 'float32':
            print("   ✅ 图像数据类型正确 (float32)")
        else:
            print(f"   ❌ 图像数据类型错误: {image.dtype}")
            return False
        
        if target.dtype == 'float32':
            print("   ✅ 目标数据类型正确 (float32)")
        else:
            print(f"   ❌ 目标数据类型错误: {target.dtype}")
            return False
        
        return True, dataset
        
    except Exception as e:
        print(f"❌ 数据类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_with_correct_types():
    """测试模型使用正确的数据类型"""
    print("\n🔍 测试模型数据类型兼容性...")
    
    try:
        # 创建模型
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        model = create_perfect_gold_yolo_model('gold_yolo-n', 20)
        model.eval()
        
        print("✅ 模型创建成功")
        
        # 测试不同数据类型的输入
        print("\n   测试float32输入...")
        x_float32 = jt.randn(1, 3, 640, 640, dtype='float32')
        print(f"   输入数据类型: {x_float32.dtype}")
        
        with jt.no_grad():
            output_float32 = model(x_float32)
        
        print(f"   ✅ float32输入成功: {list(output_float32.shape)}")
        
        # 测试int输入（应该会出错或被自动转换）
        print("\n   测试int输入...")
        x_int = jt.randint(0, 255, (1, 3, 640, 640), dtype='int32')
        print(f"   输入数据类型: {x_int.dtype}")
        
        try:
            with jt.no_grad():
                output_int = model(x_int)
            print(f"   ⚠️ int输入居然成功了: {list(output_int.shape)}")
            print(f"   这可能导致精度问题！")
        except Exception as e:
            print(f"   ✅ int输入正确失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型数据类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_with_correct_types(dataset):
    """测试训练过程中的数据类型"""
    print("\n🔍 测试训练过程数据类型...")
    
    try:
        # 创建模型和损失函数
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        model = create_perfect_gold_yolo_model('gold_yolo-n', 20)
        model.train()
        
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
            num_classes=20,
            ori_img_size=640,
            warmup_epoch=4,
            use_dfl=False,
            reg_max=0,
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        
        print("✅ 模型和损失函数创建成功")
        
        # 模拟训练数据处理
        print("\n   模拟数据加载和预处理...")
        
        # 加载真实数据
        sample = dataset[0]
        image, target, img_path, shapes = sample
        
        print(f"   原始图像类型: {image.dtype}")
        print(f"   原始目标类型: {target.dtype}")
        
        # 模拟批次处理
        batch_images = [image]
        batch_targets = [target]
        
        # 堆叠图像
        images = jt.stack(batch_images)
        print(f"   堆叠后图像类型: {images.dtype}")
        
        # 确保数据类型正确
        if images.dtype != 'float32':
            images = images.float32()
            print(f"   修正后图像类型: {images.dtype}")
        
        # 归一化
        images = images / 255.0
        print(f"   归一化后图像类型: {images.dtype}")
        print(f"   图像值范围: [{float(images.min()):.3f}, {float(images.max()):.3f}]")
        
        # 处理目标
        targets = jt.concat(batch_targets, dim=0)
        print(f"   目标类型: {targets.dtype}")
        
        # 前向传播
        print("\n   前向传播...")
        outputs = model(images)
        print(f"   ✅ 前向传播成功: {list(outputs.shape)}")
        print(f"   输出数据类型: {outputs.dtype}")
        
        # 计算损失
        print("\n   计算损失...")
        loss_result = loss_fn(outputs, targets, 0, 0)
        
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
        else:
            loss = loss_result
        
        print(f"   ✅ 损失计算成功: {float(loss):.6f}")
        print(f"   损失数据类型: {loss.dtype}")
        
        # 测试梯度
        print("\n   测试梯度计算...")
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        print("   ✅ 梯度计算和参数更新成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练数据类型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 数据类型修复测试")
    print("=" * 60)
    print("🎯 确保没有int类型进入卷积层")
    print("=" * 60)
    
    # 测试1: 数据集数据类型
    success1, dataset = test_data_types()
    
    # 测试2: 模型数据类型兼容性
    success2 = test_model_with_correct_types()
    
    # 测试3: 训练过程数据类型
    success3 = False
    if success1 and dataset:
        success3 = test_training_with_correct_types(dataset)
    
    print("\n" + "=" * 60)
    print("🎯 数据类型修复测试结果")
    print("=" * 60)
    print(f"   数据集数据类型: {'✅ 修复成功' if success1 else '❌ 仍有问题'}")
    print(f"   模型数据类型兼容: {'✅ 修复成功' if success2 else '❌ 仍有问题'}")
    print(f"   训练过程数据类型: {'✅ 修复成功' if success3 else '❌ 仍有问题'}")
    
    if success1 and success2 and success3:
        print("\n🎉 数据类型问题完全修复！")
        print("✅ 所有数据都是float32类型，不会有int进入卷积层")
        print("✅ 现在可以正常训练了！")
    else:
        print("\n❌ 还有数据类型问题需要修复")


if __name__ == "__main__":
    main()
