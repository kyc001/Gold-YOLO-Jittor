#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
测试preprocess修复 - 专门测试数据格式问题的修复
"""

import os
os.environ['JT_SYNC'] = '1'

import jittor as jt
jt.flags.use_cuda = 0

def test_preprocess_fix():
    """测试preprocess方法的修复"""
    print("🔍 测试preprocess方法修复")
    
    try:
        # 创建损失函数
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
        
        print("✅ 损失函数创建成功")
        
        # 测试1: 简单情况 - 每个batch有相同数量的目标
        print("\n🔍 测试1: 相同数量目标")
        targets1 = jt.array([
            [0, 5, 0.5, 0.5, 0.2, 0.2],    # batch 0
            [1, 3, 0.3, 0.3, 0.15, 0.15],  # batch 1
        ]).float32()
        
        batch_size = 2
        scale_tensor = jt.full((1, 4), 640.0, dtype='float32')
        
        result1 = loss_fn.preprocess(targets1, batch_size, scale_tensor)
        print(f"   ✅ 结果形状: {list(result1.shape)}")
        print(f"   ✅ 数据类型: {result1.dtype}")
        
        # 测试2: 复杂情况 - 不同数量的目标（这是导致错误的情况）
        print("\n🔍 测试2: 不同数量目标")
        targets2 = jt.array([
            [0, 5, 0.5, 0.5, 0.2, 0.2],    # batch 0: 目标1
            [0, 3, 0.3, 0.3, 0.15, 0.15],  # batch 0: 目标2
            [0, 7, 0.7, 0.7, 0.1, 0.1],    # batch 0: 目标3
            [1, 2, 0.4, 0.4, 0.25, 0.25],  # batch 1: 目标1
        ]).float32()
        
        result2 = loss_fn.preprocess(targets2, batch_size, scale_tensor)
        print(f"   ✅ 结果形状: {list(result2.shape)}")
        print(f"   ✅ batch 0有3个目标，batch 1有1个目标，成功处理！")
        
        # 测试3: 极端情况 - 某个batch没有目标
        print("\n🔍 测试3: 某个batch无目标")
        targets3 = jt.array([
            [0, 5, 0.5, 0.5, 0.2, 0.2],    # 只有batch 0有目标
            [0, 3, 0.3, 0.3, 0.15, 0.15],  # batch 1没有目标
        ]).float32()
        
        result3 = loss_fn.preprocess(targets3, batch_size, scale_tensor)
        print(f"   ✅ 结果形状: {list(result3.shape)}")
        print(f"   ✅ batch 0有2个目标，batch 1无目标，成功处理！")
        
        # 测试4: 空目标
        print("\n🔍 测试4: 完全空目标")
        targets4 = jt.zeros((0, 6)).float32()
        
        result4 = loss_fn.preprocess(targets4, batch_size, scale_tensor)
        print(f"   ✅ 结果形状: {list(result4.shape)}")
        print(f"   ✅ 空目标成功处理！")
        
        return True
        
    except Exception as e:
        print(f"❌ preprocess测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_loss_with_real_data():
    """测试完整的损失计算（使用修复后的preprocess）"""
    print("\n🔍 测试完整损失计算")
    
    try:
        # 创建模型
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        model = create_perfect_gold_yolo_model('gold_yolo-n', 20)
        model.train()
        
        # 创建损失函数
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
        
        # 测试不同数量目标的情况
        batch_size = 3
        images = jt.randn(batch_size, 3, 640, 640)
        
        # 创建复杂的目标分布
        targets = jt.array([
            [0, 5, 0.5, 0.5, 0.2, 0.2],    # batch 0: 4个目标
            [0, 3, 0.3, 0.3, 0.15, 0.15],
            [0, 7, 0.7, 0.7, 0.1, 0.1],
            [0, 2, 0.2, 0.2, 0.1, 0.1],
            [1, 8, 0.6, 0.6, 0.3, 0.3],    # batch 1: 2个目标
            [1, 1, 0.4, 0.4, 0.2, 0.2],
            [2, 9, 0.8, 0.8, 0.15, 0.15],  # batch 2: 1个目标
        ]).float32()
        
        print(f"✅ 测试数据: batch 0有4个目标，batch 1有2个目标，batch 2有1个目标")
        
        # 前向传播
        outputs = model(images)
        print(f"✅ 前向传播成功: {list(outputs.shape)}")
        
        # 计算损失
        print("🔍 开始损失计算...")
        loss_result = loss_fn(outputs, targets, 0, 0)
        
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
            print(f"✅ 损失计算成功: {float(loss):.6f}")
        else:
            loss = loss_result
            print(f"✅ 损失计算成功: {float(loss):.6f}")
        
        # 测试梯度
        print("🔍 测试梯度计算...")
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        print("✅ 梯度计算和参数更新成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整损失测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 深度修复测试 - preprocess方法")
    print("=" * 60)
    
    # 测试1: preprocess方法修复
    success1 = test_preprocess_fix()
    
    # 测试2: 完整损失计算
    success2 = test_full_loss_with_real_data()
    
    print("\n" + "=" * 60)
    print("🎯 深度修复测试结果")
    print("=" * 60)
    print(f"   preprocess方法: {'✅ 修复成功' if success1 else '❌ 仍有问题'}")
    print(f"   完整损失计算: {'✅ 修复成功' if success2 else '❌ 仍有问题'}")
    
    if success1 and success2:
        print("\n🎉 数据格式问题完全修复！inhomogeneous shape错误已解决！")
        print("✅ 现在可以正常训练了！")
    else:
        print("\n❌ 还有问题需要进一步修复")


if __name__ == "__main__":
    main()
