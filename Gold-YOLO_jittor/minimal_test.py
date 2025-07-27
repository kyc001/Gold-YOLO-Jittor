#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 最小自检脚本
全面检查模型的每个环节，确保能完整跑通
"""

import os
import sys
import traceback

# 设置环境变量
os.environ['JT_SYNC'] = '1'

def test_jittor_basic():
    """测试1: Jittor基础功能"""
    print("🔍 测试1: Jittor基础功能")
    try:
        import jittor as jt
        jt.flags.use_cuda = 0  # 强制CPU避免CUDA问题
        
        # 基础张量操作
        x = jt.randn(2, 3)
        y = x * 2 + 1
        z = jt.sum(y)
        
        print(f"   ✅ Jittor版本: {jt.__version__}")
        print(f"   ✅ 基础运算: {float(z):.3f}")
        return True, jt
    except Exception as e:
        print(f"   ❌ Jittor基础功能失败: {e}")
        return False, None


def test_model_creation(jt):
    """测试2: 模型创建"""
    print("\n🔍 测试2: 模型创建")
    try:
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        
        model = create_perfect_gold_yolo_model('gold_yolo-n', 20)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ 模型创建成功")
        print(f"   ✅ 总参数: {total_params:,} ({total_params/1e6:.2f}M)")
        
        return True, model
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        traceback.print_exc()
        return False, None


def test_model_forward(jt, model):
    """测试3: 模型前向传播"""
    print("\n🔍 测试3: 模型前向传播")
    try:
        # 创建测试输入
        x = jt.randn(1, 3, 640, 640)
        
        # 前向传播
        model.eval()
        with jt.no_grad():
            output = model(x)
        
        print(f"   ✅ 输入形状: {list(x.shape)}")
        print(f"   ✅ 输出形状: {list(output.shape)}")
        print(f"   ✅ 输出范围: [{float(output.min()):.3f}, {float(output.max()):.3f}]")
        
        # 检查输出格式
        expected_shape = [1, 8400, 25]  # batch=1, anchors=8400, features=25
        if list(output.shape) == expected_shape:
            print(f"   ✅ 输出格式正确: YOLO格式 [4坐标+1置信度+20类别]")
            return True, output
        else:
            print(f"   ⚠️ 输出格式异常: 期望{expected_shape}, 得到{list(output.shape)}")
            return False, output
            
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        traceback.print_exc()
        return False, None


def test_loss_function(jt, output):
    """测试4: 损失函数"""
    print("\n🔍 测试4: 损失函数")
    try:
        # 方法1: 尝试导入修复版本
        try:
            # 直接导入losses.py中的ComputeLoss
            import importlib.util
            losses_file = os.path.join(os.path.dirname(__file__), 'yolov6', 'models', 'losses.py')
            spec = importlib.util.spec_from_file_location("fixed_losses", losses_file)
            fixed_losses = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fixed_losses)
            ComputeLoss = fixed_losses.ComputeLoss
            print(f"   ✅ 使用修复版损失函数")
        except:
            # 备用方案: 使用包导入
            from yolov6.models.losses import ComputeLoss
            print(f"   ✅ 使用包导入损失函数")
        
        # 创建损失函数
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
        
        # 创建测试目标
        targets = jt.array([
            [0, 5, 0.5, 0.5, 0.2, 0.2],  # batch_idx, class, x, y, w, h
        ]).float32()
        
        # 计算损失
        loss_result = loss_fn(output, targets, 0, 0)
        
        if isinstance(loss_result, tuple):
            loss = loss_result[0]
        else:
            loss = loss_result
        
        loss_value = float(loss)
        print(f"   ✅ 损失计算成功: {loss_value:.6f}")
        
        # 检查损失是否合理
        if jt.isnan(loss) or jt.isinf(loss):
            print(f"   ❌ 损失值无效: {loss_value}")
            return False, None
        elif loss_value == 0:
            print(f"   ⚠️ 损失值为0，可能有问题")
            return False, None
        else:
            print(f"   ✅ 损失值有效且非零")
            return True, loss_fn
            
    except Exception as e:
        print(f"   ❌ 损失函数失败: {e}")
        traceback.print_exc()
        return False, None


def test_data_loading():
    """测试5: 数据加载"""
    print("\n🔍 测试5: 数据加载")
    try:
        import yaml
        
        # 检查数据配置文件
        config_path = 'data/voc_subset_improved.yaml'
        if not os.path.exists(config_path):
            print(f"   ❌ 数据配置文件不存在: {config_path}")
            return False, None
        
        # 加载配置
        with open(config_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"   ✅ 数据配置加载成功")
        print(f"   ✅ 类别数: {data_config.get('nc', 'unknown')}")
        print(f"   ✅ 训练路径: {data_config.get('train', 'unknown')}")
        print(f"   ✅ 验证路径: {data_config.get('val', 'unknown')}")
        
        # 检查数据路径是否存在
        train_path = data_config.get('train', '')
        val_path = data_config.get('val', '')
        
        if os.path.exists(train_path):
            print(f"   ✅ 训练数据路径存在")
        else:
            print(f"   ❌ 训练数据路径不存在: {train_path}")
            return False, None
            
        if os.path.exists(val_path):
            print(f"   ✅ 验证数据路径存在")
        else:
            print(f"   ❌ 验证数据路径不存在: {val_path}")
            return False, None
        
        return True, data_config
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        traceback.print_exc()
        return False, None


def test_training_step(jt, model, loss_fn):
    """测试6: 完整训练步骤"""
    print("\n🔍 测试6: 完整训练步骤")
    try:
        # 创建优化器
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # 设置训练模式
        model.train()
        
        # 记录初始参数
        initial_param = None
        for name, param in model.named_parameters():
            if 'weight' in name:
                initial_param = param.clone()
                break
        
        print(f"   ✅ 优化器创建成功")
        
        # 进行一步训练
        for step in range(3):
            # 创建训练数据
            images = jt.randn(1, 3, 640, 640)
            targets = jt.array([
                [0, 5, 0.5, 0.5, 0.2, 0.2],
            ]).float32()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss_result = loss_fn(outputs, targets, 0, step)
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
            else:
                loss = loss_result
            
            # 损失缩放
            loss_value = float(loss)
            if loss_value > 10.0:
                loss = loss / 5.0
            
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            print(f"     步骤{step+1}: 损失={float(loss):.6f}")
        
        # 检查参数是否变化
        final_param = None
        for name, param in model.named_parameters():
            if 'weight' in name:
                final_param = param
                break
        
        if initial_param is not None and final_param is not None:
            param_change = float(jt.mean(jt.abs(final_param - initial_param)))
            print(f"   ✅ 参数变化: {param_change:.8f}")
            
            if param_change > 1e-8:
                print(f"   ✅ 参数正常更新，模型在学习")
                return True
            else:
                print(f"   ❌ 参数几乎没有变化")
                return False
        else:
            print(f"   ⚠️ 无法检查参数变化")
            return True
            
    except Exception as e:
        print(f"   ❌ 训练步骤失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 GOLD-YOLO Jittor版本 - 最小自检脚本")
    print("=" * 60)
    print("🎯 全面检查模型的每个环节")
    print("=" * 60)
    
    # 测试结果记录
    results = {}
    
    # 测试1: Jittor基础功能
    success, jt = test_jittor_basic()
    results['Jittor基础功能'] = success
    if not success:
        print("\n❌ Jittor基础功能失败，停止测试")
        return
    
    # 测试2: 模型创建
    success, model = test_model_creation(jt)
    results['模型创建'] = success
    if not success:
        print("\n❌ 模型创建失败，停止测试")
        return
    
    # 测试3: 模型前向传播
    success, output = test_model_forward(jt, model)
    results['模型前向传播'] = success
    if not success:
        print("\n❌ 模型前向传播失败，停止测试")
        return
    
    # 测试4: 损失函数
    success, loss_fn = test_loss_function(jt, output)
    results['损失函数'] = success
    if not success:
        print("\n❌ 损失函数失败，停止测试")
        return
    
    # 测试5: 数据加载
    success, data_config = test_data_loading()
    results['数据加载'] = success
    if not success:
        print("\n❌ 数据加载失败，但可以继续测试训练步骤")
    
    # 测试6: 完整训练步骤
    success = test_training_step(jt, model, loss_fn)
    results['完整训练步骤'] = success
    
    # 总结
    print("\n" + "=" * 60)
    print("🎯 自检结果总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！GOLD-YOLO Jittor版本完全可用！")
        print("✅ 可以开始完整训练")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"🚨 以下测试失败: {', '.join(failed_tests)}")
        print("❌ 需要修复后才能开始训练")
    print("=" * 60)


if __name__ == "__main__":
    main()
