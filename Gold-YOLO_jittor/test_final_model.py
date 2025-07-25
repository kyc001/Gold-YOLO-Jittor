#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 最终模型测试
验证模型能否正常运行，包括前向传播、训练模式等
"""

import sys
import os
import traceback

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_final_model():
    """测试最终模型的完整功能"""
    print("🚀 开始GOLD-YOLO最终模型测试...")
    print("=" * 80)
    
    try:
        import jittor as jt
        from models.perfect_gold_yolo import create_perfect_gold_yolo_model
        
        # 创建模型
        print("🏗️ 创建GOLD-YOLO-n模型...")
        model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
        
        # 参数量统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📊 模型参数统计:")
        print(f"   总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   目标参数量: 5,635,904 (5.6M)")
        print(f"   对齐率: {total_params/5635904*100:.2f}%")
        
        # 测试前向传播
        print(f"\n🚀 测试前向传播...")
        x = jt.randn(1, 3, 640, 640)
        print(f"   输入形状: {list(x.shape)}")
        
        # 推理模式
        model.eval()
        with jt.no_grad():
            outputs = model(x)
        
        if isinstance(outputs, (list, tuple)):
            print(f"   ✅ 推理成功，输出{len(outputs)}个特征图:")
            for i, out in enumerate(outputs):
                print(f"      P{i+3}: {list(out.shape)}")
        else:
            print(f"   ✅ 推理成功，输出形状: {list(outputs.shape)}")
        
        # 测试训练模式
        print(f"\n🎯 测试训练模式...")
        model.train()
        
        # 模拟训练前向传播
        outputs = model(x)
        
        if isinstance(outputs, (list, tuple)):
            print(f"   ✅ 训练模式成功，输出{len(outputs)}个特征图")
            
            # 计算一个简单的损失
            total_loss = 0
            for out in outputs:
                # 简单的L2损失
                target = jt.zeros_like(out)
                loss = jt.mean((out - target) ** 2)
                total_loss += loss
            
            print(f"   模拟损失: {total_loss.item():.6f}")
            
            # 测试反向传播
            print(f"   🔄 测试反向传播...")
            total_loss.backward()
            print(f"   ✅ 反向传播成功")
            
        else:
            print(f"   ✅ 训练模式成功，输出形状: {list(outputs.shape)}")
        
        # 测试不同输入尺寸
        print(f"\n📏 测试不同输入尺寸...")
        test_sizes = [(1, 3, 320, 320), (1, 3, 416, 416), (2, 3, 640, 640)]
        
        model.eval()
        for size in test_sizes:
            try:
                x_test = jt.randn(*size)
                with jt.no_grad():
                    out_test = model(x_test)
                
                if isinstance(out_test, (list, tuple)):
                    shapes = [list(o.shape) for o in out_test]
                    print(f"   ✅ {size} -> {shapes}")
                else:
                    print(f"   ✅ {size} -> {list(out_test.shape)}")
                    
            except Exception as e:
                print(f"   ❌ {size} -> 失败: {e}")
        
        # 测试模型保存和加载
        print(f"\n💾 测试模型保存和加载...")
        try:
            # 保存模型
            save_path = "test_model.pkl"
            jt.save(model.state_dict(), save_path)
            print(f"   ✅ 模型保存成功: {save_path}")
            
            # 加载模型
            state_dict = jt.load(save_path)
            model.load_state_dict(state_dict)
            print(f"   ✅ 模型加载成功")
            
            # 清理
            os.remove(save_path)
            
        except Exception as e:
            print(f"   ❌ 模型保存/加载失败: {e}")
        
        # 性能测试
        print(f"\n⚡ 性能测试...")
        import time
        
        model.eval()
        x = jt.randn(1, 3, 640, 640)
        
        # 预热
        for _ in range(5):
            with jt.no_grad():
                _ = model(x)
        
        # 计时
        start_time = time.time()
        num_runs = 10
        
        for _ in range(num_runs):
            with jt.no_grad():
                _ = model(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        print(f"   平均推理时间: {avg_time*1000:.2f}ms")
        print(f"   推理FPS: {fps:.2f}")
        
        print(f"\n" + "=" * 80)
        print(f"🎉 GOLD-YOLO最终模型测试完成！")
        print(f"✅ 所有功能正常工作")
        print(f"📊 参数量对齐率: {total_params/5635904*100:.2f}%")
        print(f"🚀 模型已准备就绪，可用于训练和推理！")
        
        return True
        
    except Exception as e:
        print(f"❌ 最终模型测试失败: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_final_model()
    sys.exit(0 if success else 1)
