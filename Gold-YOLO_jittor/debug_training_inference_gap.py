#!/usr/bin/env python3
"""
调试训练和推理之间的差异
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np

# 设置Jittor
jt.flags.use_cuda = 1

def test_model_state_consistency():
    """测试模型状态一致性"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        from models.loss import GoldYOLOLoss
        
        print("🔍 调试训练和推理状态一致性")
        print("=" * 60)
        
        # 加载配置和模型
        config = get_config()
        model = Model(config=config, channels=3, num_classes=80)
        criterion = GoldYOLOLoss(num_classes=80)
        
        # 创建测试输入
        img_tensor = jt.randn(1, 3, 640, 640)
        
        # 创建虚拟目标
        batch = {
            'cls': jt.array([[1, 2]]).long(),
            'bboxes': jt.array([[[0.3, 0.3, 0.7, 0.7], [0.1, 0.1, 0.5, 0.5]]])
        }
        
        def test_inference(model, label):
            """测试推理能力"""
            model.eval()
            with jt.no_grad():
                outputs = model(img_tensor)
            
            # 导入后处理函数
            sys.path.append('/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/scripts')
            from gold_yolo_sanity_check import strict_post_process
            
            detections = strict_post_process(outputs, conf_thres=0.3, iou_thres=0.5, max_det=20)
            det = detections[0]
            
            print(f"{label}:")
            print(f"  检测数量: {det.shape[0]}")
            if det.shape[0] > 0:
                print(f"  置信度范围: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
                print(f"  检测类别: {set(det[:, 5].numpy().astype(int))}")
            
            return det.shape[0]
        
        # 1. 测试初始状态
        print("\n📊 初始状态测试:")
        initial_count = test_inference(model, "初始状态")
        
        # 2. 保存初始权重
        print("\n💾 保存初始权重...")
        initial_weights = {}
        for name, param in model.named_parameters():
            initial_weights[name] = param.clone()
        
        # 3. 进行训练
        print("\n🏋️ 开始训练...")
        model.train()
        optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(20):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(img_tensor)
            
            # 损失计算
            loss, loss_items = criterion(outputs, batch)
            
            # 反向传播 - 使用Jittor的正确语法
            optimizer.step(loss)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: 损失={loss.item():.4f}")
        
        # 4. 测试训练后状态
        print("\n📊 训练后状态测试:")
        trained_count = test_inference(model, "训练后状态")
        
        # 5. 恢复初始权重
        print("\n🔄 恢复初始权重...")
        for name, param in model.named_parameters():
            param.data = initial_weights[name]
        
        # 6. 测试恢复后状态
        print("\n📊 恢复权重后状态测试:")
        restored_count = test_inference(model, "恢复权重后状态")
        
        # 7. 分析结果
        print(f"\n📈 结果分析:")
        print(f"  初始检测数量: {initial_count}")
        print(f"  训练后检测数量: {trained_count}")
        print(f"  恢复后检测数量: {restored_count}")
        
        if initial_count > 0 and trained_count == 0 and restored_count > 0:
            print("  🎯 结论: 训练过程破坏了推理能力，但权重恢复后能够修复")
            print("  💡 建议: 检查训练过程中的参数更新是否合理")
        elif initial_count > 0 and trained_count == 0 and restored_count == 0:
            print("  ⚠️ 结论: 可能存在模型状态管理问题")
        elif initial_count > 0 and trained_count > 0:
            print("  ✅ 结论: 训练过程正常，推理能力保持")
        else:
            print("  ❓ 结论: 需要进一步分析")
        
        # 8. 检查关键参数变化
        print(f"\n🔍 关键参数变化分析:")
        param_changes = {}
        for name, param in model.named_parameters():
            if name in initial_weights:
                change = jt.abs(param - initial_weights[name]).mean().item()
                param_changes[name] = change
        
        # 显示变化最大的参数
        sorted_changes = sorted(param_changes.items(), key=lambda x: x[1], reverse=True)
        print("  变化最大的5个参数:")
        for name, change in sorted_changes[:5]:
            print(f"    {name}: 平均变化 {change:.6f}")
        
        print("\n" + "=" * 60)
        print("✅ 训练推理一致性调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_state_consistency()
