#!/usr/bin/env python3
"""
简单测试训练前后的差异
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np

# 设置Jittor
jt.flags.use_cuda = 1

def simple_test():
    """简单测试"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        from models.loss import GoldYOLOLoss
        
        print("🔍 简单测试训练前后差异")
        print("=" * 60)
        
        # 加载配置和模型
        config = get_config()
        model = Model(config=config, channels=3, num_classes=80)
        criterion = GoldYOLOLoss(num_classes=80)
        
        # 创建测试输入
        img_tensor = jt.randn(1, 3, 640, 640)
        
        # 创建目标
        batch = {
            'cls': jt.array([[23, 24]]).long(),
            'bboxes': jt.array([[[0.304, 0.317, 0.454, 0.564],
                                 [0.255, 0.157, 0.603, 0.681]]])
        }
        
        def test_model_output(model, label):
            """测试模型输出"""
            model.eval()
            with jt.no_grad():
                outputs = model(img_tensor)
            
            print(f"\n{label}:")
            print(f"  输出形状: {outputs.shape}")
            
            # 分析置信度
            conf = outputs[0, :, 4]
            print(f"  置信度范围: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
            print(f"  置信度均值: {conf.mean().item():.6f}")
            print(f"  置信度>0.1: {(conf > 0.1).sum().item()}")
            print(f"  置信度>0.3: {(conf > 0.3).sum().item()}")
            print(f"  置信度>0.5: {(conf > 0.5).sum().item()}")
            
            # 分析类别预测
            cls_scores = outputs[0, :, 5:]
            max_cls_scores = jt.max(cls_scores, dim=1)[0]
            max_cls_indices = jt.argmax(cls_scores, dim=1)[0]
            
            print(f"  类别分数范围: [{cls_scores.min().item():.6f}, {cls_scores.max().item():.6f}]")
            print(f"  最高类别分数: {max_cls_scores.max().item():.6f}")
            
            # 统计预测的类别分布
            unique_classes, counts = np.unique(max_cls_indices.numpy(), return_counts=True)
            top_classes = unique_classes[np.argsort(counts)[-3:]][::-1]
            top_counts = counts[np.argsort(counts)[-3:]][::-1]
            
            print(f"  预测最多的3个类别:")
            for cls_id, count in zip(top_classes, top_counts):
                print(f"    类别{cls_id}: {count}次")
            
            return outputs
        
        # 1. 测试训练前
        initial_outputs = test_model_output(model, "训练前")
        
        # 2. 进行简单训练
        print(f"\n🏋️ 进行10次训练:")
        model.train()
        optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(10):
            outputs = model(img_tensor)
            loss, loss_items = criterion(outputs, batch)
            optimizer.step(loss)
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: 损失={loss.item():.4f}")
        
        # 3. 测试训练后
        trained_outputs = test_model_output(model, "训练后")
        
        # 4. 对比分析
        print(f"\n📊 对比分析:")
        
        # 置信度对比
        initial_conf = initial_outputs[0, :, 4]
        trained_conf = trained_outputs[0, :, 4]
        
        print(f"  置信度变化:")
        print(f"    训练前均值: {initial_conf.mean().item():.6f}")
        print(f"    训练后均值: {trained_conf.mean().item():.6f}")
        print(f"    变化: {(trained_conf.mean() - initial_conf.mean()).item():.6f}")
        
        # 类别预测对比
        initial_cls = initial_outputs[0, :, 5:]
        trained_cls = trained_outputs[0, :, 5:]
        
        print(f"  类别预测变化:")
        print(f"    训练前最高分数: {initial_cls.max().item():.6f}")
        print(f"    训练后最高分数: {trained_cls.max().item():.6f}")
        
        # 检查是否有目标类别23和24的预测提升
        initial_cls23 = initial_cls[:, 23].mean().item()
        initial_cls24 = initial_cls[:, 24].mean().item()
        trained_cls23 = trained_cls[:, 23].mean().item()
        trained_cls24 = trained_cls[:, 24].mean().item()
        
        print(f"  目标类别预测变化:")
        print(f"    类别23: {initial_cls23:.6f} -> {trained_cls23:.6f} (变化: {trained_cls23-initial_cls23:.6f})")
        print(f"    类别24: {initial_cls24:.6f} -> {trained_cls24:.6f} (变化: {trained_cls24-initial_cls24:.6f})")
        
        if trained_cls23 > initial_cls23 and trained_cls24 > initial_cls24:
            print("  ✅ 目标类别预测有提升，模型在学习！")
        else:
            print("  ❌ 目标类别预测没有明显提升")
        
        print("\n" + "=" * 60)
        print("✅ 简单测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
