#!/usr/bin/env python3
"""
调试训练对推理的影响
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np

# 设置Jittor
jt.flags.use_cuda = 1

def test_inference(model, img_tensor, label=""):
    """测试推理结果"""
    model.eval()
    with jt.no_grad():
        outputs = model(img_tensor)
    
    # 导入后处理函数
    sys.path.append('/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/scripts')
    from gold_yolo_sanity_check import strict_post_process
    
    detections = strict_post_process(outputs, conf_thres=0.3, iou_thres=0.5, max_det=20)
    det = detections[0]
    
    print(f"{label}:")
    print(f"  模型输出形状: {outputs.shape}")
    print(f"  置信度范围: [{outputs[0, :, 4].min().item():.3f}, {outputs[0, :, 4].max().item():.3f}]")
    print(f"  检测数量: {det.shape[0]}")
    
    if det.shape[0] > 0:
        print(f"  检测置信度: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
        print(f"  检测类别: {set(det[:, 5].numpy().astype(int))}")
    
    return det.shape[0]

def debug_training_effect():
    """调试训练对推理的影响"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        
        print("🔍 调试训练对推理的影响")
        print("=" * 60)
        
        # 加载配置和模型
        config = get_config()
        model = Model(config=config, channels=3, num_classes=80)
        
        # 创建测试输入
        img_tensor = jt.randn(1, 3, 640, 640)
        
        # 测试训练前的推理
        print("\n📊 训练前推理测试:")
        det_count_before = test_inference(model, img_tensor, "训练前")
        
        # 进行简单的训练
        print("\n🏋️ 开始简单训练...")
        model.train()
        optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
        
        # 创建虚拟目标
        dummy_targets = [{
            'boxes': jt.array([[0.3, 0.3, 0.7, 0.7]]),  # 一个虚拟边界框
            'labels': jt.array([1])  # 虚拟类别
        }]
        
        # 训练几个epoch
        for epoch in range(10):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(img_tensor)
            
            # 简单的虚拟损失（只是为了更新权重）
            if isinstance(outputs, list):
                loss = sum([out[1].mean() for out in outputs if isinstance(out, (list, tuple)) and len(out) > 1])
            else:
                loss = outputs.mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: 损失={loss.item():.4f}")
        
        print("✅ 训练完成")
        
        # 测试训练后的推理
        print("\n📊 训练后推理测试:")
        det_count_after = test_inference(model, img_tensor, "训练后")
        
        # 对比结果
        print(f"\n📈 对比结果:")
        print(f"  训练前检测数量: {det_count_before}")
        print(f"  训练后检测数量: {det_count_after}")
        print(f"  变化: {det_count_after - det_count_before}")
        
        if det_count_before > 0 and det_count_after == 0:
            print("  ⚠️ 警告: 训练后检测数量变为0，可能存在问题")
        elif det_count_before == 0 and det_count_after > 0:
            print("  ✅ 好消息: 训练后开始检测到目标")
        elif det_count_before > 0 and det_count_after > 0:
            print("  ✅ 正常: 训练前后都能检测到目标")
        else:
            print("  ❌ 问题: 训练前后都检测不到目标")
        
        print("\n" + "=" * 60)
        print("✅ 训练影响调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training_effect()
