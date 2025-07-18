#!/usr/bin/env python3
"""
调试后处理问题
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np

# 设置Jittor
jt.flags.use_cuda = 1

def debug_postprocess_issue():
    """调试后处理问题"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        from models.loss import GoldYOLOLoss
        
        print("🔍 调试后处理问题")
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
        
        # 导入后处理函数
        sys.path.append('/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/scripts')
        from gold_yolo_sanity_check import strict_post_process
        
        def test_postprocess(model, label):
            """测试后处理"""
            model.eval()
            with jt.no_grad():
                outputs = model(img_tensor)
            
            print(f"\n{label}:")
            print(f"  输出形状: {outputs.shape}")
            
            # 分析置信度
            conf = outputs[0, :, 4]
            print(f"  置信度范围: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
            print(f"  置信度>0.1: {(conf > 0.1).sum().item()}")
            print(f"  置信度>0.3: {(conf > 0.3).sum().item()}")
            print(f"  置信度>0.5: {(conf > 0.5).sum().item()}")
            
            # 测试不同的后处理阈值
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            for thresh in thresholds:
                detections = strict_post_process(outputs, conf_thres=thresh, iou_thres=0.5, max_det=20)
                det = detections[0]
                print(f"  阈值{thresh}: 检测到{det.shape[0]}个目标")
                
                if det.shape[0] > 0:
                    print(f"    置信度: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
                    print(f"    类别: {set(det[:, 5].numpy().astype(int))}")
            
            return outputs
        
        # 1. 测试训练前
        initial_outputs = test_postprocess(model, "训练前")
        
        # 2. 进行训练
        print(f"\n🏋️ 进行100次训练:")
        model.train()
        optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(100):
            outputs = model(img_tensor)
            loss, loss_items = criterion(outputs, batch)
            optimizer.step(loss)
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: 损失={loss.item():.4f}")
        
        print(f"  最终损失: {loss.item():.4f}")
        
        # 3. 测试训练后
        trained_outputs = test_postprocess(model, "训练后")
        
        # 4. 详细分析训练后的输出
        print(f"\n🔍 详细分析训练后输出:")
        
        # 分析类别预测
        cls_scores = trained_outputs[0, :, 5:]  # [8400, 80]
        
        # 检查目标类别23和24的预测
        cls23_scores = cls_scores[:, 23]
        cls24_scores = cls_scores[:, 24]
        
        print(f"  类别23预测:")
        print(f"    最高分数: {cls23_scores.max().item():.6f}")
        print(f"    平均分数: {cls23_scores.mean().item():.6f}")
        print(f"    >0.5的数量: {(cls23_scores > 0.5).sum().item()}")
        print(f"    >0.8的数量: {(cls23_scores > 0.8).sum().item()}")
        
        print(f"  类别24预测:")
        print(f"    最高分数: {cls24_scores.max().item():.6f}")
        print(f"    平均分数: {cls24_scores.mean().item():.6f}")
        print(f"    >0.5的数量: {(cls24_scores > 0.5).sum().item()}")
        print(f"    >0.8的数量: {(cls24_scores > 0.8).sum().item()}")
        
        # 找出类别23和24分数最高的锚点
        top_cls23_indices = jt.argsort(cls23_scores, descending=True)[0][:5]
        top_cls24_indices = jt.argsort(cls24_scores, descending=True)[0][:5]

        print(f"  类别23最高分数的5个锚点:")
        top_cls23_np = top_cls23_indices.numpy()
        for i, idx_val in enumerate(top_cls23_np):
            conf = trained_outputs[0, idx_val, 4].item()
            cls_score = cls23_scores[idx_val].item()
            print(f"    锚点{idx_val}: 置信度={conf:.3f}, 类别23分数={cls_score:.3f}")

        print(f"  类别24最高分数的5个锚点:")
        top_cls24_np = top_cls24_indices.numpy()
        for i, idx_val in enumerate(top_cls24_np):
            conf = trained_outputs[0, idx_val, 4].item()
            cls_score = cls24_scores[idx_val].item()
            print(f"    锚点{idx_val}: 置信度={conf:.3f}, 类别24分数={cls_score:.3f}")
        
        print("\n" + "=" * 60)
        print("✅ 后处理问题调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_postprocess_issue()
