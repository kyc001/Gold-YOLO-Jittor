#!/usr/bin/env python3
"""
深度诊断学习问题
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np

# 设置Jittor
jt.flags.use_cuda = 1

def diagnose_learning_problem():
    """深度诊断学习问题"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        from models.loss import GoldYOLOLoss
        
        print("🔍 深度诊断Gold-YOLO学习问题")
        print("=" * 60)
        
        # 加载配置和模型
        config = get_config()
        model = Model(config=config, channels=3, num_classes=80)
        criterion = GoldYOLOLoss(num_classes=80)
        
        # 创建测试输入
        img_tensor = jt.randn(1, 3, 640, 640)
        
        # 创建精确的目标 - 模拟真实的COCO数据
        batch = {
            'cls': jt.array([[23, 24]]).long(),  # 真实类别
            'bboxes': jt.array([[[0.304, 0.317, 0.454, 0.564],  # 真实边界框
                                 [0.255, 0.157, 0.603, 0.681]]])
        }
        
        print("📊 目标信息:")
        print(f"  类别: {batch['cls'].numpy()}")
        print(f"  边界框: {batch['bboxes'].numpy()}")
        
        # 1. 分析初始模型输出
        print("\n🔍 分析初始模型输出:")
        model.eval()
        with jt.no_grad():
            initial_outputs = model(img_tensor)
        
        print(f"  输出形状: {initial_outputs.shape}")
        print(f"  置信度范围: [{initial_outputs[0, :, 4].min().item():.6f}, {initial_outputs[0, :, 4].max().item():.6f}]")
        
        # 分析类别预测
        cls_scores = initial_outputs[0, :, 5:]  # [8400, 80]
        max_cls_scores = jt.max(cls_scores, dim=1)[0]
        max_cls_indices = jt.argmax(cls_scores, dim=1)[0]

        print(f"  类别分数范围: [{cls_scores.min().item():.6f}, {cls_scores.max().item():.6f}]")
        print(f"  最高类别分数: {max_cls_scores.max().item():.6f}")

        # 统计预测的类别分布
        unique_classes, counts = np.unique(max_cls_indices.numpy(), return_counts=True)
        top_classes = unique_classes[np.argsort(counts)[-5:]][::-1]
        top_counts = counts[np.argsort(counts)[-5:]][::-1]
        
        print(f"  初始预测最多的5个类别:")
        for cls_id, count in zip(top_classes, top_counts):
            print(f"    类别{cls_id}: {count}次")
        
        # 2. 分析损失计算
        print("\n🔍 分析损失计算:")
        model.train()
        
        # 前向传播
        outputs = model(img_tensor)
        print(f"  训练输出类型: {type(outputs)}")
        
        if isinstance(outputs, list) and len(outputs) == 2:
            detection_output, featmaps = outputs
            print(f"  检测输出类型: {type(detection_output)}")
            print(f"  特征图数量: {len(featmaps)}")
            
            if isinstance(detection_output, tuple) and len(detection_output) == 3:
                feats, pred_scores, pred_distri = detection_output
                print(f"  特征图形状: {[f.shape for f in feats]}")
                print(f"  预测分数形状: {pred_scores.shape}")
                print(f"  预测分布形状: {pred_distri.shape}")
                
                # 分析预测分数
                print(f"  预测分数范围: [{pred_scores.min().item():.6f}, {pred_scores.max().item():.6f}]")
                print(f"  预测分布范围: [{pred_distri.min().item():.6f}, {pred_distri.max().item():.6f}]")
        
        # 计算损失
        loss, loss_items = criterion(outputs, batch)
        print(f"  损失值: {loss.item():.6f}")
        print(f"  损失分量: {loss_items.numpy()}")
        
        # 3. 分析目标分配
        print("\n🔍 分析目标分配:")
        print("  跳过详细分析，专注于核心问题")
        
        # 4. 进行一步训练并分析梯度
        print("\n🔍 分析梯度传播:")
        optimizer = jt.optim.Adam(model.parameters(), lr=0.001)
        
        # 保存训练前的参数
        param_before = {}
        for name, param in model.named_parameters():
            if 'detect' in name:  # 只关注检测头的参数
                param_before[name] = param.clone()
        
        # 执行一步训练
        optimizer.step(loss)
        
        # 检查参数变化
        print("  检测头参数变化:")
        param_changes = {}
        for name, param in model.named_parameters():
            if 'detect' in name and name in param_before:
                change = jt.abs(param - param_before[name]).mean().item()
                param_changes[name] = change
                if change > 1e-8:
                    print(f"    {name}: 变化 {change:.8f}")
        
        if not param_changes or all(change < 1e-8 for change in param_changes.values()):
            print("    ⚠️ 检测头参数几乎没有变化！")
        else:
            print("    ✅ 检测头参数有正常变化")
        
        # 5. 测试训练后的输出
        print("\n🔍 分析训练后输出:")
        model.eval()
        with jt.no_grad():
            trained_outputs = model(img_tensor)
        
        # 导入后处理函数
        sys.path.append('/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/scripts')
        from gold_yolo_sanity_check import strict_post_process
        
        detections = strict_post_process(trained_outputs, conf_thres=0.3, iou_thres=0.5, max_det=20)
        det = detections[0]
        
        print(f"  训练后检测数量: {det.shape[0]}")
        if det.shape[0] > 0:
            print(f"  检测类别: {set(det[:, 5].numpy().astype(int))}")
            print(f"  检测置信度: [{det[:, 4].min().item():.3f}, {det[:, 4].max().item():.3f}]")
        
        print("\n" + "=" * 60)
        print("✅ 学习问题诊断完成")
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_learning_problem()
