#!/usr/bin/env python3
"""
调试Gold-YOLO模型输出格式
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np
from PIL import Image

# 设置Jittor
jt.flags.use_cuda = 1

def debug_model_output():
    """调试模型输出格式"""
    
    try:
        from models.yolo import Model
        from configs.gold_yolo_s import get_config
        
        # 加载配置和模型
        config = get_config()
        model = Model(config=config, channels=3, num_classes=80)
        
        # 创建测试输入
        img_tensor = jt.randn(1, 3, 640, 640)
        
        print("🔍 调试Gold-YOLO模型输出格式")
        print("=" * 60)
        
        # 测试训练模式
        print("\n📊 训练模式输出:")
        model.train()
        train_outputs = model(img_tensor)
        
        print(f"训练输出类型: {type(train_outputs)}")
        if isinstance(train_outputs, list):
            print(f"训练输出列表长度: {len(train_outputs)}")
            for i, output in enumerate(train_outputs):
                if isinstance(output, (list, tuple)):
                    print(f"  输出[{i}]: {type(output)}, 长度={len(output)}")
                    for j, item in enumerate(output):
                        if hasattr(item, 'shape'):
                            print(f"    项[{j}]: 形状={item.shape}")
                        else:
                            print(f"    项[{j}]: {type(item)}")
                elif hasattr(output, 'shape'):
                    print(f"  输出[{i}]: 形状={output.shape}")
                else:
                    print(f"  输出[{i}]: {type(output)}")
        
        # 测试推理模式
        print("\n📊 推理模式输出:")
        model.eval()
        with jt.no_grad():
            infer_outputs = model(img_tensor)
        
        print(f"推理输出类型: {type(infer_outputs)}")
        if hasattr(infer_outputs, 'shape'):
            print(f"推理输出形状: {infer_outputs.shape}")
            print(f"推理输出数据类型: {infer_outputs.dtype}")
            
            # 检查输出的数值范围
            print(f"推理输出最小值: {infer_outputs.min().item():.6f}")
            print(f"推理输出最大值: {infer_outputs.max().item():.6f}")
            print(f"推理输出均值: {infer_outputs.mean().item():.6f}")
            
            # 检查输出的具体格式
            if len(infer_outputs.shape) == 3:
                batch, num_anchors, features = infer_outputs.shape
                print(f"批次大小: {batch}")
                print(f"锚点数量: {num_anchors}")
                print(f"特征维度: {features}")
                
                if features == 85:  # 4 + 1 + 80
                    print("✅ 输出格式正确: [x1, y1, x2, y2, conf, cls1, cls2, ..., cls80]")
                    
                    # 分析各部分的数值范围
                    boxes = infer_outputs[0, :, :4]
                    conf = infer_outputs[0, :, 4]
                    classes = infer_outputs[0, :, 5:]
                    
                    print(f"\n📦 边界框分析:")
                    print(f"  X坐标范围: [{boxes[:, 0].min().item():.2f}, {boxes[:, 0].max().item():.2f}]")
                    print(f"  Y坐标范围: [{boxes[:, 1].min().item():.2f}, {boxes[:, 1].max().item():.2f}]")
                    print(f"  宽度范围: [{(boxes[:, 2] - boxes[:, 0]).min().item():.2f}, {(boxes[:, 2] - boxes[:, 0]).max().item():.2f}]")
                    print(f"  高度范围: [{(boxes[:, 3] - boxes[:, 1]).min().item():.2f}, {(boxes[:, 3] - boxes[:, 1]).max().item():.2f}]")
                    
                    print(f"\n🎯 置信度分析:")
                    print(f"  置信度范围: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
                    print(f"  置信度均值: {conf.mean().item():.6f}")
                    print(f"  置信度>0.1的数量: {(conf > 0.1).sum().item()}")
                    print(f"  置信度>0.5的数量: {(conf > 0.5).sum().item()}")
                    print(f"  置信度>0.7的数量: {(conf > 0.7).sum().item()}")
                    
                    print(f"\n🏷️ 类别分析:")
                    print(f"  类别分数范围: [{classes.min().item():.6f}, {classes.max().item():.6f}]")
                    print(f"  类别分数均值: {classes.mean().item():.6f}")
                    
                    # 简化的置信度分析
                    print(f"\n🔝 置信度分布分析:")
                    conf_np = conf.numpy()
                    print(f"  置信度标准差: {np.std(conf_np):.6f}")
                    print(f"  置信度中位数: {np.median(conf_np):.6f}")

                    # 检查是否所有置信度都相同
                    unique_conf = np.unique(conf_np)
                    print(f"  唯一置信度值数量: {len(unique_conf)}")
                    if len(unique_conf) <= 5:
                        print(f"  唯一置信度值: {unique_conf}")

                    # 类别分析
                    cls_np = classes.numpy()
                    max_cls_scores = np.max(cls_np, axis=1)
                    max_cls_indices = np.argmax(cls_np, axis=1)

                    print(f"\n🏷️ 类别预测分析:")
                    print(f"  最高类别分数范围: [{np.min(max_cls_scores):.6f}, {np.max(max_cls_scores):.6f}]")
                    print(f"  预测类别范围: [{np.min(max_cls_indices)}, {np.max(max_cls_indices)}]")

                    # 统计最常预测的类别
                    unique_classes, counts = np.unique(max_cls_indices, return_counts=True)
                    top_classes = unique_classes[np.argsort(counts)[-5:]][::-1]
                    top_counts = counts[np.argsort(counts)[-5:]][::-1]

                    print(f"  最常预测的5个类别:")
                    for cls_id, count in zip(top_classes, top_counts):
                        print(f"    类别{cls_id}: {count}次 ({count/len(max_cls_indices)*100:.1f}%)")
                
                else:
                    print(f"❌ 输出格式异常: 期望85维特征，实际{features}维")
            else:
                print(f"❌ 输出形状异常: 期望3维张量，实际{len(infer_outputs.shape)}维")
        
        print("\n" + "=" * 60)
        print("✅ 模型输出格式调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_output()
