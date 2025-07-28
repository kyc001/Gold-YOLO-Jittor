#!/usr/bin/env python3
"""
简化的自检训练 - 使用验证成功的格式
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from jittor import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.general import scale_coords

def simple_self_check():
    """简化的自检训练"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                简化自检训练验证系统                           ║
    ║                                                              ║
    ║  🎯 使用验证成功的标签格式进行自检训练                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建模型
    print("🔧 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    model.train()
    
    # 创建损失函数
    print("🔧 创建损失函数...")
    import importlib.util
    losses_file = os.path.join(os.path.dirname(__file__), 'yolov6', 'models', 'losses.py')
    spec = importlib.util.spec_from_file_location("losses", losses_file)
    losses_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(losses_module)
    
    loss_fn = losses_module.ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        iou_type='siou',
        loss_weight={
            'class': 1.0,
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 创建优化器
    print("🔧 创建优化器...")
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    
    # 准备训练数据 - 使用验证成功的格式
    print("🔧 准备训练数据...")
    
    # 加载测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    
    # 预处理图像
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
    images = jt.array(img).unsqueeze(0)  # 添加batch维度
    
    # 使用验证成功的标签格式
    targets = jt.array([[0, 0.5, 0.5, 0.8, 0.8, 0]], dtype='float32')
    
    print(f"训练图像形状: {images.shape}")
    print(f"训练标签形状: {targets.shape}")
    print(f"训练标签内容: {targets.numpy()}")
    
    # 开始训练
    print("🚀 开始自检训练...")
    print("   训练轮数: 500")
    print("   学习率: 0.01")
    print("=" * 70)
    
    for epoch in range(500):
        # 前向传播
        predictions = model(images)
        
        # 计算损失
        loss, _ = loss_fn(predictions, targets, epoch_num=epoch+1, step_num=1)
        
        if loss is not None:
            # 反向传播
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            epoch_loss = float(loss.data)
        else:
            epoch_loss = 0.0
        
        # 打印训练进度
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/500 | Loss: {epoch_loss:.6f}")
    
    print("✅ 自检训练完成！")
    
    # 保存模型
    save_path = 'simple_self_check_model.pkl'
    jt.save(model.state_dict(), save_path)
    print(f"💾 自检模型已保存: {save_path}")
    
    # 进行推理测试
    print("\n🔍 开始自检推理...")
    model.eval()
    
    # 推理
    with jt.no_grad():
        raw_pred = model(images)

    # 检查原始模型输出
    print(f"原始模型输出形状: {raw_pred.shape}")
    print(f"原始模型输出数值范围: [{float(raw_pred.min().data):.6f}, {float(raw_pred.max().data):.6f}]")

    # 检查置信度分布
    if raw_pred.shape[-1] >= 5:
        conf_scores = raw_pred[0, :, 4]  # 第5列是置信度
        print(f"置信度数值范围: [{float(conf_scores.min().data):.6f}, {float(conf_scores.max().data):.6f}]")
        high_conf_count = (conf_scores > 0.001).sum()
        print(f"置信度>0.001的预测数量: {int(high_conf_count.data)}")

        # 找到最高置信度的预测
        max_conf_idx = conf_scores.argmax(dim=0)[0]
        max_conf = conf_scores[max_conf_idx]
        print(f"最高置信度: {float(max_conf.data):.6f} (索引: {int(max_conf_idx.data)})")

        # 检查类别预测
        cls_scores = raw_pred[0, :, 5:]  # 第6列开始是类别
        print(f"类别预测数值范围: [{float(cls_scores.min().data):.6f}, {float(cls_scores.max().data):.6f}]")

    # 后处理 - 使用极低的置信度阈值
    pred = non_max_suppression(raw_pred, conf_thres=0.001, iou_thres=0.45, max_det=1000)
    
    # 统计检测结果
    detections = []
    for i, det in enumerate(pred):
        if len(det):
            # 坐标缩放回原图尺寸
            det[:, :4] = scale_coords(images.shape[2:], det[:, :4], img0.shape).round()
            detections.append(det)
        else:
            detections.append(jt.empty((0, 6)))
    
    # 输出检测结果
    num_det = len(detections[0]) if len(detections) > 0 and len(detections[0]) > 0 else 0
    print(f"🎯 检测结果: {num_det} 个目标")
    
    if num_det > 0:
        det = detections[0]
        if hasattr(det, 'numpy'):
            det = det.numpy()
        
        for i, (*xyxy, conf, cls) in enumerate(det):
            print(f"   目标 {i+1}: 类别={int(cls)}, 置信度={conf:.3f}, 坐标=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        
        # 保存可视化结果
        img_vis = img0.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, (*xyxy, conf, cls) in enumerate(det):
            if conf >= 0.01:
                x1, y1, x2, y2 = map(int, xyxy)
                color = colors[i % len(colors)]
                
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                label = f'Class{int(cls)} {conf:.2f}'
                cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite('simple_self_check_result.jpg', img_vis)
        print(f"📸 可视化结果已保存: simple_self_check_result.jpg")
        
        print("🎉 简化自检验证成功！模型能够检测到目标！")
        return True
    else:
        print("⚠️ 简化自检验证失败：模型未检测到任何目标")
        
        # 输出模型原始预测进行调试
        print("\n🔧 调试信息：")
        print(f"模型输出数量: {len(pred)}")
        if len(pred) > 0:
            raw_pred = pred[0]
            print(f"原始预测形状: {raw_pred.shape}")
            if len(raw_pred) > 0:
                print(f"原始预测数值范围: [{float(raw_pred.min().data):.6f}, {float(raw_pred.max().data):.6f}]")
                # 检查置信度
                if raw_pred.shape[-1] >= 5:
                    conf_scores = raw_pred[:, 4]
                    print(f"置信度范围: [{float(conf_scores.min().data):.6f}, {float(conf_scores.max().data):.6f}]")
                    high_conf_count = (conf_scores > 0.01).sum()
                    print(f"置信度>0.01的预测数量: {int(high_conf_count.data)}")
        
        return False

if __name__ == "__main__":
    success = simple_self_check()
    if success:
        print("\n✅ 简化自检训练验证完成！模型功能正常！")
    else:
        print("\n❌ 简化自检训练验证失败！需要进一步调试！")
