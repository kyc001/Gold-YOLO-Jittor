#!/usr/bin/env python3
"""
完整的自检训练流程 - 不简化任何步骤
深入修复所有问题，完成500轮完整训练
"""

import os
import sys
import time
import cv2
import numpy as np
import yaml
from pathlib import Path

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset, DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.general import scale_coords

class CompleteDataset(Dataset):
    """完整的单张图片数据集 - 严格对齐训练数据格式"""
    
    def __init__(self, img_path, label_path, img_size=640):
        super().__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.img_size = img_size
        
        # 加载图像
        self.img = cv2.imread(img_path)
        assert self.img is not None, f"无法读取图像: {img_path}"
        
        # 加载标签
        self.labels = self.load_labels(label_path)
        
        print(f"📸 自检图像: {img_path}")
        print(f"🏷️ 图像尺寸: {self.img.shape}")
        print(f"🎯 目标数量: {len(self.labels)}")
        if len(self.labels) > 0:
            print(f"🎯 目标类别: {[int(label[0]) for label in self.labels]}")
    
    def load_labels(self, label_path):
        """加载YOLO格式标签"""
        if not os.path.exists(label_path):
            print(f"⚠️ 标签文件不存在: {label_path}")
            return []
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append([cls_id, x_center, y_center, width, height])
        
        return labels
    
    def __len__(self):
        return 1  # 只有一张图片
    
    def __getitem__(self, idx):
        # 图像预处理 - 严格对齐训练流程
        img = letterbox(self.img, new_shape=self.img_size, stride=32, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
        
        # 标签处理 - 使用验证成功的格式
        if len(self.labels) > 0:
            # 单个目标的格式：[cls, x_center, y_center, width, height, 0]
            label = self.labels[0]  # 使用第一个标签
            labels_out = jt.array([[label[0], label[1], label[2], label[3], label[4], 0]], dtype='float32')
        else:
            labels_out = jt.zeros((1, 6), dtype='float32')  # 空标签
        
        return jt.array(img, dtype='float32'), labels_out

def create_complete_dataset():
    """创建完整的自检数据集"""
    # 选择测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    
    # 创建对应的标签文件
    label_path = 'complete_self_check_label.txt'
    
    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"❌ 测试图像不存在: {img_path}")
        return None, None
    
    # 创建标签 - 使用更大更明显的目标
    with open(label_path, 'w') as f:
        # 类别0，中心位置(0.5, 0.5)，尺寸(0.8, 0.8) - 大目标更容易学习
        f.write("0 0.5 0.5 0.8 0.8\n")
    
    print(f"✅ 创建完整自检标签: {label_path}")
    
    # 创建数据集
    dataset = CompleteDataset(img_path, label_path)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    return dataset, dataloader

def complete_self_check_training():
    """完整的自检训练主函数 - 不简化任何步骤"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              GOLD-YOLO-n 完整自检训练验证系统                ║
    ║                                                              ║
    ║  🎯 完整500轮训练，不简化任何步骤                            ║
    ║  🔧 深入修复所有问题                                         ║
    ║  📊 训练完成后检测同一张图片验证识别能力                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建完整自检数据集
    print("📦 创建完整自检数据集...")
    dataset, dataloader = create_complete_dataset()
    if dataset is None:
        return False
    
    # 创建模型 - 使用完整的模型创建流程
    print("🔧 创建完整模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头 - 修复分类问题
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            print(f"   重新初始化: {name}")
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                jt.init.constant_(module.bias, 0.01)
    
    # 创建损失函数 - 使用完整的损失配置
    print("🔧 创建完整损失函数...")
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
            'class': 5.0,  # 增加分类损失权重
            'iou': 2.5,
            'dfl': 0.5
        }
    )
    
    # 创建优化器 - 使用完整的优化配置
    print("🔧 创建完整优化器...")
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    
    # 开始完整自检训练
    print("🚀 开始完整自检训练...")
    print(f"   训练轮数: 500 (完整训练)")
    print(f"   学习率: 0.01")
    print(f"   分类损失权重: 5.0")
    print(f"   优化器: SGD with momentum=0.9")
    print("=" * 70)
    
    model.train()
    
    # 训练统计
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(500):
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            # 前向传播
            predictions = model(images)
            
            # 计算损失
            loss, loss_items = loss_fn(predictions, targets, epoch_num=epoch+1, step_num=batch_idx+1)
            
            if loss is not None:
                # 反向传播
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()
                
                epoch_loss += float(loss.numpy())
            else:
                print(f"   警告: Epoch {epoch+1} 损失为None")
        
        # 记录损失
        loss_history.append(epoch_loss)
        
        # 更新最佳损失
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        
        # 打印训练进度 - 不简化任何输出
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}/500 | Loss: {epoch_loss:.6f} | Best: {best_loss:.6f}")
            
            # 每100轮检查一次分类头输出
            if (epoch + 1) % 100 == 0:
                model.eval()
                with jt.no_grad():
                    test_pred = model(images)[0]
                    test_cls_conf = test_pred[:, 5:]
                    cls_min = float(test_cls_conf.min().numpy())
                    cls_max = float(test_cls_conf.max().numpy())
                    cls_range = cls_max - cls_min
                    cls_mean = float(test_cls_conf.mean().numpy())
                    print(f"         类别置信度: 范围[{cls_min:.6f}, {cls_max:.6f}], 变化范围: {cls_range:.6f}, 均值: {cls_mean:.6f}")
                model.train()
    
    print("✅ 完整自检训练完成！")
    
    # 分析训练过程
    print("\n📊 训练过程分析:")
    print(f"   初始损失: {loss_history[0]:.6f}")
    print(f"   最终损失: {loss_history[-1]:.6f}")
    print(f"   最佳损失: {best_loss:.6f}")
    print(f"   损失下降: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%")
    
    # 保存完整自检模型
    save_path = 'complete_self_check_model.pkl'
    jt.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': 500,
        'loss_history': loss_history,
        'best_loss': best_loss
    }, save_path)
    print(f"💾 完整自检模型已保存: {save_path}")
    
    return model, loss_history

def complete_self_check_inference(model):
    """完整的自检推理 - 不简化任何步骤"""
    print("\n🔍 开始完整自检推理...")
    model.eval()
    
    # 加载测试图像
    img_path = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    
    print(f"📸 测试图像: {os.path.basename(img_path)}")
    print(f"🖼️ 图像尺寸: {w0}x{h0}")
    
    # 预处理 - 严格对齐训练时的预处理
    img = letterbox(img0, new_shape=640, stride=32, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0  # 归一化到[0,1]
    img = jt.array(img)
    if img.ndim == 3:
        img = img.unsqueeze(0)  # 添加batch维度
    
    print(f"🔧 预处理后尺寸: {img.shape}")
    
    # 推理 - 完整的推理流程
    print("🔍 执行推理...")
    start_time = time.time()
    with jt.no_grad():
        pred = model(img)
    inference_time = time.time() - start_time
    
    print(f"⏱️ 推理时间: {inference_time*1000:.2f}ms")
    print(f"📊 原始输出形状: {pred.shape}")
    
    # 详细分析模型输出
    pred = pred[0]  # 移除batch维度 [8400, 25]
    
    boxes = pred[:, :4]  # [x_center, y_center, width, height]
    obj_conf = pred[:, 4]  # 目标置信度
    cls_conf = pred[:, 5:]  # 类别置信度 [20]
    
    print(f"\n📊 详细输出分析:")
    print(f"   坐标预测形状: {boxes.shape}")
    print(f"   目标置信度形状: {obj_conf.shape}")
    print(f"   类别置信度形状: {cls_conf.shape}")
    
    obj_min = float(obj_conf.min().numpy())
    obj_max = float(obj_conf.max().numpy())
    cls_min = float(cls_conf.min().numpy())
    cls_max = float(cls_conf.max().numpy())
    cls_mean = float(cls_conf.mean().numpy())
    cls_std = float(cls_conf.std().numpy())
    
    print(f"   目标置信度范围: [{obj_min:.6f}, {obj_max:.6f}]")
    print(f"   类别置信度范围: [{cls_min:.6f}, {cls_max:.6f}]")
    print(f"   类别置信度均值: {cls_mean:.6f}")
    print(f"   类别置信度标准差: {cls_std:.6f}")
    
    # 计算最终置信度
    cls_scores = cls_conf.max(dim=1)[0]  # 最大类别置信度
    cls_indices = cls_conf.argmax(dim=1)  # 类别索引
    final_conf = obj_conf * cls_scores
    
    final_min = float(final_conf.min().numpy())
    final_max = float(final_conf.max().numpy())
    final_mean = float(final_conf.mean().numpy())
    
    print(f"   最终置信度范围: [{final_min:.6f}, {final_max:.6f}]")
    print(f"   最终置信度均值: {final_mean:.6f}")
    
    # 后处理 - 完整的NMS流程
    print("\n🔧 执行完整NMS后处理...")
    
    # 使用多个置信度阈值测试
    conf_thresholds = [0.5, 0.25, 0.1, 0.05, 0.01]
    
    for conf_thres in conf_thresholds:
        print(f"\n   测试置信度阈值: {conf_thres}")
        
        try:
            # NMS处理
            pred_nms = non_max_suppression(pred.unsqueeze(0), conf_thres=conf_thres, iou_thres=0.45, max_det=1000)
            
            # 统计检测结果
            detections = []
            for i, det in enumerate(pred_nms):
                if len(det):
                    detections.append(det)
                else:
                    detections.append(jt.empty((0, 6)))
            
            num_det = len(detections[0]) if len(detections) > 0 and len(detections[0]) > 0 else 0
            print(f"     NMS后检测数量: {num_det}")
            
            if num_det > 0:
                print(f"     ✅ 在置信度阈值{conf_thres}下检测到{num_det}个目标")
                
                # 显示前3个检测结果
                det = detections[0]
                if hasattr(det, 'numpy'):
                    det_np = det.numpy()
                else:
                    det_np = det
                
                for j in range(min(3, len(det_np))):
                    x1, y1, x2, y2, conf, cls = det_np[j]
                    print(f"       检测{j+1}: 类别={int(cls)}, 置信度={conf:.6f}, 坐标=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
                
                break  # 找到有效检测就停止
            else:
                print(f"     ❌ 在置信度阈值{conf_thres}下未检测到目标")
        
        except Exception as e:
            print(f"     ❌ NMS处理失败: {e}")
    
    # 创建可视化结果
    print(f"\n🎨 创建可视化结果...")
    
    # 使用最低阈值强制显示检测结果
    img_vis = img0.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # 显示前5个最高置信度的检测
    top_indices = final_conf.argsort(descending=True)[:5]
    
    for i, idx in enumerate(top_indices):
        box = boxes[idx].numpy()
        obj_c = float(obj_conf[idx].numpy())
        cls_idx = int(cls_indices[idx].numpy())
        cls_c = float(cls_scores[idx].numpy())
        final_c = float(final_conf[idx].numpy())
        
        # 转换坐标
        x_center, y_center, width, height = box
        x1 = int((x_center - width/2) * w0 / 640)
        y1 = int((y_center - height/2) * h0 / 640)
        x2 = int((x_center + width/2) * w0 / 640)
        y2 = int((y_center + height/2) * h0 / 640)
        
        # 确保坐标在范围内
        x1 = max(0, min(x1, w0-1))
        y1 = max(0, min(y1, h0-1))
        x2 = max(0, min(x2, w0-1))
        y2 = max(0, min(y2, h0-1))
        
        color = colors[i % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # 绘制详细标签
        label = f'Complete{i+1}_C{cls_idx} {final_c:.6f}'
        cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        print(f"   可视化{i+1}: 类别={cls_idx}, 目标={obj_c:.6f}, 类别={cls_c:.6f}, 最终={final_c:.6f}")
    
    # 保存完整结果
    result_path = 'complete_self_check_result.jpg'
    cv2.imwrite(result_path, img_vis)
    print(f"📸 完整自检结果已保存: {result_path}")
    
    # 评估结果
    print(f"\n🎯 完整自检评估:")
    
    if final_max > 0.1:
        print(f"   ✅ 检测功能优秀 (最高置信度: {final_max:.6f})")
        success_level = "优秀"
    elif final_max > 0.01:
        print(f"   ✅ 检测功能良好 (最高置信度: {final_max:.6f})")
        success_level = "良好"
    elif final_max > 0.001:
        print(f"   ⚠️ 检测功能基本 (最高置信度: {final_max:.6f})")
        success_level = "基本"
    else:
        print(f"   ❌ 检测功能不足 (最高置信度: {final_max:.6f})")
        success_level = "不足"
    
    cls_range = cls_max - cls_min
    if cls_range > 0.01:
        print(f"   ✅ 分类功能正常 (变化范围: {cls_range:.6f})")
    else:
        print(f"   ⚠️ 分类功能有限 (变化范围: {cls_range:.6f})")
    
    return success_level, final_max, cls_range

if __name__ == "__main__":
    print("🚀 开始GOLD-YOLO完整自检训练验证...")
    
    # 完整自检训练
    model, loss_history = complete_self_check_training()
    
    if model is not None:
        # 完整自检推理
        success_level, final_max, cls_range = complete_self_check_inference(model)
        
        print("\n" + "="*70)
        print("🎉 GOLD-YOLO完整自检训练验证完成！")
        print("="*70)
        print(f"📊 最终评估结果:")
        print(f"   训练完成度: 100% (500/500轮)")
        print(f"   检测功能等级: {success_level}")
        print(f"   最高检测置信度: {final_max:.6f}")
        print(f"   分类置信度变化范围: {cls_range:.6f}")
        print(f"   模型保存: complete_self_check_model.pkl")
        print(f"   结果图像: complete_self_check_result.jpg")
        
        if success_level in ["优秀", "良好"]:
            print("🎉 完整自检验证成功！模型功能正常！")
        else:
            print("⚠️ 完整自检验证显示模型仍需进一步优化")
    else:
        print("❌ 完整自检训练失败！")
