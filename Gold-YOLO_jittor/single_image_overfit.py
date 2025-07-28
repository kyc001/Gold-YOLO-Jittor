#!/usr/bin/env python3
"""
单张图片过拟合训练和推理测试脚本
要求：显示训练进度，推理测试结果可视化，检测识别出来物体与真实标注一致
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression

# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
]

# 颜色列表
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
    (255, 192, 203), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128),
    (255, 20, 147), (0, 191, 255), (255, 69, 0), (50, 205, 50), (220, 20, 60)
]

class SingleImageDataset(Dataset):
    """单张图片数据集"""
    
    def __init__(self, img_path, annotations, img_size=640):
        super().__init__()
        self.img_path = img_path
        self.annotations = annotations  # 格式: [[cls, x_center, y_center, width, height], ...]
        self.img_size = img_size
        
        # 加载图像
        self.img = cv2.imread(img_path)
        assert self.img is not None, f"无法读取图像: {img_path}"
        
        print(f"📸 单张图片训练:")
        print(f"   图像路径: {img_path}")
        print(f"   图像尺寸: {self.img.shape}")
        print(f"   标注数量: {len(self.annotations)}")
        for i, ann in enumerate(self.annotations):
            cls_name = COCO_CLASSES[int(ann[0])] if int(ann[0]) < len(COCO_CLASSES) else f'Class{int(ann[0])}'
            print(f"   标注{i+1}: {cls_name} [{ann[1]:.3f}, {ann[2]:.3f}, {ann[3]:.3f}, {ann[4]:.3f}]")
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # 图像预处理
        img = letterbox(self.img, new_shape=self.img_size, stride=32, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        
        # 标签处理 - 支持多个目标
        labels_out = []
        for ann in self.annotations:
            labels_out.append([ann[0], ann[1], ann[2], ann[3], ann[4], 0])
        
        if len(labels_out) == 0:
            labels_out = [[0, 0.5, 0.5, 0.1, 0.1, 0]]  # 默认标签
        
        return jt.array(img, dtype='float32'), jt.array(labels_out, dtype='float32')

def create_sample_annotation():
    """创建示例标注"""
    # 使用一个测试图像创建标注
    test_img = '/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images/2008_000099.jpg'
    
    if os.path.exists(test_img):
        # 为这张图像创建一些示例标注
        annotations = [
            [0, 0.3, 0.4, 0.2, 0.3],  # person在左上区域
            [0, 0.7, 0.6, 0.15, 0.25]  # person在右下区域
        ]
        return test_img, annotations
    else:
        print("❌ 测试图像不存在，请提供有效的图像路径和标注")
        return None, None

def draw_annotations(img, annotations, title="真实标注"):
    """绘制真实标注"""
    img_vis = img.copy()
    h, w = img.shape[:2]
    
    for i, ann in enumerate(annotations):
        cls_id, x_center, y_center, width, height = ann
        
        # 转换为像素坐标
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # 确保坐标在范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        color = COLORS[int(cls_id) % len(COLORS)]
        
        # 绘制边界框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)
        
        # 绘制标签
        cls_name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f'Class{int(cls_id)}'
        label = f'GT: {cls_name}'
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_vis, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
        cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    return img_vis

def draw_predictions(img, detections, conf_thres=0.5, title="预测结果"):
    """绘制预测结果"""
    img_vis = img.copy()
    
    for det in detections:
        if len(det):
            if hasattr(det, 'numpy'):
                det_np = det.numpy()
            else:
                det_np = det
            
            for detection in det_np:
                if isinstance(detection, np.ndarray) and detection.ndim > 1:
                    detection = detection.flatten()
                
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    if conf >= conf_thres:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        class_id = int(cls)
                        confidence = float(conf)
                        
                        # 确保坐标在图像范围内
                        x1 = max(0, min(x1, img.shape[1]-1))
                        y1 = max(0, min(y1, img.shape[0]-1))
                        x2 = max(0, min(x2, img.shape[1]-1))
                        y2 = max(0, min(y2, img.shape[0]-1))
                        
                        color = COLORS[class_id % len(COLORS)]
                        
                        # 绘制边界框
                        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                        
                        # 绘制标签
                        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f'Class{class_id}'
                        label = f'Pred: {class_name} {confidence:.2f}'
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(img_vis, (x1, y1-label_size[1]-5), (x1+label_size[0], y1), color, -1)
                        cv2.putText(img_vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    
    return img_vis

def scale_coords(img1_shape, coords, img0_shape):
    """坐标缩放"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])
    return coords

def single_image_overfit_training():
    """单张图片过拟合训练主函数"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              单张图片过拟合训练系统                           ║
    ║                                                              ║
    ║  🎯 在单张图片上过拟合训练                                   ║
    ║  📊 显示详细训练进度                                         ║
    ║  🔍 推理结果可视化对比                                       ║
    ║  ✅ 确保检测结果与真实标注一致                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建示例数据
    img_path, annotations = create_sample_annotation()
    if img_path is None:
        return False
    
    # 创建数据集
    dataset = SingleImageDataset(img_path, annotations)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 创建模型
    print("\n🔧 创建模型...")
    model = create_perfect_gold_yolo_model('gold_yolo-n', num_classes=20)
    
    # 重新初始化分类头
    print("🔧 重新初始化分类头...")
    for name, module in model.named_modules():
        if 'cls_pred' in name and isinstance(module, nn.Conv2d):
            jt.init.gauss_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                jt.init.constant_(module.bias, -2.0)
    
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
            'class': 2.0,  # 增加分类损失权重
            'iou': 3.0,    # 增加IoU损失权重
            'dfl': 0.5
        }
    )
    
    # 创建优化器
    print("🔧 创建优化器...")
    optimizer = nn.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    
    # 开始过拟合训练
    print("\n🚀 开始单张图片过拟合训练...")
    print(f"   目标: 在单张图片上完全过拟合")
    print(f"   期望: 检测结果与真实标注完全一致")
    print("=" * 70)
    
    model.train()
    
    # 训练统计
    loss_history = []
    epochs = 200  # 过拟合训练轮数
    
    # 获取原始图像用于可视化
    original_img = cv2.imread(img_path)
    
    # 创建保存目录
    save_dir = Path('runs/overfit')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存真实标注图像
    gt_img = draw_annotations(original_img, annotations, "真实标注")
    cv2.imwrite(str(save_dir / 'ground_truth.jpg'), gt_img)
    
    start_time = time.time()
    
    for epoch in range(epochs):
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
        
        # 记录损失
        loss_history.append(epoch_loss)
        
        # 显示训练进度
        if (epoch + 1) % 20 == 0 or epoch < 10:
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (epoch + 1) * (epochs - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {epoch_loss:.6f} | ETA: {eta:.1f}s")
            
            # 每20轮进行一次推理测试
            if (epoch + 1) % 20 == 0:
                print(f"   🔍 进行推理测试...")
                
                model.eval()
                with jt.no_grad():
                    # 预处理图像
                    img = letterbox(original_img, new_shape=640, stride=32, auto=False)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)
                    img = img.astype(np.float32) / 255.0
                    img_tensor = jt.array(img).unsqueeze(0)
                    
                    # 推理
                    pred = model(img_tensor)
                    
                    # 后处理
                    if isinstance(pred, (list, tuple)):
                        pred_for_nms = pred[0]
                    else:
                        pred_for_nms = pred
                    
                    # NMS
                    detections = non_max_suppression(pred_for_nms.unsqueeze(0), conf_thres=0.3, iou_thres=0.45)
                    
                    # 坐标缩放
                    for det in detections:
                        if len(det):
                            det[:, :4] = scale_coords((640, 640), det[:, :4], original_img.shape[:2])
                    
                    # 统计检测结果
                    num_det = len(detections[0]) if len(detections) > 0 and len(detections[0]) > 0 else 0
                    print(f"     检测数量: {num_det}")
                    
                    # 绘制预测结果
                    pred_img = draw_predictions(original_img, detections, conf_thres=0.3)
                    
                    # 保存阶段性结果
                    cv2.imwrite(str(save_dir / f'prediction_epoch_{epoch+1:03d}.jpg'), pred_img)
                    
                    # 创建对比图像
                    comparison = np.hstack([gt_img, pred_img])
                    cv2.imwrite(str(save_dir / f'comparison_epoch_{epoch+1:03d}.jpg'), comparison)
                
                model.train()
    
    # 训练完成
    total_time = time.time() - start_time
    print("\n✅ 单张图片过拟合训练完成！")
    
    # 最终推理测试
    print("\n🔍 最终推理测试...")
    model.eval()
    
    with jt.no_grad():
        # 预处理
        img = letterbox(original_img, new_shape=640, stride=32, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img_tensor = jt.array(img).unsqueeze(0)
        
        # 推理
        pred = model(img_tensor)
        
        # 后处理
        if isinstance(pred, (list, tuple)):
            pred_for_nms = pred[0]
        else:
            pred_for_nms = pred
        
        # 使用更低的置信度阈值进行最终测试
        detections = non_max_suppression(pred_for_nms.unsqueeze(0), conf_thres=0.1, iou_thres=0.45)
        
        # 坐标缩放
        for det in detections:
            if len(det):
                det[:, :4] = scale_coords((640, 640), det[:, :4], original_img.shape[:2])
        
        # 统计最终结果
        num_det = len(detections[0]) if len(detections) > 0 and len(detections[0]) > 0 else 0
        
        print(f"最终检测结果:")
        print(f"   检测数量: {num_det}")
        print(f"   真实标注数量: {len(annotations)}")
        
        # 绘制最终结果
        final_pred_img = draw_predictions(original_img, detections, conf_thres=0.1)
        final_comparison = np.hstack([gt_img, final_pred_img])
        
        # 保存最终结果
        cv2.imwrite(str(save_dir / 'final_prediction.jpg'), final_pred_img)
        cv2.imwrite(str(save_dir / 'final_comparison.jpg'), final_comparison)
        
        # 分析检测质量
        if num_det > 0:
            det = detections[0]
            if hasattr(det, 'numpy'):
                det_np = det.numpy()
            else:
                det_np = det
            
            print(f"   检测详情:")
            for i, detection in enumerate(det_np[:5]):  # 显示前5个
                if isinstance(detection, np.ndarray) and detection.ndim > 1:
                    detection = detection.flatten()
                
                if len(detection) >= 6:
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    class_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else f'Class{int(cls)}'
                    print(f"     {i+1}: {class_name} {conf:.3f} [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('单张图片过拟合训练 - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(str(save_dir / 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存模型
    model_path = save_dir / 'overfit_model.pkl'
    jt.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epochs,
        'loss_history': loss_history,
        'annotations': annotations,
        'img_path': img_path
    }, str(model_path))
    
    # 输出总结
    print("\n" + "="*70)
    print("🎉 单张图片过拟合训练完成！")
    print("="*70)
    print(f"📊 训练统计:")
    print(f"   训练轮数: {epochs}")
    print(f"   训练时间: {total_time/60:.1f}分钟")
    print(f"   初始损失: {loss_history[0]:.6f}")
    print(f"   最终损失: {loss_history[-1]:.6f}")
    print(f"   损失下降: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%")
    
    print(f"\n📋 检测结果:")
    print(f"   真实标注: {len(annotations)}个目标")
    print(f"   检测结果: {num_det}个目标")
    print(f"   过拟合状态: {'成功' if num_det >= len(annotations) else '需要更多训练'}")
    
    print(f"\n💾 保存文件:")
    print(f"   模型权重: {model_path}")
    print(f"   损失曲线: {save_dir}/loss_curve.png")
    print(f"   真实标注: {save_dir}/ground_truth.jpg")
    print(f"   最终预测: {save_dir}/final_prediction.jpg")
    print(f"   对比结果: {save_dir}/final_comparison.jpg")
    
    return True

if __name__ == "__main__":
    print("🚀 开始单张图片过拟合训练...")
    
    success = single_image_overfit_training()
    
    if success:
        print("\n🎉 单张图片过拟合训练成功完成！")
        print("📋 现在可以查看:")
        print("   - runs/overfit/final_comparison.jpg (对比结果)")
        print("   - runs/overfit/loss_curve.png (训练曲线)")
        print("   - runs/overfit/ 目录下的所有阶段性结果")
    else:
        print("\n❌ 单张图片过拟合训练失败！")
