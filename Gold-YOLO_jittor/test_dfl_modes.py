#!/usr/bin/env python3
"""
测试DFL损失在开启和关闭两种模式下的工作情况
"""

import jittor as jt
jt.flags.use_cuda = 1

from yolov6.models.losses import ComputeLoss
import numpy as np

def test_dfl_mode(use_dfl, reg_max, mode_name):
    """测试指定的DFL模式"""
    print(f"\n{'='*50}")
    print(f"🔍 测试 {mode_name}")
    print(f"   use_dfl={use_dfl}, reg_max={reg_max}")
    print(f"{'='*50}")
    
    try:
        # 创建损失函数
        loss_fn = ComputeLoss(
            fpn_strides=[8, 16, 32],
            grid_cell_size=5.0,
            grid_cell_offset=0.5,
            num_classes=20,
            ori_img_size=640,
            warmup_epoch=4,
            use_dfl=use_dfl,
            reg_max=reg_max,
            iou_type='giou',
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        
        # 不需要创建head，直接测试损失函数
        
        # 创建测试数据
        batch_size = 1
        
        # 模拟模型输出
        if use_dfl and reg_max > 0:
            # DFL模式：每个坐标有(reg_max+1)个分布参数
            reg_channels = 4 * (reg_max + 1)
        else:
            # 传统模式：每个anchor有4个坐标
            reg_channels = 4 * 3  # 3个anchor
        
        cls_channels = 20 * 3  # 20个类别 * 3个anchor
        
        # 创建模拟的模型输出 - 使用损失函数期望的格式
        feats = []
        for i, stride in enumerate([8, 16, 32]):
            h, w = 640 // stride, 640 // stride

            # 分类输出
            cls_output = jt.randn(batch_size, cls_channels, h, w) * 0.1

            # 回归输出
            reg_output = jt.randn(batch_size, reg_channels, h, w) * 0.1

            feats.append([cls_output, reg_output])

        # 创建pred_scores和pred_distri
        total_anchors = sum([(640//stride)**2 * 3 for stride in [8, 16, 32]])  # 总anchor数
        pred_scores = jt.randn(batch_size, total_anchors, 20) * 0.1  # [batch, anchors, classes]

        if use_dfl and reg_max > 0:
            pred_distri = jt.randn(batch_size, total_anchors, 4 * (reg_max + 1)) * 0.1  # DFL格式
        else:
            pred_distri = jt.randn(batch_size, total_anchors, 4) * 0.1  # 传统格式

        outputs = (feats, pred_scores, pred_distri)
        
        # 创建测试标签
        targets = jt.array([
            [0.0, 0.0, 14.0, 0.5, 0.5, 0.1, 0.1],  # [batch_idx, padding, class_id, x, y, w, h]
            [0.0, 0.0, 18.0, 0.3, 0.3, 0.2, 0.2],
        ], dtype='float32')
        
        # 计算损失
        total_loss, loss_items = loss_fn(outputs, targets, epoch_num=1, step_num=1)
        
        print(f"✅ {mode_name} 测试成功！")
        print(f"   总损失: {float(total_loss.data):.6f}")
        print(f"   分类损失: {float(loss_items[0].data):.6f}")
        print(f"   IoU损失: {float(loss_items[1].data):.6f}")
        print(f"   DFL损失: {float(loss_items[2].data):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ {mode_name} 测试失败！")
        print(f"   错误: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🎯 开始测试DFL损失的两种模式")
    
    # 测试模式1：DFL禁用（gold-yolo-n默认配置）
    success1 = test_dfl_mode(
        use_dfl=False, 
        reg_max=0, 
        mode_name="DFL禁用模式（gold-yolo-n默认）"
    )
    
    # 测试模式2：DFL启用（其他模型配置）
    success2 = test_dfl_mode(
        use_dfl=True, 
        reg_max=16, 
        mode_name="DFL启用模式（gold-yolo-s/m/l）"
    )
    
    # 总结测试结果
    print(f"\n{'='*60}")
    print("🎯 测试结果总结")
    print(f"{'='*60}")
    print(f"DFL禁用模式: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"DFL启用模式: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过！DFL损失在两种模式下都能正常工作！")
        return True
    else:
        print("\n🚨 部分测试失败！需要进一步修复！")
        return False

if __name__ == "__main__":
    main()
