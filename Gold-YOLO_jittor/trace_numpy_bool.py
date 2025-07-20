#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
精确追踪numpy布尔判断错误的工具
"""

import jittor as jt
import numpy as np
import traceback
import sys

# Set Jittor flags
jt.flags.use_cuda = 1

class NumpyBoolTracker:
    """numpy布尔判断追踪器"""
    
    def __init__(self):
        self.original_bool = np.ndarray.__bool__
        self.call_count = 0
        
    def __enter__(self):
        # 替换numpy的__bool__方法
        np.ndarray.__bool__ = self.debug_bool
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始方法
        np.ndarray.__bool__ = self.original_bool
        
    def debug_bool(self, arr):
        """调试版本的__bool__方法"""
        self.call_count += 1
        
        if arr.size > 1:
            print(f"\n❌ 第{self.call_count}次错误的numpy数组布尔判断:")
            print(f"   数组形状: {arr.shape}")
            print(f"   数组大小: {arr.size}")
            print(f"   数组内容: {arr}")
            print("   调用栈:")
            
            # 打印调用栈，但跳过这个方法本身
            stack = traceback.extract_stack()[:-1]
            for i, frame in enumerate(stack[-10:]):  # 只显示最后10层
                print(f"     {i+1}. {frame.filename}:{frame.lineno} in {frame.name}")
                print(f"        {frame.line}")
            
            print("\n   建议修复方案:")
            print("   - 如果要检查是否有任何True: 使用 arr.any()")
            print("   - 如果要检查是否全部True: 使用 arr.all()")
            print("   - 如果要检查数组长度: 使用 len(arr) > 0")
            print("   - 如果要检查元素个数: 使用 arr.size > 0")
            
            # 抛出详细错误
            raise ValueError(f"The truth value of an array with more than one element is ambiguous. "
                           f"Array shape: {arr.shape}, size: {arr.size}. Use a.any() or a.all()")
        
        return self.original_bool(arr)

def test_complete_assigner_with_tracking():
    """使用追踪器测试完整分配器"""
    print("🔧 使用追踪器测试完整分配器...")
    
    with NumpyBoolTracker() as tracker:
        try:
            from yolov6.models.losses.loss import ComputeLoss
            
            # 创建损失函数，强制使用完整分配器
            criterion = ComputeLoss(
                num_classes=80,
                ori_img_size=640,
                warmup_epoch=0,  # 强制使用formal_assigner
                use_dfl=True,
                reg_max=16,
                iou_type='giou'
            )
            
            # 创建测试数据
            batch_size = 1
            n_anchors = 100  # 使用小数据便于调试
            
            # 模拟模型输出
            feats = [
                jt.randn(1, 256, 10, 10),   # P3
                jt.randn(1, 512, 5, 5),     # P4  
                jt.randn(1, 1024, 3, 3)     # P5
            ]
            
            pred_scores = jt.randn(batch_size, n_anchors, 80)
            pred_distri = jt.randn(batch_size, n_anchors, 68)
            
            outputs = (feats, pred_scores, pred_distri)
            
            # 创建目标数据
            targets = [{
                'cls': jt.array([1]),
                'bboxes': jt.array([[100, 100, 200, 200]]),
                'batch_idx': jt.array([0])
            }]
            
            print(f"  输入形状:")
            print(f"    pred_scores: {pred_scores.shape}")
            print(f"    pred_distri: {pred_distri.shape}")
            print(f"    targets: {len(targets)} 个目标")
            
            # 调用损失函数
            print("  调用损失函数...")
            loss, loss_items = criterion(outputs, targets, epoch_num=1, step_num=0)
            
            print(f"  ✅ 损失函数调用成功: {loss}")
            print(f"  总共进行了 {tracker.call_count} 次numpy布尔判断")
            return True
            
        except Exception as e:
            print(f"  ❌ 损失函数调用失败: {e}")
            print(f"  在第 {tracker.call_count} 次numpy布尔判断时失败")
            return False

def main():
    """主测试函数"""
    print("🔧 精确追踪numpy布尔判断错误")
    print("=" * 60)
    
    success = test_complete_assigner_with_tracking()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {'✅ 成功' if success else '❌ 失败'}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
