#!/usr/bin/env python3
"""
调试scale_coords函数
"""

import sys
import os
sys.path.insert(0, '/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor')

import jittor as jt
import numpy as np

# 设置Jittor
jt.flags.use_cuda = 1

def debug_scale_coords():
    """调试scale_coords函数"""
    
    try:
        # 导入scale_coords函数
        sys.path.append('/home/kyc/project/GOLD-YOLO/Gold-YOLO_jittor/scripts')
        from gold_yolo_sanity_check import scale_coords
        
        print("🔍 调试scale_coords函数")
        print("=" * 60)
        
        # 创建测试检测结果
        test_detections = jt.array([
            [100, 100, 200, 200, 0.8, 1],  # 一个测试检测框
            [300, 300, 400, 400, 0.9, 2],  # 另一个测试检测框
        ])
        
        print(f"原始检测结果:")
        print(f"  形状: {test_detections.shape}")
        print(f"  内容: {test_detections.numpy()}")
        
        # 测试缩放
        img1_shape = (640, 640)  # 模型输入尺寸
        img0_shape = (424, 640)  # 原始图像尺寸
        
        print(f"\n缩放参数:")
        print(f"  从: {img1_shape}")
        print(f"  到: {img0_shape}")
        
        # 复制检测结果以避免原地修改
        test_coords = test_detections.clone()
        
        print(f"\n缩放前:")
        print(f"  坐标: {test_coords[:, :4].numpy()}")
        
        try:
            # 调用scale_coords函数
            scaled_coords = scale_coords(img1_shape, test_coords, img0_shape)
            
            print(f"\n缩放后:")
            print(f"  坐标: {scaled_coords[:, :4].numpy()}")
            print(f"  形状: {scaled_coords.shape}")
            
            # 检查是否有有效的检测框
            valid_boxes = 0
            for i in range(scaled_coords.shape[0]):
                x1, y1, x2, y2 = scaled_coords[i, :4].numpy()
                if x2 > x1 and y2 > y1:
                    valid_boxes += 1
                    print(f"  检测框{i+1}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] - 有效")
                else:
                    print(f"  检测框{i+1}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] - 无效")
            
            print(f"\n有效检测框数量: {valid_boxes}")
            
        except Exception as e:
            print(f"❌ scale_coords函数调用失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 手动实现一个简单的缩放函数进行对比
        print(f"\n🛠️ 手动缩放测试:")
        
        def simple_scale_coords(coords_np, from_shape, to_shape):
            """简单的坐标缩放函数"""
            scale_x = to_shape[1] / from_shape[1]  # width scale
            scale_y = to_shape[0] / from_shape[0]  # height scale
            
            coords_np[:, 0] *= scale_x  # x1
            coords_np[:, 1] *= scale_y  # y1
            coords_np[:, 2] *= scale_x  # x2
            coords_np[:, 3] *= scale_y  # y2
            
            # 限制坐标范围
            coords_np[:, 0] = np.clip(coords_np[:, 0], 0, to_shape[1])
            coords_np[:, 1] = np.clip(coords_np[:, 1], 0, to_shape[0])
            coords_np[:, 2] = np.clip(coords_np[:, 2], 0, to_shape[1])
            coords_np[:, 3] = np.clip(coords_np[:, 3], 0, to_shape[0])
            
            return coords_np
        
        # 测试简单缩放
        test_coords_np = test_detections.numpy().copy()
        print(f"简单缩放前: {test_coords_np[:, :4]}")
        
        simple_scaled = simple_scale_coords(test_coords_np[:, :4], img1_shape, img0_shape)
        print(f"简单缩放后: {simple_scaled}")
        
        # 检查简单缩放的有效性
        valid_simple = 0
        for i in range(simple_scaled.shape[0]):
            x1, y1, x2, y2 = simple_scaled[i]
            if x2 > x1 and y2 > y1:
                valid_simple += 1
        
        print(f"简单缩放有效检测框: {valid_simple}")
        
        print("\n" + "=" * 60)
        print("✅ scale_coords调试完成")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_scale_coords()
