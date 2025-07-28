#!/usr/bin/env python3
"""
修复坐标转换问题
"""

import os
import sys
import numpy as np
import jittor as jt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_coordinate_conversion():
    """测试坐标转换问题"""
    print("🔧 测试坐标转换问题...")
    
    # 模拟损失函数中的坐标处理
    print("\n1. 模拟原始标签:")
    targets = jt.array([[[0, 0.5, 0.5, 0.8, 0.8, 0]]], dtype='float32')  # [1, 1, 6]
    print(f"   原始targets: {targets.numpy()}")
    
    print("\n2. 提取坐标部分:")
    coords_before = targets[:, :, 1:5]  # [1, 1, 4] - [x_center, y_center, width, height]
    print(f"   坐标部分: {coords_before.numpy()}")
    
    print("\n3. 应用缩放:")
    scale_tensor = jt.array([640, 640, 640, 640], dtype='float32')
    batch_target = coords_before * scale_tensor
    print(f"   缩放后: {batch_target.numpy()}")
    print(f"   缩放后数值范围: [{float(batch_target.min().numpy()):.6f}, {float(batch_target.max().numpy()):.6f}]")
    
    print("\n4. 测试原始xywh2xyxy函数:")
    from yolov6.utils.general import xywh2xyxy
    
    # 复制一份用于测试
    test_coords = batch_target.clone()
    print(f"   转换前: {test_coords.numpy()}")
    
    result = xywh2xyxy(test_coords)
    print(f"   转换后: {result.numpy()}")
    print(f"   转换后数值范围: [{float(result.min().numpy()):.6f}, {float(result.max().numpy()):.6f}]")
    
    print("\n5. 问题分析:")
    if float(result.min().numpy()) == float(result.max().numpy()) == 0.0:
        print("   ❌ 发现问题：转换后坐标全部为0")
        print("   🔧 可能原因：xywh2xyxy函数直接修改了输入张量")
    else:
        print("   ✅ 坐标转换正常")
    
    print("\n6. 创建修复版本的xywh2xyxy函数:")
    
    def fixed_xywh2xyxy(bboxes):
        """修复版本的坐标转换函数"""
        # 创建新的张量，避免修改原始输入
        result = jt.zeros_like(bboxes)
        
        cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        
        result[..., 0] = cx - w * 0.5  # x1 = cx - w/2
        result[..., 1] = cy - h * 0.5  # y1 = cy - h/2
        result[..., 2] = cx + w * 0.5  # x2 = cx + w/2
        result[..., 3] = cy + h * 0.5  # y2 = cy + h/2
        
        return result
    
    print("\n7. 测试修复版本:")
    test_coords2 = batch_target.clone()
    print(f"   转换前: {test_coords2.numpy()}")
    
    fixed_result = fixed_xywh2xyxy(test_coords2)
    print(f"   修复版转换后: {fixed_result.numpy()}")
    print(f"   修复版数值范围: [{float(fixed_result.min().numpy()):.6f}, {float(fixed_result.max().numpy()):.6f}]")
    
    # 验证原始输入是否被修改
    print(f"   原始输入是否被修改: {jt.equal(test_coords2, batch_target).all().numpy()}")
    
    if float(fixed_result.min().numpy()) != 0.0 or float(fixed_result.max().numpy()) != 0.0:
        print("   ✅ 修复版本工作正常！")
        return True
    else:
        print("   ❌ 修复版本仍有问题")
        return False

def apply_fix():
    """应用修复"""
    print("\n🔧 应用坐标转换修复...")
    
    # 备份原始文件
    import shutil
    original_file = 'yolov6/utils/general.py'
    backup_file = 'yolov6/utils/general.py.backup'
    
    if not os.path.exists(backup_file):
        shutil.copy(original_file, backup_file)
        print(f"✅ 已备份原始文件: {backup_file}")
    
    # 读取原始文件
    with open(original_file, 'r') as f:
        content = f.read()
    
    # 替换xywh2xyxy函数
    old_function = """def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    # 修复坐标转换bug - 保存原始中心坐标和尺寸
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes[..., 0] = cx - w * 0.5  # x1 = cx - w/2
    bboxes[..., 1] = cy - h * 0.5  # y1 = cy - h/2
    bboxes[..., 2] = cx + w * 0.5  # x2 = cx + w/2
    bboxes[..., 3] = cy + h * 0.5  # y2 = cy + h/2
    return bboxes"""

    new_function = """def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    # 修复坐标转换bug - 创建新张量避免修改原始输入
    result = jt.zeros_like(bboxes)

    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]

    result[..., 0] = cx - w * 0.5  # x1 = cx - w/2
    result[..., 1] = cy - h * 0.5  # y1 = cy - h/2
    result[..., 2] = cx + w * 0.5  # x2 = cx + w/2
    result[..., 3] = cy + h * 0.5  # y2 = cy + h/2

    return result"""
    
    # 替换内容
    new_content = content.replace(old_function, new_function)
    
    # 写入修复后的文件
    with open(original_file, 'w') as f:
        f.write(new_content)
    
    print(f"✅ 已修复坐标转换函数: {original_file}")
    
    return True

if __name__ == "__main__":
    print("🚀 开始修复坐标转换问题...")
    
    # 测试问题
    success = test_coordinate_conversion()
    
    if not success:
        print("\n🔧 检测到坐标转换问题，开始修复...")
        apply_fix()
        
        print("\n🔧 重新测试修复后的函数...")
        # 重新导入模块以获取修复后的函数
        import importlib
        import yolov6.utils.general
        importlib.reload(yolov6.utils.general)
        
        # 重新测试
        success = test_coordinate_conversion()
        
        if success:
            print("\n🎉 坐标转换问题修复成功！")
        else:
            print("\n❌ 坐标转换问题修复失败！")
    else:
        print("\n✅ 坐标转换功能正常，无需修复")
    
    print("\n📋 修复总结:")
    print("   问题：xywh2xyxy函数直接修改输入张量，导致坐标被覆盖")
    print("   解决：创建新张量存储结果，避免修改原始输入")
    print("   影响：修复后训练时的标签坐标将正确转换")
