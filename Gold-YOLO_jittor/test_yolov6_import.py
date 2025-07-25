#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 模块导入测试脚本
验证所有模块可以正确导入，解决依赖问题
"""

import sys
import os
import traceback

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """测试基础导入"""
    print("🔍 测试基础导入...")
    try:
        import jittor as jt
        print("✅ Jittor导入成功")
        
        import jittor.nn as nn
        print("✅ Jittor.nn导入成功")

        # Jittor没有单独的functional模块，功能函数在nn中
        print("✅ Jittor API结构确认")
        
        return True
    except Exception as e:
        print(f"❌ 基础导入失败: {e}")
        traceback.print_exc()
        return False


def test_yolov6_layers():
    """测试yolov6.layers模块"""
    print("\n🔍 测试yolov6.layers模块...")
    try:
        from yolov6.layers.common import (
            Conv, SimConv, ConvWrapper, SimConvWrapper,
            SimSPPF, SPPF, RepVGGBlock, RepBlock, BottleRep
        )
        print("✅ yolov6.layers.common导入成功")
        
        from yolov6.layers.dbb_transforms import transI_fusebn
        print("✅ yolov6.layers.dbb_transforms导入成功")
        
        return True
    except Exception as e:
        print(f"❌ yolov6.layers导入失败: {e}")
        traceback.print_exc()
        return False


def test_yolov6_utils():
    """测试yolov6.utils模块"""
    print("\n🔍 测试yolov6.utils模块...")
    try:
        from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy
        print("✅ yolov6.utils.general导入成功")
        
        from yolov6.utils.jittor_utils import initialize_weights, time_sync
        print("✅ yolov6.utils.jittor_utils导入成功")
        
        from yolov6.utils.events import LOGGER, load_yaml
        print("✅ yolov6.utils.events导入成功")
        
        return True
    except Exception as e:
        print(f"❌ yolov6.utils导入失败: {e}")
        traceback.print_exc()
        return False


def test_yolov6_assigners():
    """测试yolov6.assigners模块"""
    print("\n🔍 测试yolov6.assigners模块...")
    try:
        from yolov6.assigners.anchor_generator import generate_anchors
        print("✅ yolov6.assigners.anchor_generator导入成功")
        
        from yolov6.assigners.assigner_utils import dist_calculator
        print("✅ yolov6.assigners.assigner_utils导入成功")
        
        from yolov6.assigners.iou2d_calculator import iou2d_calculator
        print("✅ yolov6.assigners.iou2d_calculator导入成功")
        
        from yolov6.assigners.atss_assigner import ATSSAssigner
        print("✅ yolov6.assigners.atss_assigner导入成功")
        
        return True
    except Exception as e:
        print(f"❌ yolov6.assigners导入失败: {e}")
        traceback.print_exc()
        return False


def test_yolov6_models():
    """测试yolov6.models模块"""
    print("\n🔍 测试yolov6.models模块...")
    try:
        from yolov6.models.yolo import Model, build_model, build_network
        print("✅ yolov6.models.yolo导入成功")
        
        from yolov6.models.efficientrep import EfficientRep
        print("✅ yolov6.models.efficientrep导入成功")
        
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        print("✅ yolov6.models.effidehead导入成功")
        
        from yolov6.models.reppan import RepPANNeck
        print("✅ yolov6.models.reppan导入成功")
        
        return True
    except Exception as e:
        print(f"❌ yolov6.models导入失败: {e}")
        traceback.print_exc()
        return False


def test_yolov6_main():
    """测试yolov6主模块"""
    print("\n🔍 测试yolov6主模块...")
    try:
        import yolov6
        print("✅ yolov6主模块导入成功")
        
        print(f"📋 版本信息: {yolov6.get_version()}")
        print(f"📋 模型信息: {yolov6.get_model_info()}")
        
        return True
    except Exception as e:
        print(f"❌ yolov6主模块导入失败: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """测试基础功能"""
    print("\n🔍 测试基础功能...")
    try:
        import jittor as jt
        from yolov6.layers.common import Conv
        
        # 测试创建一个简单的Conv层
        conv = Conv(3, 64, 3, 1)
        print("✅ Conv层创建成功")
        
        # 测试前向传播
        x = jt.randn(1, 3, 224, 224)
        y = conv(x)
        print(f"✅ Conv层前向传播成功，输出形状: {y.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始GOLD-YOLO Jittor版本模块导入测试...")
    print("=" * 60)
    
    tests = [
        ("基础导入", test_basic_imports),
        ("yolov6.layers", test_yolov6_layers),
        ("yolov6.utils", test_yolov6_utils),
        ("yolov6.assigners", test_yolov6_assigners),
        ("yolov6.models", test_yolov6_models),
        ("yolov6主模块", test_yolov6_main),
        ("基础功能", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！GOLD-YOLO Jittor版本模块导入正常！")
        return True
    else:
        print("⚠️  部分测试失败，请检查依赖问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
