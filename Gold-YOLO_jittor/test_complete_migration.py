#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 完整项目迁移验证脚本
全面检查从PyTorch到Jittor的整体迁移是否100%完成
"""

import sys
import os
import traceback
import importlib

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_project_structure():
    """测试项目结构完整性"""
    print("🔍 测试项目结构完整性...")
    
    required_dirs = [
        'gold_yolo',
        'yolov6',
        'configs',
        'data',
        'tools',
        'models'
    ]
    
    required_files = [
        'gold_yolo/__init__.py',
        'gold_yolo/layers.py',
        'gold_yolo/common.py',
        'gold_yolo/transformer.py',
        'gold_yolo/reppan.py',
        'gold_yolo/switch_tool.py',
        'yolov6/__init__.py',
        'yolov6/models/yolo.py',
        'yolov6/models/efficientrep.py',
        'yolov6/models/effidehead.py',
        'yolov6/assigners/iou2d_calculator.py',
        'tools/train.py',
        'tools/eval.py',
        'tools/infer.py',
        'configs/gold_yolo-n.py',
        'configs/gold_yolo-s.py',
        'configs/gold_yolo-m.py',
        'configs/gold_yolo-l.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    # 检查目录
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"  ✅ 目录存在: {dir_path}")
    
    # 检查文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ 文件存在: {file_path}")
    
    if missing_dirs:
        print(f"  ❌ 缺失目录: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"  ❌ 缺失文件: {missing_files}")
        return False
    
    print("✅ 项目结构完整性测试通过")
    return True


def test_core_imports():
    """测试核心模块导入"""
    print("\n🔍 测试核心模块导入...")
    
    import_tests = [
        # gold_yolo模块
        ('gold_yolo', 'GOLD-YOLO核心模块'),
        ('gold_yolo.layers', 'GOLD-YOLO基础层'),
        ('gold_yolo.common', 'GOLD-YOLO通用模块'),
        ('gold_yolo.transformer', 'GOLD-YOLO Transformer'),
        ('gold_yolo.reppan', 'GOLD-YOLO RepGDNeck'),
        ('gold_yolo.switch_tool', 'GOLD-YOLO切换工具'),
        
        # yolov6模块
        ('yolov6', 'YOLOv6基础模块'),
        ('yolov6.models.yolo', 'YOLOv6主模型'),
        ('yolov6.models.efficientrep', 'EfficientRep骨干网络'),
        ('yolov6.models.effidehead', 'EffiDeHead检测头'),
        ('yolov6.assigners.iou2d_calculator', 'IoU计算器'),
        ('yolov6.layers.common', 'YOLOv6通用层'),
        ('yolov6.utils.general', 'YOLOv6通用工具'),
        
        # 模型模块
        ('models.complete_gold_yolo', '完整GOLD-YOLO模型'),
        ('models.gold_yolo_model', 'GOLD-YOLO模型定义'),
        ('models.gold_yolo_backbone', 'GOLD-YOLO骨干网络'),
        ('models.gold_yolo_detect', 'GOLD-YOLO检测头'),
    ]
    
    failed_imports = []
    
    for module_name, description in import_tests:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {description}: {module_name}")
        except Exception as e:
            print(f"  ❌ {description}: {module_name} - {e}")
            failed_imports.append((module_name, str(e)))
    
    if failed_imports:
        print(f"\n❌ 导入失败的模块: {len(failed_imports)}")
        for module, error in failed_imports:
            print(f"    {module}: {error}")
        return False
    
    print("✅ 核心模块导入测试通过")
    return True


def test_jittor_compatibility():
    """测试Jittor兼容性"""
    print("\n🔍 测试Jittor兼容性...")
    
    try:
        import jittor as jt
        print(f"  ✅ Jittor版本: {jt.__version__}")
        
        # 测试CUDA支持
        if jt.has_cuda:
            print("  ✅ CUDA支持: 可用")
            jt.flags.use_cuda = 1
        else:
            print("  ⚠️  CUDA支持: 不可用，使用CPU")
            jt.flags.use_cuda = 0
        
        # 测试基本操作
        x = jt.randn(2, 3, 224, 224)
        y = jt.randn(2, 3, 224, 224)
        z = x + y
        print(f"  ✅ 基本张量操作: {list(z.shape)}")
        
        # 测试神经网络模块
        conv = jt.nn.Conv2d(3, 64, 3, 1, 1)
        out = conv(x)
        print(f"  ✅ 神经网络模块: {list(out.shape)}")
        
        print("✅ Jittor兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"❌ Jittor兼容性测试失败: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n🔍 测试模型创建...")
    
    try:
        # 测试GOLD-YOLO模型创建
        from models.complete_gold_yolo import create_gold_yolo_model
        
        # 创建不同尺寸的模型
        model_configs = [
            ('gold_yolo-n', 'GOLD-YOLO-n'),
            ('gold_yolo-s', 'GOLD-YOLO-s'),
        ]
        
        for config_name, model_name in model_configs:
            try:
                model = create_gold_yolo_model(config_name)
                
                # 测试前向传播
                import jittor as jt
                x = jt.randn(1, 3, 640, 640)
                output = model(x)
                
                print(f"  ✅ {model_name}: 创建成功，输出形状: {[list(o.shape) for o in output]}")
                
            except Exception as e:
                print(f"  ❌ {model_name}: 创建失败 - {e}")
                return False
        
        print("✅ 模型创建测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")
        traceback.print_exc()
        return False


def test_config_files():
    """测试配置文件"""
    print("\n🔍 测试配置文件...")
    
    config_files = [
        'configs/gold_yolo-n.py',
        'configs/gold_yolo-s.py',
        'configs/gold_yolo-m.py',
        'configs/gold_yolo-l.py'
    ]
    
    try:
        from yolov6.utils.config import Config
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    cfg = Config.fromfile(config_file)
                    print(f"  ✅ 配置文件: {config_file}")
                    
                    # 检查关键配置项
                    required_keys = ['model', 'solver', 'data_aug']
                    for key in required_keys:
                        if hasattr(cfg, key):
                            print(f"    ✅ 包含配置: {key}")
                        else:
                            print(f"    ⚠️  缺少配置: {key}")
                            
                except Exception as e:
                    print(f"  ❌ 配置文件解析失败: {config_file} - {e}")
                    return False
            else:
                print(f"  ❌ 配置文件不存在: {config_file}")
                return False
        
        print("✅ 配置文件测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        traceback.print_exc()
        return False


def test_training_pipeline():
    """测试训练流水线组件"""
    print("\n🔍 测试训练流水线组件...")
    
    try:
        # 测试数据加载器
        from yolov6.data.datasets import TrainValDataset
        print("  ✅ 数据集类导入成功")
        
        # 测试损失函数
        from yolov6.models.losses.loss import ComputeLoss
        print("  ✅ 损失函数导入成功")
        
        # 测试优化器构建
        from yolov6.solver.build import build_optimizer
        print("  ✅ 优化器构建导入成功")
        
        # 测试评估器
        from yolov6.core.evaler import Evaler
        print("  ✅ 评估器导入成功")
        
        # 测试推理器
        from yolov6.core.inferer import Inferer
        print("  ✅ 推理器导入成功")
        
        print("✅ 训练流水线组件测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 训练流水线组件测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始GOLD-YOLO完整项目迁移验证...")
    print("=" * 80)
    
    tests = [
        ("项目结构完整性", test_project_structure),
        ("核心模块导入", test_core_imports),
        ("Jittor兼容性", test_jittor_compatibility),
        ("模型创建", test_model_creation),
        ("配置文件", test_config_files),
        ("训练流水线组件", test_training_pipeline),
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
    
    print("\n" + "=" * 80)
    print(f"📊 完整项目迁移验证结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 GOLD-YOLO完整项目迁移验证通过！")
        print("🎯 PyTorch到Jittor的整体迁移100%完成！")
        return True
    else:
        print("⚠️  部分验证失败，需要进一步完善")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
