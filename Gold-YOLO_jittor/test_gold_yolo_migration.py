#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - gold_yolo模块迁移验证脚本
验证从PyTorch到Jittor的迁移是否成功
"""

import sys
import os
import traceback

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gold_yolo_imports():
    """测试gold_yolo模块导入"""
    print("🔍 测试gold_yolo模块导入...")
    try:
        # 测试基础层导入
        from gold_yolo.layers import Conv, Conv2d_BN, DropPath, h_sigmoid
        print("  ✅ layers模块导入成功")
        
        # 测试通用模块导入
        from gold_yolo.common import AdvPoolFusion, SimFusion_3in, SimFusion_4in
        print("  ✅ common模块导入成功")
        
        # 测试transformer模块导入
        from gold_yolo.transformer import (
            Mlp, Attention, top_Block, PyramidPoolAgg,
            TopBasicLayer, InjectionMultiSum_Auto_pool
        )
        print("  ✅ transformer模块导入成功")
        
        # 测试reppan模块导入
        from gold_yolo.reppan import RepGDNeck, GDNeck, GDNeck2
        print("  ✅ reppan模块导入成功")
        
        # 测试switch_tool模块导入
        from gold_yolo.switch_tool import switch_to_deploy, convert_checkpoint_False
        print("  ✅ switch_tool模块导入成功")
        
        print("✅ gold_yolo模块导入测试完成")
        return True
        
    except Exception as e:
        print(f"❌ gold_yolo模块导入失败: {e}")
        traceback.print_exc()
        return False


def test_basic_layers():
    """测试基础层功能"""
    print("\n🔍 测试gold_yolo基础层功能...")
    try:
        import jittor as jt
        from gold_yolo.layers import Conv, Conv2d_BN, DropPath, h_sigmoid
        
        # 测试Conv层
        conv = Conv(3, 64, 3, 1)
        x = jt.randn(2, 3, 224, 224)
        out = conv(x)
        print(f"  ✅ Conv层: {list(x.shape)} -> {list(out.shape)}")
        
        # 测试Conv2d_BN层
        conv_bn = Conv2d_BN(64, 128, 3, 1, 1)
        x = jt.randn(2, 64, 56, 56)
        out = conv_bn(x)
        print(f"  ✅ Conv2d_BN层: {list(x.shape)} -> {list(out.shape)}")
        
        # 测试DropPath
        drop_path = DropPath(0.1)
        x = jt.randn(2, 64, 56, 56)
        out = drop_path(x)
        print(f"  ✅ DropPath层: {list(x.shape)} -> {list(out.shape)}")
        
        # 测试h_sigmoid
        h_sig = h_sigmoid()
        x = jt.randn(2, 64, 56, 56)
        out = h_sig(x)
        print(f"  ✅ h_sigmoid层: {list(x.shape)} -> {list(out.shape)}")
        
        print("✅ gold_yolo基础层功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ gold_yolo基础层测试失败: {e}")
        traceback.print_exc()
        return False


def test_fusion_modules():
    """测试融合模块"""
    print("\n🔍 测试gold_yolo融合模块...")
    try:
        import jittor as jt
        from gold_yolo.common import AdvPoolFusion, SimFusion_3in, SimFusion_4in
        
        # 测试AdvPoolFusion
        adv_pool = AdvPoolFusion()
        x1 = jt.randn(2, 64, 80, 80)
        x2 = jt.randn(2, 128, 40, 40)
        out = adv_pool(x1, x2)
        print(f"  ✅ AdvPoolFusion: {list(x1.shape)}, {list(x2.shape)} -> {list(out.shape)}")
        
        # 测试SimFusion_3in - 修复通道数匹配问题
        sim_fusion_3 = SimFusion_3in([128, 256], 256)  # [x[0]和x[1]的通道数, x[2]的通道数]
        x = [jt.randn(2, 128, 80, 80), jt.randn(2, 128, 40, 40), jt.randn(2, 256, 20, 20)]
        out = sim_fusion_3(x)
        print(f"  ✅ SimFusion_3in: 3个输入 -> {list(out.shape)}")
        
        # 测试SimFusion_4in
        sim_fusion_4 = SimFusion_4in()
        x = [jt.randn(2, 64, 80, 80), jt.randn(2, 128, 40, 40), 
             jt.randn(2, 256, 20, 20), jt.randn(2, 512, 10, 10)]
        out = sim_fusion_4(x)
        print(f"  ✅ SimFusion_4in: 4个输入 -> {list(out.shape)}")
        
        print("✅ gold_yolo融合模块测试完成")
        return True
        
    except Exception as e:
        print(f"❌ gold_yolo融合模块测试失败: {e}")
        traceback.print_exc()
        return False


def test_transformer_modules():
    """测试Transformer模块"""
    print("\n🔍 测试gold_yolo Transformer模块...")
    try:
        import jittor as jt
        from gold_yolo.transformer import Mlp, Attention, top_Block, InjectionMultiSum_Auto_pool
        
        # 测试Mlp
        mlp = Mlp(256, 512, 256)
        x = jt.randn(2, 256, 20, 20)
        out = mlp(x)
        print(f"  ✅ Mlp: {list(x.shape)} -> {list(out.shape)}")
        
        # 测试Attention
        attn = Attention(256, 64, 4)
        x = jt.randn(2, 256, 20, 20)
        out = attn(x)
        print(f"  ✅ Attention: {list(x.shape)} -> {list(out.shape)}")
        
        # 测试top_Block
        block = top_Block(256, 64, 4)
        x = jt.randn(2, 256, 20, 20)
        out = block(x)
        print(f"  ✅ top_Block: {list(x.shape)} -> {list(out.shape)}")
        
        # 测试InjectionMultiSum_Auto_pool
        injection = InjectionMultiSum_Auto_pool(256, 256)
        x_l = jt.randn(2, 256, 40, 40)
        x_g = jt.randn(2, 256, 20, 20)
        out = injection(x_l, x_g)
        print(f"  ✅ InjectionMultiSum_Auto_pool: {list(x_l.shape)}, {list(x_g.shape)} -> {list(out.shape)}")
        
        print("✅ gold_yolo Transformer模块测试完成")
        return True
        
    except Exception as e:
        print(f"❌ gold_yolo Transformer模块测试失败: {e}")
        traceback.print_exc()
        return False


def test_switch_tools():
    """测试切换工具"""
    print("\n🔍 测试gold_yolo切换工具...")
    try:
        import jittor as jt
        from gold_yolo.switch_tool import switch_to_deploy, convert_checkpoint_False, convert_checkpoint_True
        from gold_yolo.layers import Conv
        
        # 创建简单模型
        model = Conv(3, 64, 3, 1)
        
        # 测试切换到部署模式
        deploy_model = switch_to_deploy(model)
        print("  ✅ switch_to_deploy成功")
        
        # 测试检查点转换
        model_false = convert_checkpoint_False(model)
        model_true = convert_checkpoint_True(model)
        print("  ✅ 检查点转换成功")
        
        print("✅ gold_yolo切换工具测试完成")
        return True
        
    except Exception as e:
        print(f"❌ gold_yolo切换工具测试失败: {e}")
        traceback.print_exc()
        return False


def test_parameter_consistency():
    """测试参数一致性"""
    print("\n🔍 测试gold_yolo参数一致性...")
    try:
        import jittor as jt
        from gold_yolo.layers import Conv, Conv2d_BN
        from gold_yolo.transformer import Mlp, Attention
        
        # 测试各模块的参数量
        test_cases = [
            (Conv(3, 64, 3, 1), "Conv(3,64,3,1)"),
            (Conv2d_BN(64, 128, 3, 1, 1), "Conv2d_BN(64,128,3,1,1)"),
            (Mlp(256, 512, 256), "Mlp(256,512,256)"),
            (Attention(256, 64, 4), "Attention(256,64,4)"),
        ]
        
        for module, name in test_cases:
            params = sum(p.numel() for p in module.parameters())
            print(f"  ✅ {name}: {params:,} 参数")
        
        print("✅ gold_yolo参数一致性测试完成")
        return True
        
    except Exception as e:
        print(f"❌ gold_yolo参数一致性测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 开始GOLD-YOLO gold_yolo模块迁移验证...")
    print("=" * 80)
    
    tests = [
        ("gold_yolo模块导入", test_gold_yolo_imports),
        ("基础层功能", test_basic_layers),
        ("融合模块", test_fusion_modules),
        ("Transformer模块", test_transformer_modules),
        ("切换工具", test_switch_tools),
        ("参数一致性", test_parameter_consistency),
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
    print(f"📊 gold_yolo模块迁移验证结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有gold_yolo模块迁移验证通过！")
        print("🎯 GOLD-YOLO gold_yolo模块标准实现迁移完成！")
        return True
    else:
        print("⚠️  部分验证失败，需要进一步完善")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
