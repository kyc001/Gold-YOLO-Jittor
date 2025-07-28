#!/usr/bin/env python3
"""
简化的DFL模式测试 - 直接测试训练脚本的两种配置
"""

import os
import subprocess
import sys

def test_dfl_mode(use_dfl, reg_max, mode_name):
    """测试指定的DFL模式"""
    print(f"\n{'='*60}")
    print(f"🔍 测试 {mode_name}")
    print(f"   use_dfl={use_dfl}, reg_max={reg_max}")
    print(f"{'='*60}")
    
    # 修改配置文件
    config_file = "configs/gold_yolo-n.py"
    
    # 读取配置文件
    with open(config_file, 'r') as f:
        content = f.read()
    
    # 备份原始内容
    original_content = content
    
    try:
        # 修改配置
        if use_dfl:
            new_content = content.replace(
                "use_dfl=False,  # gold-yolo-n原始配置：禁用DFL",
                "use_dfl=True,   # 测试：启用DFL"
            ).replace(
                "reg_max=0,      # gold-yolo-n原始配置：reg_max=0",
                f"reg_max={reg_max},     # 测试：设置reg_max={reg_max}"
            )
        else:
            new_content = content.replace(
                "use_dfl=True,   # 测试：启用DFL",
                "use_dfl=False,  # 测试：禁用DFL"
            ).replace(
                f"reg_max={reg_max},     # 测试：设置reg_max={reg_max}",
                "reg_max=0,      # 测试：设置reg_max=0"
            )
        
        # 写入修改后的配置
        with open(config_file, 'w') as f:
            f.write(new_content)
        
        # 修改训练脚本
        train_file = "train_pytorch_aligned_stable.py"
        with open(train_file, 'r') as f:
            train_content = f.read()
        
        original_train_content = train_content
        
        if use_dfl:
            new_train_content = train_content.replace(
                "use_dfl=False,  # gold-yolo-n原始配置：禁用DFL",
                "use_dfl=True,   # 测试：启用DFL"
            ).replace(
                "reg_max=0,      # gold-yolo-n原始配置：reg_max=0",
                f"reg_max={reg_max},     # 测试：设置reg_max={reg_max}"
            )
        else:
            new_train_content = train_content.replace(
                "use_dfl=True,   # 测试：启用DFL",
                "use_dfl=False,  # 测试：禁用DFL"
            ).replace(
                f"reg_max={reg_max},     # 测试：设置reg_max={reg_max}",
                "reg_max=0,      # 测试：设置reg_max=0"
            )
        
        with open(train_file, 'w') as f:
            f.write(new_train_content)
        
        # 运行训练测试
        cmd = [
            "conda", "run", "-n", "yolo_jt", "timeout", "15",
            "python", "train_pytorch_aligned_stable.py",
            "--epochs", "1", "--batch-size", "1",
            "--lr-initial", "0.01", "--lr-final", "0.001",
            "--data", "../data/voc2012_subset/voc20.yaml"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        
        # 检查结果
        if result.returncode == 124:  # timeout成功
            # 检查输出中是否有损失值
            if "loss_cls原始值:" in result.stdout and "loss_iou原始值:" in result.stdout:
                print(f"✅ {mode_name} 测试成功！")
                
                # 提取损失值
                lines = result.stdout.split('\n')
                for line in lines:
                    if "loss_cls原始值:" in line or "loss_iou原始值:" in line or "loss_dfl原始值:" in line:
                        print(f"   {line.strip()}")
                
                return True
            else:
                print(f"❌ {mode_name} 测试失败！未找到损失值")
                return False
        else:
            print(f"❌ {mode_name} 测试失败！")
            print(f"   返回码: {result.returncode}")
            if result.stderr:
                print(f"   错误: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ {mode_name} 测试失败！")
        print(f"   异常: {str(e)}")
        return False
    
    finally:
        # 恢复原始配置
        with open(config_file, 'w') as f:
            f.write(original_content)
        with open(train_file, 'w') as f:
            f.write(original_train_content)

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
