#!/usr/bin/env python3
"""
GOLD-YOLO-n Jittor版本 - 点击即用训练脚本
与PyTorch版gold-yolo-n完全对齐的参数配置

使用方法:
1. 直接运行: python train.py
2. 自定义参数: python train.py --epochs 100 --batch-size 8
"""

import subprocess
import sys
import os

def main():
    """启动训练"""
    print("🚀 GOLD-YOLO-n Jittor版本 - 点击即用训练")
    print("=" * 50)
    
    # 检查是否在正确的目录
    if not os.path.exists('train_pytorch_aligned_stable.py'):
        print("❌ 请在Gold-YOLO_jittor目录下运行此脚本")
        return
    
    # 构建命令
    cmd = [sys.executable, 'train_pytorch_aligned_stable.py']
    
    # 添加命令行参数
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # 启动训练
        subprocess.run(cmd, check=True)
        print("✅ 训练完成")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
