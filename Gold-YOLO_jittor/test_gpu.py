#!/usr/bin/env python3
"""
GPU测试脚本 - 验证Jittor GPU是否正常工作
"""

import os
import jittor as jt

def test_gpu():
    """测试GPU功能"""
    print("=== Jittor GPU 测试 ===")
    
    # 强制启用GPU
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 0
    
    print(f"Jittor版本: {jt.__version__}")
    print(f"has_cuda: {jt.has_cuda}")
    print(f"use_cuda: {jt.flags.use_cuda}")
    print(f"lazy_execution: {jt.flags.lazy_execution}")
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['JT_SYNC'] = '0'
    
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # 测试GPU计算
    print("\n=== GPU计算测试 ===")
    
    # 创建大矩阵测试
    size = 2000
    print(f"创建 {size}x{size} 矩阵...")
    
    import time
    start_time = time.time()
    
    a = jt.randn(size, size)
    b = jt.randn(size, size)
    c = jt.matmul(a, b)
    result = c.sum()
    
    end_time = time.time()
    
    print(f"矩阵乘法完成")
    print(f"结果: {float(result)}")
    print(f"耗时: {end_time - start_time:.4f}秒")
    
    # 检查GPU使用情况
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util = result.stdout.strip()
            print(f"GPU利用率: {gpu_util}%")
        else:
            print("无法获取GPU利用率")
    except:
        print("nvidia-smi不可用")
    
    return end_time - start_time

if __name__ == "__main__":
    gpu_time = test_gpu()
    
    print("\n=== CPU对比测试 ===")
    jt.flags.use_cuda = 0
    
    import time
    start_time = time.time()
    
    a = jt.randn(2000, 2000)
    b = jt.randn(2000, 2000)
    c = jt.matmul(a, b)
    result = c.sum()
    
    cpu_time = time.time() - start_time
    
    print(f"CPU耗时: {cpu_time:.4f}秒")
    print(f"GPU加速比: {cpu_time/gpu_time:.2f}x")
