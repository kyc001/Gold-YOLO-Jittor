#!/usr/bin/env python3
"""
GPU满载压力测试脚本 - 跑满GPU性能
"""

import os
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor

# 设置GPU环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JT_SYNC'] = '0'
os.environ['JT_CUDA_MEMORY_POOL'] = '1'
os.environ['JT_ENABLE_TUNER'] = '1'

import jittor as jt

# 强制GPU模式
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0

def gpu_compute_task(task_id, duration=60, matrix_size=4000):
    """GPU计算任务"""
    print(f"🔥 启动GPU任务 {task_id} - 矩阵大小: {matrix_size}x{matrix_size}")
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        # 创建大矩阵
        a = jt.randn(matrix_size, matrix_size)
        b = jt.randn(matrix_size, matrix_size)
        
        # 矩阵乘法
        c = jt.matmul(a, b)
        
        # 更多GPU密集操作
        d = jt.nn.relu(c)
        e = jt.nn.softmax(d, dim=1)
        f = jt.sum(e)
        
        # 卷积操作
        conv_input = jt.randn(32, 64, 256, 256)
        conv = jt.nn.Conv2d(64, 128, 3, padding=1)
        conv_output = conv(conv_input)
        conv_result = jt.sum(conv_output)
        
        # 强制同步，确保计算完成
        result = float(f) + float(conv_result)
        
        iteration += 1
        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            print(f"   任务 {task_id}: {iteration} 次迭代, {elapsed:.1f}s, 结果: {result:.2f}")
    
    total_time = time.time() - start_time
    print(f"✅ 任务 {task_id} 完成: {iteration} 次迭代, 总时间: {total_time:.1f}s")
    return iteration

def monitor_gpu_usage(duration=60):
    """监控GPU使用率"""
    print("📊 开始监控GPU使用率...")
    
    start_time = time.time()
    max_util = 0
    
    while time.time() - start_time < duration:
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                gpu_util = int(data[0])
                mem_used = int(data[1])
                mem_total = int(data[2])
                temp = int(data[3])
                
                max_util = max(max_util, gpu_util)
                mem_percent = (mem_used / mem_total) * 100
                
                print(f"🔥 GPU: {gpu_util}% | 内存: {mem_used}/{mem_total}MB ({mem_percent:.1f}%) | 温度: {temp}°C")
            
        except Exception as e:
            print(f"⚠️ 监控错误: {e}")
        
        time.sleep(2)
    
    print(f"📊 监控完成 - 最大GPU利用率: {max_util}%")
    return max_util

def stress_test_single_thread(duration=60):
    """单线程GPU压力测试"""
    print(f"\n🚀 单线程GPU压力测试 ({duration}秒)")
    print("=" * 50)
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(duration,))
    monitor_thread.start()
    
    # 执行GPU计算
    iterations = gpu_compute_task(1, duration, 4000)
    
    # 等待监控完成
    monitor_thread.join()
    
    print(f"✅ 单线程测试完成: {iterations} 次迭代")
    return iterations

def stress_test_multi_thread(duration=60, num_threads=4):
    """多线程GPU压力测试"""
    print(f"\n🚀 多线程GPU压力测试 ({num_threads}线程, {duration}秒)")
    print("=" * 50)
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(duration,))
    monitor_thread.start()
    
    # 启动多个GPU计算线程
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            matrix_size = 3000 + i * 200  # 不同大小的矩阵
            future = executor.submit(gpu_compute_task, i+1, duration, matrix_size)
            futures.append(future)
        
        # 等待所有任务完成
        total_iterations = 0
        for future in futures:
            iterations = future.result()
            total_iterations += iterations
    
    # 等待监控完成
    monitor_thread.join()
    
    print(f"✅ 多线程测试完成: 总计 {total_iterations} 次迭代")
    return total_iterations

def extreme_stress_test(duration=60):
    """极限GPU压力测试"""
    print(f"\n🔥 极限GPU压力测试 ({duration}秒)")
    print("=" * 50)
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(duration,))
    monitor_thread.start()
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        # 同时进行多种GPU密集操作
        tasks = []
        
        # 任务1: 大矩阵乘法
        a1 = jt.randn(5000, 5000)
        b1 = jt.randn(5000, 5000)
        c1 = jt.matmul(a1, b1)
        
        # 任务2: 深度卷积
        conv_input = jt.randn(64, 256, 512, 512)
        conv1 = jt.nn.Conv2d(256, 512, 3, padding=1)
        conv2 = jt.nn.Conv2d(512, 1024, 3, padding=1)
        x = conv1(conv_input)
        x = jt.nn.relu(x)
        x = conv2(x)
        
        # 任务3: 大规模softmax
        softmax_input = jt.randn(1000, 10000)
        softmax_output = jt.nn.softmax(softmax_input, dim=1)
        
        # 任务4: 批量归一化
        bn_input = jt.randn(128, 1024, 64, 64)
        bn = jt.nn.BatchNorm2d(1024)
        bn_output = bn(bn_input)
        
        # 强制同步所有计算
        result1 = float(jt.sum(c1))
        result2 = float(jt.sum(x))
        result3 = float(jt.sum(softmax_output))
        result4 = float(jt.sum(bn_output))
        
        total_result = result1 + result2 + result3 + result4
        
        iteration += 1
        if iteration % 5 == 0:
            elapsed = time.time() - start_time
            print(f"   极限测试: {iteration} 次迭代, {elapsed:.1f}s, 结果: {total_result:.2f}")
    
    # 等待监控完成
    monitor_thread.join()
    
    total_time = time.time() - start_time
    print(f"🔥 极限测试完成: {iteration} 次迭代, 总时间: {total_time:.1f}s")
    return iteration

def main():
    parser = argparse.ArgumentParser(description='GPU满载压力测试')
    parser.add_argument('--duration', type=int, default=60, help='测试时长(秒)')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['single', 'multi', 'extreme', 'all'],
                       help='测试模式')
    parser.add_argument('--threads', type=int, default=4, help='多线程数量')
    
    args = parser.parse_args()
    
    print(f"🚀 GPU满载压力测试")
    print(f"   Jittor版本: {jt.__version__}")
    print(f"   GPU可用: {jt.has_cuda}")
    print(f"   use_cuda: {jt.flags.use_cuda}")
    print(f"   测试时长: {args.duration}秒")
    
    if args.mode in ['single', 'all']:
        stress_test_single_thread(args.duration)
    
    if args.mode in ['multi', 'all']:
        stress_test_multi_thread(args.duration, args.threads)
    
    if args.mode in ['extreme', 'all']:
        extreme_stress_test(args.duration)
    
    print(f"\n🎉 所有GPU压力测试完成！")

if __name__ == "__main__":
    main()
