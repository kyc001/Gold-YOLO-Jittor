#!/usr/bin/env python3
"""
GPU最大负载测试 - 智能调节负载以跑满GPU
"""

import os
import time
import threading
import argparse

# 设置GPU环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JT_SYNC'] = '0'
os.environ['JT_CUDA_MEMORY_POOL'] = '1'

import jittor as jt

# 强制GPU模式
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0

class GPULoadBalancer:
    def __init__(self):
        self.current_load = 1000  # 初始矩阵大小
        self.target_gpu_util = 95  # 目标GPU利用率
        self.max_memory_usage = 0.8  # 最大内存使用率
        
    def get_gpu_status(self):
        """获取GPU状态"""
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                gpu_util = int(data[0])
                mem_used = int(data[1])
                mem_total = int(data[2])
                mem_usage = mem_used / mem_total
                return gpu_util, mem_usage
        except:
            pass
        return 0, 0
    
    def adjust_load(self, gpu_util, mem_usage):
        """动态调整负载"""
        if mem_usage > self.max_memory_usage:
            # 内存使用过高，减少负载
            self.current_load = max(500, int(self.current_load * 0.9))
            print(f"🔽 内存过高({mem_usage:.1%})，减少负载到 {self.current_load}")
        elif gpu_util < self.target_gpu_util - 10:
            # GPU利用率太低，增加负载
            self.current_load = min(4000, int(self.current_load * 1.1))
            print(f"🔼 GPU利用率低({gpu_util}%)，增加负载到 {self.current_load}")
        elif gpu_util > self.target_gpu_util + 5:
            # GPU利用率过高，稍微减少负载
            self.current_load = max(500, int(self.current_load * 0.95))
            print(f"🔽 GPU利用率过高({gpu_util}%)，减少负载到 {self.current_load}")
        
        return self.current_load

def gpu_intensive_task(load_balancer, duration=60):
    """GPU密集计算任务"""
    print(f"🔥 启动GPU密集计算任务 ({duration}秒)")
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        try:
            # 获取当前负载大小
            matrix_size = load_balancer.current_load
            
            # 矩阵运算
            a = jt.randn(matrix_size, matrix_size)
            b = jt.randn(matrix_size, matrix_size)
            c = jt.matmul(a, b)
            
            # 卷积运算
            batch_size = min(32, max(4, 8000 // matrix_size))
            channels = min(128, max(32, 4000 // matrix_size))
            size = min(256, max(64, 16000 // matrix_size))
            
            conv_input = jt.randn(batch_size, channels, size, size)
            conv = jt.nn.Conv2d(channels, channels, 3, padding=1)
            conv_output = conv(conv_input)
            
            # 激活函数
            relu_output = jt.nn.relu(conv_output)
            
            # 强制计算完成
            result1 = float(jt.sum(c))
            result2 = float(jt.sum(relu_output))
            
            iteration += 1
            
            # 每10次迭代调整一次负载
            if iteration % 10 == 0:
                gpu_util, mem_usage = load_balancer.get_gpu_status()
                new_load = load_balancer.adjust_load(gpu_util, mem_usage)
                
                elapsed = time.time() - start_time
                print(f"   迭代 {iteration}: GPU {gpu_util}% | 内存 {mem_usage:.1%} | 负载 {new_load} | {elapsed:.1f}s")
        
        except Exception as e:
            if "memory" in str(e).lower() or "overflow" in str(e).lower():
                # 内存不足，减少负载
                load_balancer.current_load = max(500, int(load_balancer.current_load * 0.8))
                print(f"⚠️ 内存不足，减少负载到 {load_balancer.current_load}")
            else:
                print(f"⚠️ 计算错误: {e}")
            time.sleep(1)
    
    total_time = time.time() - start_time
    print(f"✅ GPU任务完成: {iteration} 次迭代, 总时间: {total_time:.1f}s")
    return iteration

def monitor_gpu_detailed(duration=60):
    """详细监控GPU状态"""
    print("📊 开始详细GPU监控...")
    
    start_time = time.time()
    max_util = 0
    max_memory = 0
    
    while time.time() - start_time < duration:
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                gpu_util = int(data[0])
                mem_used = int(data[1])
                mem_total = int(data[2])
                temp = int(data[3])
                power = float(data[4])
                
                max_util = max(max_util, gpu_util)
                mem_percent = (mem_used / mem_total) * 100
                max_memory = max(max_memory, mem_percent)
                
                print(f"🔥 GPU: {gpu_util:3d}% | 内存: {mem_percent:5.1f}% | 温度: {temp:2d}°C | 功耗: {power:5.1f}W")
            
        except Exception as e:
            print(f"⚠️ 监控错误: {e}")
        
        time.sleep(3)
    
    print(f"📊 监控完成 - 最大GPU利用率: {max_util}% | 最大内存使用: {max_memory:.1f}%")
    return max_util, max_memory

def main():
    parser = argparse.ArgumentParser(description='GPU最大负载测试')
    parser.add_argument('--duration', type=int, default=120, help='测试时长(秒)')
    parser.add_argument('--target-util', type=int, default=95, help='目标GPU利用率(%)')
    
    args = parser.parse_args()
    
    print(f"🚀 GPU最大负载测试")
    print(f"   Jittor版本: {jt.__version__}")
    print(f"   GPU可用: {jt.has_cuda}")
    print(f"   目标利用率: {args.target_util}%")
    print(f"   测试时长: {args.duration}秒")
    print("=" * 60)
    
    # 创建负载均衡器
    load_balancer = GPULoadBalancer()
    load_balancer.target_gpu_util = args.target_util
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_gpu_detailed, args=(args.duration,))
    monitor_thread.start()
    
    # 执行GPU密集任务
    iterations = gpu_intensive_task(load_balancer, args.duration)
    
    # 等待监控完成
    monitor_thread.join()
    
    print(f"\n🎉 GPU最大负载测试完成!")
    print(f"   总迭代次数: {iterations}")
    print(f"   平均迭代速度: {iterations/args.duration:.2f} iter/s")

if __name__ == "__main__":
    main()
