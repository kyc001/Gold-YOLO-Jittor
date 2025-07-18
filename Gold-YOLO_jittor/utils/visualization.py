#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


class Visualizer:
    """可视化工具类"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_training_curves(self, train_losses, val_metrics, save_path):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gold-YOLO Training Curves', fontsize=16)
        
        epochs = range(1, len(train_losses) + 1)
        
        # 训练损失
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # mAP@0.5
        if val_metrics:
            val_epochs = range(1, len(val_metrics) + 1)
            map_50 = [m['mAP@0.5'] for m in val_metrics]
            axes[0, 1].plot(val_epochs, map_50, 'r-', label='mAP@0.5')
            axes[0, 1].set_title('mAP@0.5')
            axes[0, 1].set_xlabel('Validation Epoch')
            axes[0, 1].set_ylabel('mAP@0.5')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Precision & Recall
        if val_metrics:
            precision = [m['precision'] for m in val_metrics]
            recall = [m['recall'] for m in val_metrics]
            axes[1, 0].plot(val_epochs, precision, 'g-', label='Precision')
            axes[1, 0].plot(val_epochs, recall, 'orange', label='Recall')
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].set_xlabel('Validation Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # mAP@0.5:0.95
        if val_metrics:
            map_50_95 = [m['mAP@0.5:0.95'] for m in val_metrics]
            axes[1, 1].plot(val_epochs, map_50_95, 'purple', label='mAP@0.5:0.95')
            axes[1, 1].set_title('mAP@0.5:0.95')
            axes[1, 1].set_xlabel('Validation Epoch')
            axes[1, 1].set_ylabel('mAP@0.5:0.95')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison_chart(self, jittor_results, pytorch_results, save_path):
        """绘制Jittor vs PyTorch对比图"""
        if not pytorch_results:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Jittor vs PyTorch Comparison', fontsize=16)
        
        # 速度对比
        frameworks = ['Jittor', 'PyTorch']
        fps_values = [
            jittor_results['speed']['fps'],
            pytorch_results['speed']['fps']
        ]
        
        axes[0].bar(frameworks, fps_values, color=['blue', 'orange'])
        axes[0].set_title('Inference Speed (FPS)')
        axes[0].set_ylabel('FPS')
        for i, v in enumerate(fps_values):
            axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
        
        # 精度对比
        metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
        jittor_values = [
            jittor_results['accuracy']['mAP@0.5'],
            jittor_results['accuracy']['mAP@0.5:0.95'],
            jittor_results['accuracy']['precision'],
            jittor_results['accuracy']['recall']
        ]
        pytorch_values = [
            pytorch_results['accuracy']['mAP@0.5'],
            pytorch_results['accuracy']['mAP@0.5:0.95'],
            pytorch_results['accuracy']['precision'],
            pytorch_results['accuracy']['recall']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1].bar(x - width/2, jittor_values, width, label='Jittor', color='blue')
        axes[1].bar(x + width/2, pytorch_values, width, label='PyTorch', color='orange')
        axes[1].set_title('Accuracy Metrics')
        axes[1].set_ylabel('Score')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(metrics, rotation=45)
        axes[1].legend()
        
        # 显存使用对比（如果有数据）
        if 'memory' in jittor_results and 'memory' in pytorch_results:
            batch_sizes = [1, 2, 4, 6, 8]
            jittor_memory = []
            pytorch_memory = []
            
            for bs in batch_sizes:
                key = f'batch_{bs}'
                if key in jittor_results['memory'] and jittor_results['memory'][key]['success']:
                    jittor_memory.append(jittor_results['memory'][key]['memory_mb'])
                else:
                    jittor_memory.append(0)
                
                if key in pytorch_results['memory'] and pytorch_results['memory'][key]['success']:
                    pytorch_memory.append(pytorch_results['memory'][key]['memory_mb'])
                else:
                    pytorch_memory.append(0)
            
            axes[2].plot(batch_sizes, jittor_memory, 'b-o', label='Jittor')
            axes[2].plot(batch_sizes, pytorch_memory, 'r-o', label='PyTorch')
            axes[2].set_title('Memory Usage')
            axes[2].set_xlabel('Batch Size')
            axes[2].set_ylabel('Memory (MB)')
            axes[2].legend()
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_training_report(self, log_file, output_dir):
        """生成训练报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载训练日志
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # 绘制训练曲线
        self.plot_training_curves(
            log_data['train_losses'],
            log_data['val_metrics'],
            output_dir / 'training_curves.png'
        )
        
        # 生成HTML报告
        html_content = self._generate_html_report(log_data)
        with open(output_dir / 'training_report.html', 'w') as f:
            f.write(html_content)
    
    def _generate_html_report(self, log_data):
        """生成HTML格式的训练报告"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gold-YOLO Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 20px 0; }}
                .metrics {{ display: flex; justify-content: space-around; }}
                .metric {{ text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Gold-YOLO Jittor Training Report</h1>
                <p>Generated on: {log_data.get('timestamp', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Final Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Best mAP@0.5</h3>
                        <p>{log_data.get('best_map', 0):.4f}</p>
                    </div>
                    <div class="metric">
                        <h3>Total Epochs</h3>
                        <p>{log_data.get('total_epochs', 0)}</p>
                    </div>
                    <div class="metric">
                        <h3>Final Loss</h3>
                        <p>{log_data['train_losses'][-1] if log_data.get('train_losses') else 'N/A':.4f}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Training Curves</h2>
                <img src="training_curves.png" alt="Training Curves">
            </div>
            
            <div class="section">
                <h2>⚙️ Configuration</h2>
                <pre>{json.dumps(log_data.get('args', {}), indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        return html
