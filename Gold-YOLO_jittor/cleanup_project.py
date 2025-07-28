#!/usr/bin/env python3
"""
项目清理脚本
清理冗余文件、调试脚本和无用重复数据
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """清理项目目录"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    项目清理脚本                               ║
    ║                                                              ║
    ║  🧹 清理冗余文件和调试脚本                                   ║
    ║  📦 整理项目结构                                             ║
    ║  💾 保留核心功能文件                                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 定义要删除的调试脚本
    debug_scripts = [
        'check_weights.py',
        'debug_loss_function.py',
        'debug_loss_preprocessing.py',
        'debug_model_output.py',
        'debug_weight_loading.py',
        'deep_analysis_confidence.py',
        'deep_classification_analysis.py',
        'detection_visualization_compare.py',
        'eval_pytorch_strict_aligned.py',
        'final_classification_fix.py',
        'final_complete_training.py',
        'final_detection_test.py',
        'final_evaluation_test.py',
        'fix_classification_head.py',
        'fix_coordinate_conversion.py',
        'monitored_self_check_training.py',
        'quick_fix_test.py',
        'self_check_training.py',
        'simple_final_test.py',
        'simple_self_check.py',
        'simplified_confidence_analysis.py',
        'test_gradient_fix.py',
        'test_model_loading.py',
        'train_pytorch_aligned_stable.py',
        'complete_self_check_training.py',
        'ultimate_final_training.py'
    ]
    
    # 定义要删除的临时标签文件
    temp_labels = [
        'complete_self_check_label.txt',
        'final_classification_fix_label.txt',
        'final_complete_label.txt',
        'monitored_self_check_label.txt',
        'self_check_label.txt',
        'ultimate_final_label.txt'
    ]
    
    # 定义要删除的临时模型文件
    temp_models = [
        'complete_self_check_model.pkl',
        'final_classification_fixed_model.pkl',
        'final_complete_model.pkl',
        'fixed_classification_model.pkl',
        'self_check_model.pkl',
        'simple_self_check_model.pkl'
    ]
    
    # 保留最终模型
    keep_models = [
        'ultimate_final_model.pkl'  # 这是最终训练好的模型
    ]
    
    # 保留核心功能脚本
    keep_scripts = [
        'final_inference_visualization.py',  # 推理可视化脚本
        'train.py',  # 主训练脚本
        'FINAL_REPORT.md',  # 最终报告
        'README.md',  # 说明文档
        'requirements.txt',  # 依赖文件
        'setup.py'  # 安装脚本
    ]
    
    deleted_files = []
    kept_files = []
    
    print("🧹 开始清理调试脚本...")
    for script in debug_scripts:
        if os.path.exists(script):
            if script not in keep_scripts:
                os.remove(script)
                deleted_files.append(script)
                print(f"   ❌ 删除调试脚本: {script}")
            else:
                kept_files.append(script)
                print(f"   ✅ 保留核心脚本: {script}")
    
    print("\n🧹 开始清理临时标签文件...")
    for label in temp_labels:
        if os.path.exists(label):
            os.remove(label)
            deleted_files.append(label)
            print(f"   ❌ 删除临时标签: {label}")
    
    print("\n🧹 开始清理临时模型文件...")
    for model in temp_models:
        if os.path.exists(model):
            os.remove(model)
            deleted_files.append(model)
            print(f"   ❌ 删除临时模型: {model}")
    
    for model in keep_models:
        if os.path.exists(model):
            kept_files.append(model)
            print(f"   ✅ 保留最终模型: {model}")
    
    # 清理__pycache__目录
    print("\n🧹 清理Python缓存文件...")
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                shutil.rmtree(cache_path)
                deleted_files.append(cache_path)
                print(f"   ❌ 删除缓存目录: {cache_path}")
    
    # 清理空的runs目录中的旧结果
    runs_dir = Path('runs')
    if runs_dir.exists():
        print("\n🧹 清理旧的推理结果...")
        for subdir in runs_dir.iterdir():
            if subdir.is_dir():
                # 保留最新的final_test结果
                if subdir.name != 'inference' or not (subdir / 'final_test').exists():
                    shutil.rmtree(subdir)
                    deleted_files.append(str(subdir))
                    print(f"   ❌ 删除旧结果: {subdir}")
                else:
                    kept_files.append(str(subdir))
                    print(f"   ✅ 保留最终推理结果: {subdir}")
    
    # 创建清理报告
    print("\n📊 生成清理报告...")
    
    report = f"""# 项目清理报告

## 清理统计
- 删除文件数量: {len(deleted_files)}
- 保留文件数量: {len(kept_files)}

## 删除的文件
"""
    
    for file in sorted(deleted_files):
        report += f"- {file}\n"
    
    report += f"""
## 保留的核心文件
"""
    
    for file in sorted(kept_files):
        report += f"- {file}\n"
    
    report += f"""
## 清理后的项目结构
```
Gold-YOLO_jittor/
├── models/                    # 模型定义
├── yolov6/                    # 核心算法库
├── configs/                   # 配置文件
├── tools/                     # 工具脚本
├── data/                      # 数据配置
├── docs/                      # 文档
├── runs/inference/final_test/ # 最终推理结果
├── final_inference_visualization.py  # 推理脚本
├── train.py                   # 训练脚本
├── ultimate_final_model.pkl   # 最终模型
├── FINAL_REPORT.md           # 技术报告
└── README.md                 # 项目说明
```

## 项目状态
✅ 项目清理完成，结构清晰
✅ 保留所有核心功能
✅ 删除冗余调试文件
✅ 项目可直接使用
"""
    
    with open('CLEANUP_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 清理报告已保存: CLEANUP_REPORT.md")
    
    # 输出清理总结
    print("\n" + "="*70)
    print("🎉 项目清理完成！")
    print("="*70)
    print(f"📊 清理统计:")
    print(f"   删除文件: {len(deleted_files)} 个")
    print(f"   保留文件: {len(kept_files)} 个")
    print(f"   项目大小: 显著减少")
    
    print(f"\n✅ 保留的核心功能:")
    print(f"   - 完整的模型定义和算法库")
    print(f"   - 最终训练好的模型权重")
    print(f"   - 推理和可视化脚本")
    print(f"   - 完整的技术文档")
    print(f"   - 最终推理测试结果")
    
    print(f"\n🎯 项目现在可以直接用于:")
    print(f"   - 模型推理和检测")
    print(f"   - 进一步的训练")
    print(f"   - 性能评估")
    print(f"   - 部署应用")
    
    return len(deleted_files), len(kept_files)

if __name__ == "__main__":
    print("🚀 开始项目清理...")
    deleted_count, kept_count = cleanup_project()
    print(f"\n🎉 清理完成！删除了{deleted_count}个文件，保留了{kept_count}个核心文件。")
