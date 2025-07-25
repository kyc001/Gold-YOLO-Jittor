#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
项目清理分析器
识别并清理无用文件，保持项目整洁
"""

import os
import glob
from pathlib import Path
import json


def analyze_project_files():
    """分析项目文件"""
    print("🧹 项目深度清理分析")
    print("=" * 80)
    
    # 重复的推理文件
    inference_files = [
        "final_aligned_inference_test.py",
        "final_smart_inference_test.py", 
        "pytorch_exact_inferer.py",
        "pytorch_style_inference_visualizer.py",
        "simple_correct_inference.py",
        "strictly_aligned_inference.py",
        "comprehensive_inference_test.py",
        "fixed_gold_yolo_inference.py",
        "final_gold_yolo_inference.py"
    ]
    
    # 重复的权重转换文件
    weight_converter_files = [
        "complete_weight_converter.py",
        "exact_weight_converter_and_tester.py",
        "pytorch_exact_weight_converter.py",
        "perfect_weight_converter_v2.py",
        "true_weight_converter.py",
        "final_weight_converter.py",
        "final_weight_optimizer.py"
    ]
    
    # 重复的架构分析文件
    architecture_files = [
        "deep_architecture_analyzer.py",
        "deep_architecture_diagnosis.py",
        "comprehensive_architecture_analyzer.py",
        "parameter_alignment_analyzer.py",
        "weight_structure_analyzer.py",
        "pytorch_weight_analyzer.py"
    ]
    
    # 过时的实验文件
    obsolete_files = [
        "analyze_pytorch_weights_structure.py",
        "complete_architecture_rebuilder.py",
        "complete_neck_implementation.py", 
        "complete_parameter_alignment.py",
        "detection_accuracy_validator.py",
        "detection_performance_analyzer.py",
        "final_detection_accuracy_test.py",
        "final_project_summary.py",
        "final_weight_conversion_and_inference.py",
        "pytorch_jittor_comparison.py",
        "visualization_results_viewer.py",
        "emergency_weight_fix.py",
        "precise_architecture_fixer.py",
        "ultimate_diagnosis_fixer.py"
    ]
    
    print("\n🔍 文件分析结果:")
    print("-" * 60)
    
    # 检查重复推理文件
    existing_inference = [f for f in inference_files if os.path.exists(f)]
    print(f"📁 重复推理文件: {len(existing_inference)}个")
    for f in existing_inference:
        print(f"   {f}")
    
    # 检查重复权重转换文件
    existing_converters = [f for f in weight_converter_files if os.path.exists(f)]
    print(f"\n📁 重复权重转换文件: {len(existing_converters)}个")
    for f in existing_converters:
        print(f"   {f}")
    
    # 检查重复架构分析文件
    existing_analyzers = [f for f in architecture_files if os.path.exists(f)]
    print(f"\n📁 重复架构分析文件: {len(existing_analyzers)}个")
    for f in existing_analyzers:
        print(f"   {f}")
    
    # 检查过时文件
    existing_obsolete = [f for f in obsolete_files if os.path.exists(f)]
    print(f"\n🗑️ 过时实验文件: {len(existing_obsolete)}个")
    for f in existing_obsolete:
        print(f"   {f}")
    
    # 检查JSON分析文件
    json_files = glob.glob("*.json")
    print(f"\n📄 临时JSON文件: {len(json_files)}个")
    for f in json_files:
        print(f"   {f}")
    
    # 检查权重文件
    weight_files = glob.glob("weights/*.npz")
    print(f"\n⚖️ 权重文件: {len(weight_files)}个")
    for f in weight_files:
        print(f"   {f}")
    
    return {
        "inference_files": existing_inference,
        "converter_files": existing_converters,
        "analyzer_files": existing_analyzers,
        "obsolete_files": existing_obsolete,
        "json_files": json_files,
        "weight_files": weight_files
    }


def create_cleanup_script(file_analysis):
    """创建清理脚本"""
    print(f"\n📝 创建清理脚本")
    print("-" * 60)
    
    script_content = """#!/bin/bash
# Gold-YOLO Jittor项目清理脚本

echo "🧹 开始清理Gold-YOLO Jittor项目..."

# 备份重要文件
echo "📦 备份重要文件..."
mkdir -p backup
cp weights/pytorch_original_weights.npz backup/ 2>/dev/null || true
cp weights/smart_matched_weights.npz backup/ 2>/dev/null || true
cp weights/final_objectness_fixed_weights.npz backup/ 2>/dev/null || true

# 保留最重要的文件
KEEP_FILES=(
    "final_smart_inference_test.py"
    "pytorch_aligned_model.py"
    "smart_weight_matcher.py"
    "final_objectness_fixer.py"
    "architecture_aligned_backbone.py"
)

echo "✅ 保留核心文件:"
for file in "${KEEP_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    fi
done

echo ""
echo "🗑️ 清理重复和过时文件..."

"""
    
    # 删除重复的推理文件（保留final_smart_inference_test.py）
    for f in file_analysis["inference_files"]:
        if f != "final_smart_inference_test.py":
            script_content += f'rm -f "{f}"\n'
    
    # 删除重复的转换文件（保留smart_weight_matcher.py）
    for f in file_analysis["converter_files"]:
        if f != "smart_weight_matcher.py":
            script_content += f'rm -f "{f}"\n'
    
    # 删除重复的分析文件（保留architecture_aligned_backbone.py）
    for f in file_analysis["analyzer_files"]:
        if f != "architecture_aligned_backbone.py":
            script_content += f'rm -f "{f}"\n'
    
    # 删除过时文件
    for f in file_analysis["obsolete_files"]:
        script_content += f'rm -f "{f}"\n'
    
    # 删除临时JSON文件
    for f in file_analysis["json_files"]:
        script_content += f'rm -f "{f}"\n'
    
    script_content += """
# 清理临时文件
echo "🧽 清理临时文件..."
rm -rf __pycache__
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# 清理多余的权重文件（保留核心权重）
echo "⚖️ 清理多余权重文件..."
cd weights
KEEP_WEIGHTS=(
    "pytorch_original_weights.npz"
    "smart_matched_weights.npz" 
    "final_objectness_fixed_weights.npz"
)

for weight_file in *.npz; do
    keep=false
    for keep_weight in "${KEEP_WEIGHTS[@]}"; do
        if [ "$weight_file" = "$keep_weight" ]; then
            keep=true
            break
        fi
    done
    
    if [ "$keep" = false ]; then
        echo "   删除: $weight_file"
        rm -f "$weight_file"
    else
        echo "   保留: $weight_file"
    fi
done
cd ..

# 清理多余的输出目录（保留重要结果）
echo "📁 清理输出目录..."
cd outputs
KEEP_OUTPUTS=(
    "final_smart_inference"
    "architecture_analysis"
)

for output_dir in */; do
    dir_name=${output_dir%/}
    keep=false
    for keep_output in "${KEEP_OUTPUTS[@]}"; do
        if [ "$dir_name" = "$keep_output" ]; then
            keep=true
            break
        fi
    done
    
    if [ "$keep" = false ]; then
        echo "   删除目录: $dir_name"
        rm -rf "$dir_name"
    else
        echo "   保留目录: $dir_name"
    fi
done
cd ..

echo ""
echo "✅ 清理完成!"
echo "📊 项目现在更加整洁，只保留核心文件:"
echo "   ✅ 最终成功的推理: final_smart_inference_test.py"
echo "   ✅ 对齐的模型: pytorch_aligned_model.py"
echo "   ✅ 智能权重匹配: smart_weight_matcher.py"
echo "   ✅ 目标置信度修复: final_objectness_fixer.py"
echo "   ✅ 架构对齐backbone: architecture_aligned_backbone.py"
echo "   ✅ 核心权重文件: 3个"
echo "   ✅ 重要输出结果: 2个目录"
echo ""
echo "🎯 项目清理完成，可以正常使用!"
"""
    
    with open("cleanup_project.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("cleanup_project.sh", 0o755)
    
    print(f"   ✅ 清理脚本已创建: cleanup_project.sh")


def main():
    """主函数"""
    file_analysis = analyze_project_files()
    create_cleanup_script(file_analysis)
    
    print(f"\n🎉 项目清理分析完成!")
    print("=" * 80)
    
    total_files = (len(file_analysis["inference_files"]) + 
                  len(file_analysis["converter_files"]) + 
                  len(file_analysis["analyzer_files"]) + 
                  len(file_analysis["obsolete_files"]) +
                  len(file_analysis["json_files"]))
    
    print(f"📊 清理统计:")
    print(f"   待清理文件: {total_files}个")
    print(f"   权重文件: {len(file_analysis['weight_files'])}个")
    
    print(f"\n🚀 执行清理:")
    print(f"   运行命令: bash cleanup_project.sh")
    print(f"   清理后验证: python final_smart_inference_test.py")


if __name__ == '__main__':
    main()
