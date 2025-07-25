#!/bin/bash
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

rm -f "final_aligned_inference_test.py"
rm -f "pytorch_exact_inferer.py"
rm -f "pytorch_style_inference_visualizer.py"
rm -f "simple_correct_inference.py"
rm -f "strictly_aligned_inference.py"
rm -f "comprehensive_inference_test.py"
rm -f "fixed_gold_yolo_inference.py"
rm -f "final_gold_yolo_inference.py"
rm -f "complete_weight_converter.py"
rm -f "exact_weight_converter_and_tester.py"
rm -f "pytorch_exact_weight_converter.py"
rm -f "perfect_weight_converter_v2.py"
rm -f "true_weight_converter.py"
rm -f "final_weight_converter.py"
rm -f "final_weight_optimizer.py"
rm -f "deep_architecture_analyzer.py"
rm -f "deep_architecture_diagnosis.py"
rm -f "comprehensive_architecture_analyzer.py"
rm -f "parameter_alignment_analyzer.py"
rm -f "weight_structure_analyzer.py"
rm -f "pytorch_weight_analyzer.py"
rm -f "analyze_pytorch_weights_structure.py"
rm -f "complete_architecture_rebuilder.py"
rm -f "complete_neck_implementation.py"
rm -f "complete_parameter_alignment.py"
rm -f "detection_accuracy_validator.py"
rm -f "detection_performance_analyzer.py"
rm -f "final_detection_accuracy_test.py"
rm -f "final_project_summary.py"
rm -f "final_weight_conversion_and_inference.py"
rm -f "pytorch_jittor_comparison.py"
rm -f "visualization_results_viewer.py"
rm -f "emergency_weight_fix.py"
rm -f "precise_architecture_fixer.py"
rm -f "ultimate_diagnosis_fixer.py"
rm -f "final_project_report.json"
rm -f "deep_architecture_analysis.json"
rm -f "final_detection_accuracy_results.json"
rm -f "weight_structure_analysis.json"
rm -f "pytorch_architecture_analysis.json"
rm -f "parameter_alignment_report.json"
rm -f "detection_performance_report.json"
rm -f "deep_weight_analysis.json"
rm -f "exact_weight_conversion_results.json"
rm -f "pytorch_weights_analysis.json"
rm -f "diagnosis_results.json"

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
