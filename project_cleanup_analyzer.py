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


class ProjectCleanupAnalyzer:
    """项目清理分析器"""
    
    def __init__(self):
        """初始化"""
        self.project_root = Path(".")
        self.cleanup_report = {
            "duplicate_files": [],
            "obsolete_files": [],
            "temporary_files": [],
            "keep_files": [],
            "cleanup_actions": []
        }
        
        print("🧹 项目深度清理分析器")
        print("=" * 80)
    
    def analyze_duplicate_files(self):
        """分析重复文件"""
        print("\n🔍 分析重复文件")
        print("-" * 60)
        
        # 识别重复的推理文件
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
        
        # 识别重复的权重转换文件
        weight_converter_files = [
            "complete_weight_converter.py",
            "exact_weight_converter_and_tester.py",
            "pytorch_exact_weight_converter.py",
            "perfect_weight_converter_v2.py",
            "true_weight_converter.py",
            "final_weight_converter.py",
            "final_weight_optimizer.py"
        ]
        
        # 识别重复的架构分析文件
        architecture_files = [
            "deep_architecture_analyzer.py",
            "deep_architecture_diagnosis.py",
            "comprehensive_architecture_analyzer.py",
            "parameter_alignment_analyzer.py",
            "weight_structure_analyzer.py",
            "pytorch_weight_analyzer.py"
        ]
        
        # 识别重复的模型文件
        model_files = [
            "pytorch_aligned_model.py",
            "true_pytorch_matched_model.py",
            "architecture_aligned_backbone.py"
        ]
        
        duplicates = {
            "inference_files": inference_files,
            "weight_converter_files": weight_converter_files,
            "architecture_files": architecture_files,
            "model_files": model_files
        }
        
        for category, files in duplicates.items():
            existing_files = [f for f in files if os.path.exists(f)]
            if len(existing_files) > 1:
                self.cleanup_report["duplicate_files"].append({
                    "category": category,
                    "files": existing_files,
                    "keep": existing_files[-1],  # 保留最新的
                    "remove": existing_files[:-1]
                })
                
                print(f"   📁 {category}:")
                print(f"      保留: {existing_files[-1]}")
                print(f"      删除: {existing_files[:-1]}")
    
    def analyze_obsolete_files(self):
        """分析过时文件"""
        print(f"\n🗑️ 分析过时文件")
        print("-" * 60)
        
        # 过时的实验文件
        obsolete_patterns = [
            "*diagnosis*.py",
            "*emergency*.py", 
            "*precise*.py",
            "*ultimate*.py",
            "complete_architecture_rebuilder.py",
            "complete_neck_implementation.py",
            "complete_parameter_alignment.py",
            "detection_accuracy_validator.py",
            "detection_performance_analyzer.py",
            "final_detection_accuracy_test.py",
            "final_project_summary.py",
            "final_weight_conversion_and_inference.py",
            "pytorch_jittor_comparison.py",
            "visualization_results_viewer.py"
        ]
        
        obsolete_files = []
        for pattern in obsolete_patterns:
            obsolete_files.extend(glob.glob(pattern))
        
        self.cleanup_report["obsolete_files"] = obsolete_files
        
        print(f"   发现过时文件: {len(obsolete_files)}个")
        for file in obsolete_files:
            print(f"      {file}")
    
    def analyze_temporary_files(self):
        """分析临时文件"""
        print(f"\n🗂️ 分析临时文件")
        print("-" * 60)
        
        # 临时文件模式
        temp_patterns = [
            "*.json",  # 分析结果文件
            "__pycache__/*",
            "*.pyc",
            "*.log"
        ]
        
        temp_files = []
        for pattern in temp_patterns:
            temp_files.extend(glob.glob(pattern, recursive=True))
        
        # 过滤掉重要的配置文件
        important_json = [
            "configs/gold_yolo-n.py",
            "data/voc_gold_yolo_jittor.yaml"
        ]
        
        temp_files = [f for f in temp_files if f not in important_json]
        
        self.cleanup_report["temporary_files"] = temp_files
        
        print(f"   发现临时文件: {len(temp_files)}个")
        for file in temp_files[:10]:  # 只显示前10个
            print(f"      {file}")
        if len(temp_files) > 10:
            print(f"      ... 还有{len(temp_files)-10}个")
    
    def identify_keep_files(self):
        """识别需要保留的核心文件"""
        print(f"\n✅ 识别核心保留文件")
        print("-" * 60)
        
        # 核心文件
        core_files = [
            # 最终成功的模型和推理
            "final_smart_inference_test.py",
            "pytorch_aligned_model.py", 
            "smart_weight_matcher.py",
            "final_objectness_fixer.py",
            "architecture_aligned_backbone.py",
            
            # 核心模块
            "gold_yolo/",
            "yolov6/",
            "models/",
            "configs/",
            "data/",
            "scripts/",
            "tools/",
            
            # 重要权重
            "weights/pytorch_original_weights.npz",
            "weights/smart_matched_weights.npz",
            "weights/final_objectness_fixed_weights.npz",
            
            # 文档
            "README.md",
            "USAGE_GUIDE.md",
            "requirements.txt",
            "setup.py",
            
            # 最终成功的输出
            "outputs/final_smart_inference/",
            "outputs/architecture_analysis/"
        ]
        
        self.cleanup_report["keep_files"] = core_files
        
        print(f"   核心保留文件/目录: {len(core_files)}个")
        for file in core_files:
            print(f"      ✅ {file}")
    
    def generate_cleanup_actions(self):
        """生成清理操作"""
        print(f"\n🎯 生成清理操作")
        print("-" * 60)
        
        actions = []
        
        # 删除重复文件
        for dup_group in self.cleanup_report["duplicate_files"]:
            for file in dup_group["remove"]:
                actions.append(f"rm {file}")
        
        # 删除过时文件
        for file in self.cleanup_report["obsolete_files"]:
            actions.append(f"rm {file}")
        
        # 清理临时文件
        actions.extend([
            "rm -rf __pycache__",
            "find . -name '*.pyc' -delete",
            "rm -f *.json",  # 删除分析结果文件
        ])
        
        # 清理多余的权重文件
        weight_files_to_remove = [
            "weights/complete_matched_weights.npz",
            "weights/complete_rebuilt_weights.npz", 
            "weights/emergency_fixed_weights.npz",
            "weights/exact_matched_weights.npz",
            "weights/final_exact_weights.npz",
            "weights/final_optimized_weights.npz",
            "weights/precisely_fixed_weights.npz",
            "weights/pytorch_exact_weights.npz",
            "weights/pytorch_weights.npz",
            "weights/strictly_aligned_weights.npz",
            "weights/ultimate_fixed_weights.npz"
        ]
        
        for weight_file in weight_files_to_remove:
            actions.append(f"rm -f {weight_file}")
        
        # 清理多余的输出目录
        output_dirs_to_remove = [
            "outputs/detection_validation",
            "outputs/detection_visualizations",
            "outputs/final_aligned_inference",
            "outputs/final_objectness_fix", 
            "outputs/final_optimization",
            "outputs/final_visualization_showcase",
            "outputs/inference_results",
            "outputs/pytorch_exact_inference",
            "outputs/pytorch_style_inference",
            "outputs/simple_correct_inference",
            "outputs/visualizations"
        ]
        
        for output_dir in output_dirs_to_remove:
            actions.append(f"rm -rf {output_dir}")
        
        self.cleanup_report["cleanup_actions"] = actions
        
        print(f"   生成清理操作: {len(actions)}个")
        print(f"   预计释放空间: 大量临时文件和重复文件")
    
    def create_cleanup_script(self):
        """创建清理脚本"""
        print(f"\n📝 创建清理脚本")
        print("-" * 60)
        
        script_content = """#!/bin/bash
# Gold-YOLO Jittor项目清理脚本
# 自动生成，清理重复和无用文件

echo "🧹 开始清理Gold-YOLO Jittor项目..."

# 备份重要文件
echo "📦 备份重要文件..."
mkdir -p backup
cp weights/pytorch_original_weights.npz backup/
cp weights/smart_matched_weights.npz backup/
cp weights/final_objectness_fixed_weights.npz backup/

# 清理重复文件
echo "🗑️ 清理重复文件..."
"""
        
        for action in self.cleanup_report["cleanup_actions"]:
            script_content += f"{action}\n"
        
        script_content += """
# 重新组织目录结构
echo "📁 重新组织目录结构..."
mkdir -p core_files
mv final_smart_inference_test.py core_files/
mv pytorch_aligned_model.py core_files/
mv smart_weight_matcher.py core_files/
mv final_objectness_fixer.py core_files/
mv architecture_aligned_backbone.py core_files/

echo "✅ 清理完成!"
echo "📊 清理统计:"
echo "   - 删除重复文件: 多个"
echo "   - 删除过时文件: 多个" 
echo "   - 清理临时文件: 多个"
echo "   - 保留核心文件: 重要文件已保留"
echo ""
echo "🎯 项目现在更加整洁!"
"""
        
        with open("cleanup_project.sh", "w") as f:
            f.write(script_content)
        
        os.chmod("cleanup_project.sh", 0o755)
        
        print(f"   ✅ 清理脚本已创建: cleanup_project.sh")
        print(f"   运行方式: bash cleanup_project.sh")
    
    def save_cleanup_report(self):
        """保存清理报告"""
        with open("cleanup_report.json", "w") as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        print(f"\n💾 清理报告已保存: cleanup_report.json")
    
    def run_analysis(self):
        """运行完整分析"""
        print("🧹 运行项目深度清理分析")
        print("=" * 80)
        
        self.analyze_duplicate_files()
        self.analyze_obsolete_files() 
        self.analyze_temporary_files()
        self.identify_keep_files()
        self.generate_cleanup_actions()
        self.create_cleanup_script()
        self.save_cleanup_report()
        
        print(f"\n🎉 项目清理分析完成!")
        print("=" * 80)
        
        # 统计
        total_duplicates = sum(len(group["remove"]) for group in self.cleanup_report["duplicate_files"])
        total_obsolete = len(self.cleanup_report["obsolete_files"])
        total_temp = len(self.cleanup_report["temporary_files"])
        
        print(f"📊 清理统计:")
        print(f"   重复文件: {total_duplicates}个")
        print(f"   过时文件: {total_obsolete}个")
        print(f"   临时文件: {total_temp}个")
        print(f"   清理操作: {len(self.cleanup_report['cleanup_actions'])}个")
        
        print(f"\n🚀 建议操作:")
        print(f"   1. 查看清理报告: cleanup_report.json")
        print(f"   2. 执行清理脚本: bash cleanup_project.sh")
        print(f"   3. 验证核心功能: python final_smart_inference_test.py")


def main():
    """主函数"""
    analyzer = ProjectCleanupAnalyzer()
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
