#!/usr/bin/env python3
"""
清理训练脚本中的调试信息，准备生产训练
"""

import re

def clean_debug_prints(file_path):
    """清理文件中的调试print语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 注释掉包含🔍的print语句
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if 'print(' in line and '🔍' in line:
            # 保持缩进，注释掉这行
            indent = len(line) - len(line.lstrip())
            cleaned_lines.append(' ' * indent + '# ' + line.strip())
        else:
            cleaned_lines.append(line)
    
    cleaned_content = '\n'.join(cleaned_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"✅ 已清理 {file_path}")

if __name__ == "__main__":
    # 清理主要文件
    files_to_clean = [
        'yolov6/models/losses.py',
        'train_pytorch_aligned_stable.py'
    ]
    
    for file_path in files_to_clean:
        try:
            clean_debug_prints(file_path)
        except Exception as e:
            print(f"❌ 清理 {file_path} 失败: {e}")
    
    print("🎯 调试信息清理完成！")
