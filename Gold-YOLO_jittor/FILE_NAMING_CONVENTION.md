# Gold-YOLO Jittor版本文件命名规范

## 📋 核心原则
1. **不频繁创建新脚本** - 在现有文件基础上修改
2. **文件名对齐** - 与功能和模块对应
3. **版本控制** - 避免重复和混乱
4. **功能明确** - 文件名清晰表达功能

## 🏗️ 标准文件结构

### 核心模型文件
```
models/
├── gold_yolo_backbone.py      # Backbone模块
├── gold_yolo_neck.py          # Neck模块  
├── gold_yolo_detect.py        # 检测头模块
├── gold_yolo_model.py         # 完整模型
└── gold_yolo_utils.py         # 工具函数
```

### 权重处理文件
```
weights/
├── weight_converter.py        # 权重转换器
├── weight_matcher.py          # 权重匹配器
└── weight_validator.py        # 权重验证器
```

### 推理测试文件
```
inference/
├── inference_engine.py       # 推理引擎
├── inference_test.py          # 推理测试
└── inference_utils.py         # 推理工具
```

### 分析工具文件
```
analysis/
├── architecture_analyzer.py  # 架构分析器
├── performance_analyzer.py   # 性能分析器
└── comparison_analyzer.py     # 对比分析器
```

## 🎯 当前文件重命名计划

### 需要重命名的文件
1. `pytorch_aligned_model.py` → `models/gold_yolo_model.py`
2. `architecture_aligned_backbone.py` → `models/gold_yolo_backbone.py`
3. `smart_weight_matcher.py` → `weights/weight_matcher.py`
4. `final_objectness_fixer.py` → `models/gold_yolo_detect.py`
5. `final_smart_inference_test.py` → `inference/inference_test.py`

## 📁 目录结构规范
```
Gold-YOLO_jittor/
├── models/                    # 模型定义
├── weights/                   # 权重相关
├── inference/                 # 推理相关
├── analysis/                  # 分析工具
├── configs/                   # 配置文件
├── data/                      # 数据处理
├── outputs/                   # 输出结果
├── tests/                     # 测试文件
└── docs/                      # 文档
```

## 🔧 修改原则
1. **在现有文件基础上修改** - 不创建新文件
2. **保持功能完整性** - 重命名后功能不变
3. **更新导入路径** - 修改相关的import语句
4. **保持向后兼容** - 必要时创建软链接

## 📝 实施步骤
1. 创建标准目录结构
2. 移动并重命名核心文件
3. 更新所有import语句
4. 测试功能完整性
5. 更新文档和README
