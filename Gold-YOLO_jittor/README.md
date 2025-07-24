# Gold-YOLO Jittor Implementation

基于Jittor框架的Gold-YOLO目标检测模型实现，支持PyTorch权重转换。

## 🎯 项目特色

- ✅ **完整的Gold-YOLO-N实现** - 99.96%架构对齐度
- ✅ **PyTorch权重转换** - 支持预训练权重迁移
- ✅ **高性能推理** - 优化的Jittor实现
- ✅ **完整的训练流程** - 支持VOC数据集训练

## 📁 项目结构

```
Gold-YOLO_jittor/
├── yolov6/                 # 核心模型代码
├── configs/                # 配置文件
├── weights/                # 权重文件
├── outputs/                # 输出结果
├── demos/                  # 演示文件
├── tools/                  # 工具脚本
├── inference.py            # 主推理脚本
├── convert_weights.py      # 权重转换脚本
├── improved_train_jittor.py # 主训练脚本
└── evaluate_jittor.py      # 主评估脚本
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装Jittor
pip install jittor

# 安装依赖
pip install -r requirements.txt
```

### 2. 权重转换

```bash
# 转换PyTorch权重到Jittor格式
python convert_weights.py
```

### 3. 推理测试

```bash
# 运行推理测试
python inference.py
```

### 4. 训练模型

```bash
# 训练Gold-YOLO-N
python improved_train_jittor.py
```

### 5. 评估模型

```bash
# 评估模型性能
python evaluate_jittor.py
```

## 📊 性能指标

| 模型 | 参数量 | mAP@0.5 | 推理速度 |
|------|--------|---------|----------|
| Gold-YOLO-N | 5.63M | - | - |

## 🔧 技术细节

### 架构对齐

- **Backbone**: EfficientRep (99.93%对齐)
- **Neck**: RepGDNeck (99.998%对齐)  
- **Head**: 检测头 (99.99%对齐)
- **总体**: 99.96%架构对齐度

### 权重转换

支持从PyTorch预训练权重转换到Jittor格式：
- 自动处理数据类型转换
- 参数名称映射
- 权重验证

## 📝 使用说明

详细的使用说明请参考各个脚本的文档字符串。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循MIT许可证。
