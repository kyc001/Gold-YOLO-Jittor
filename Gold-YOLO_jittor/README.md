# Gold-YOLO Jittor Implementation

新芽第二阶段：使用Jittor框架实现Gold-YOLO

## 🏗️ 项目结构

```
Gold-YOLO_jittor/
├── gold_yolo/              # 核心包
│   ├── models/             # 模型组件
│   │   ├── backbone.py     # 骨干网络
│   │   ├── neck.py         # 颈部网络
│   │   ├── head.py         # 检测头
│   │   └── gold_yolo.py    # 完整模型
│   ├── data/               # 数据处理
│   │   ├── dataset.py      # 数据集
│   │   ├── transforms.py   # 数据变换
│   │   └── dataloader.py   # 数据加载
│   ├── utils/              # 工具函数
│   │   ├── decoder.py      # YOLO解码器
│   │   ├── losses.py       # 损失函数
│   │   └── metrics.py      # 评估指标
│   ├── training/           # 训练组件
│   │   ├── trainer.py      # 训练器
│   │   └── validator.py    # 验证器
│   └── inference/          # 推理组件
│       ├── predictor.py    # 预测器
│       └── postprocess.py  # 后处理
├── scripts/                # 脚本
│   ├── train.py           # 训练脚本
│   ├── evaluate.py        # 评估脚本
│   └── inference.py       # 推理脚本
├── configs/                # 配置文件
├── tests/                  # 测试文件
└── tools/                  # 工具脚本
```

## 🚀 快速开始

### 安装

```bash
pip install -e .
```

### 训练

```bash
python scripts/train.py --config configs/gold_yolo_small.yaml
```

### 评估

```bash
python scripts/evaluate.py --weights runs/best.pkl
```

### 推理

```bash
python scripts/inference.py --weights runs/best.pkl --source image.jpg
```

## 📊 模型性能

- **参数量**: 8.5M
- **配置**: depth_multiple=0.33, width_multiple=0.5
- **输入尺寸**: 640x640
- **检测类别**: 80 (COCO)

## 🔧 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black gold_yolo/
isort gold_yolo/
```

## 📝 许可证

MIT License
