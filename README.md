# Gold-YOLO (Jittor)

本仓库提供 Gold-YOLO 的 Jittor 迁移版本，并保留 PyTorch 参考实现。

- `Gold-YOLO_jittor`: Jittor 迁移版（主要工作目录）
- `Gold-YOLO_pytorch`: PyTorch 参考实现

## 当前状态（截至 2025-01-30）

- 代码迁移与参数量对齐已完成
- 前向传播与 Loss 计算已对齐
- 训练反向在 Jittor 1.3.10 + cuDNN 组合下存在兼容性问题（float64/float32 混合精度）
- 详细进度与排查记录见 `Gold-YOLO_jittor/MIGRATION_PROGRESS.md`

## 快速开始

以下命令以 `Gold-YOLO_jittor` 为工作目录。

### 1) 环境与依赖

```bash
# Python >= 3.7
pip install jittor  # 或者 pip install jittor[cuda]

pip install -r requirements.txt
```

### 2) 准备数据

准备一个 COCO/YOLO 风格的数据集，并提供数据配置 YAML（示例）：

```yaml
train: /path/to/train/images
val: /path/to/val/images
nc: 2
names: ["class0", "class1"]
```

> 将 `nc` 与 `names` 按你的数据集类别数替换。

### 3) 推理

```bash
python tools/infer.py \
  --weights /path/to/weights.pkl \
  --source /path/to/images \
  --yaml /path/to/data.yaml \
  --img-size 640 \
  --device cpu
```

### 4) 训练（当前建议 CPU）

```bash
python tools/train.py \
  --conf-file configs/gold_yolo-n.py \
  --data-path /path/to/data.yaml \
  --epochs 10 \
  --batch-size 4 \
  --device cpu
```

> 说明：GPU 训练在部分环境下可能触发 Jittor cuDNN 兼容问题，详见 `Gold-YOLO_jittor/MIGRATION_PROGRESS.md`。

### 5) 评估

```bash
python tools/eval.py \
  --data /path/to/data.yaml \
  --weights /path/to/weights.pkl \
  --batch-size 32
```

## 目录结构（核心）

```
Gold-YOLO_jittor/
├── configs/            # 模型配置
├── gold_yolo/          # Gold-YOLO 相关模块
├── tools/              # 训练/推理/评估脚本
├── yolov6/             # 核心训练与模型组件
└── MIGRATION_PROGRESS.md
```

## 贡献与许可

欢迎提交 Issue/PR 改进迁移流程与功能实现。

本项目遵循 MIT 许可证，详见 `LICENSE`。
