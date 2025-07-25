
# Gold-YOLO Jittor版本使用指南

## 1. 环境准备
```bash
conda activate jt  # 激活Jittor环境
cd /path/to/Gold-YOLO_jittor
```

## 2. 模型使用
```python
from yolov6.models.exact_pytorch_matched_gold_yolo import build_exact_pytorch_matched_gold_yolo
import numpy as np
import jittor as jt

# 创建模型
model = build_exact_pytorch_matched_gold_yolo(num_classes=20)

# 加载权重
weights = np.load("weights/exact_matched_weights.npz")
jt_state_dict = {name: jt.array(weight) for name, weight in weights.items()}
model.load_state_dict(jt_state_dict)
model.eval()

# 推理
input_tensor = jt.randn(1, 3, 640, 640)
output = model(input_tensor)  # [1, 8400, 25]
```

## 3. 检测可视化
```bash
python simple_correct_inference.py  # 运行简化推理
```

## 4. 性能测试
```bash
python exact_weight_converter_and_tester.py  # 完整性能测试
```

## 5. 文件结构
- `yolov6/models/exact_pytorch_matched_gold_yolo.py` - 主模型文件
- `weights/exact_matched_weights.npz` - 转换后的权重
- `outputs/` - 输出结果和可视化
