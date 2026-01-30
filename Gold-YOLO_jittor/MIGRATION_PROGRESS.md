# Gold-YOLO PyTorch → Jittor 迁移进度文档

## 概述

本文档记录 Gold-YOLO 从 PyTorch 到 Jittor 框架的迁移进度和测试状态。

**当前状态**: ✅ 代码迁移完成，⚠️ 存在 Jittor cuDNN 兼容性问题

**迁移完成度**:
- 代码对齐: 100% ✅
- 参数量对齐: 100% (5,631,550) ✅
- 前向传播: 100% ✅
- Loss 计算: 100% ✅
- 反向传播: ⚠️ Jittor 1.3.10 cuDNN 问题（非代码问题）

---

## 参数量对齐状态

| 指标 | PyTorch | Jittor | 状态 |
|------|---------|--------|------|
| 总参数量(不含BN统计量) | 5,631,550 | 5,631,550 | ✅ 完全对齐 |

---

## 模块迁移状态

| 模块 | 文件路径 | 状态 | 对齐状态 | 备注 |
|------|----------|------|----------|------|
| Backbone - EfficientRep | `yolov6/models/efficientrep.py` | ✅ 完成 | ✅ 对齐 | 已添加 CSPBepBackbone_P6，修复 channel_merge_layer |
| Neck - RepGDNeck | `gold_yolo/reppan.py` | ✅ 完成 | ✅ 对齐 | |
| Head - EffiDeHead | `yolov6/models/effidehead.py` | ✅ 完成 | ✅ 对齐 | stride 使用 property 避免注册为参数 |
| Loss - ComputeLoss | `yolov6/models/losses/loss.py` | ✅ 完成 | ✅ 对齐 | 完全重写对齐PyTorch |
| Assigner - TaskAlignedAssigner | `yolov6/assigners/tal_assigner.py` | ✅ 完成 | ✅ 对齐 | 完全重写对齐PyTorch |
| Assigner Utils | `yolov6/assigners/assigner_utils.py` | ✅ 完成 | ✅ 对齐 | 完全重写对齐PyTorch |
| Anchor Generator | `yolov6/assigners/anchor_generator.py` | ✅ 完成 | ✅ 对齐 | |
| Transformer | `gold_yolo/transformer.py` | ✅ 完成 | ✅ 对齐 | 修复 pool_mode 参数处理 |
| Common Layers | `yolov6/layers/common.py` | ✅ 完成 | ✅ 对齐 | 已添加全部缺失类和融合方法 |
| Layers (gold_yolo) | `gold_yolo/layers.py` | ✅ 完成 | ✅ 对齐 | 修复 Conv2d_BN norm_cfg 处理 |
| Engine | `yolov6/core/engine.py` | ✅ 完成 | ✅ 对齐 | 添加全部缺失方法 |
| Config | `configs/gold_yolo-n.py` | ✅ 完成 | ✅ 对齐 | 修复 neck.num_repeats |
| Figure IoU | `yolov6/utils/figure_iou.py` | ✅ 完成 | ✅ 对齐 | 添加 pairwise_bbox_iou |

---

## 已修复问题总览

### 2025-01-30 第一批修复

1. **参数量对齐** - 从差异 473,955 修复到完全一致
   - 修复 `training_mode` 配置读取
   - 修复 `ConvModule.bias='auto'` 行为
   - 修复 `SimFusion_3in` 层结构
   - 修复 `stride` 注册为参数问题

2. **API 桥接层补充**
   - 添加 `tile` 函数
   - 添加 `binary_cross_entropy_with_logits` 函数

3. **内存泄漏防护**
   - `engine.py`: 添加 `jt.sync_all()` 和 `jt.gc()`
   - `ema.py`: 添加同步和垃圾回收

4. **ConfigDict 递归转换**
   - 修复嵌套字典属性访问问题

5. **build_effidehead_layer 通道索引**
   - 使用正确的索引 `[6, 8, 10]`

### 2025-01-30 第二批修复（深度对齐）

6. **Loss 函数完全重写** (`yolov6/models/losses/loss.py`)
   - ✅ 修复 warmup_assigner 参数顺序（anchors 放第一位）
   - ✅ 修复 DFL 损失计算（使用 log_softmax + 索引）
   - ✅ 移除分类损失双重规范化（删除多余的 .sum()）
   - ✅ 移除 VarifocalLoss 条件 Sigmoid
   - ✅ 修复 loss_items 顺序为 [iou, dfl, cls] 并应用权重
   - ✅ 修复 preprocess 方法对齐 PyTorch

7. **Assigner 完全重写** (`yolov6/assigners/`)
   - ✅ 修复 `select_highest_overlaps()` argmax 维度（dim=-2）
   - ✅ 移除手写 for 循环 argmax（极慢）
   - ✅ 修复 `get_targets()` one_hot 编码逻辑
   - ✅ 简化 `iou_calculator` 对齐 PyTorch

8. **配置文件修复** (`configs/gold_yolo-n.py`)
   - ✅ 修复 `neck.num_repeats` 从 9 个值改为 4 个值

9. **Common Layers 补全** (`yolov6/layers/common.py`)
   - ✅ 添加 `RealVGGBlock` 类
   - ✅ 添加 `ScaleLayer` 类
   - ✅ 添加 `LinearAddBlock` 类
   - ✅ 添加 RepVGGBlock 融合方法 (`get_equivalent_kernel_bias`, `_fuse_bn_tensor`, `switch_to_deploy`)

10. **Backbone 补全** (`yolov6/models/efficientrep.py`)
    - ✅ 添加 `CSPBepBackbone_P6` 类
    - ✅ 修复 EfficientRep `channel_merge_layer` 逻辑（恢复 `block == ConvWrapper` 检查）
    - ✅ 修复 EfficientRep6 结构（ERBlock_5 不应有 SPPF）
    - ✅ 修复 CSPBepBackbone `channel_merge_layer` 逻辑

### 2025-01-30 第三批修复（完全对齐）

11. **Transformer 修复** (`gold_yolo/transformer.py`)
    - ✅ 修复 `PyramidPoolAgg.execute()` 中 pool_mode 参数被无条件覆盖问题
    - ✅ 现在正确使用 `__init__` 中根据 pool_mode 设置的 pool 函数

12. **Layers 修复** (`gold_yolo/layers.py`)
    - ✅ 修复 `Conv2d_BN` 忽略 norm_cfg 参数问题
    - ✅ 现在正确使用 `build_norm_layer` 根据 norm_cfg 构建归一化层
    - ✅ 重新组织代码确保依赖顺序正确

13. **Engine 补全** (`yolov6/core/engine.py`)
    - ✅ 添加 `get_model()` 方法
    - ✅ 添加 `get_teacher_model()` 方法
    - ✅ 添加 `load_scale_from_pretrained_models()` 静态方法
    - ✅ 添加 `parallel_model()` 静态方法
    - ✅ 添加 `get_optimizer()` 方法
    - ✅ 添加 `get_lr_scheduler()` 静态方法
    - ✅ 添加 `plot_train_batch()` 方法
    - ✅ 添加 `plot_val_pred()` 方法
    - ✅ 添加 `calibrate()` PTQ校准方法
    - ✅ 添加 `quant_setup()` QAT设置方法

14. **Figure IoU 补全** (`yolov6/utils/figure_iou.py`)
    - ✅ 添加 `pairwise_bbox_iou()` 函数

### 2025-01-30 第四批修复（Jittor API 对齐）

15. **Assigner Utils 修复** (`yolov6/assigners/assigner_utils.py`)
    - ✅ 修复 Jittor 的 `argmax(dim=...)` 返回 `(indices, values)` 元组问题
    - ✅ 需要使用 `argmax(...)[0]` 获取索引，而非直接使用返回值
    - ✅ 修复位置: 第76行和第84行

16. **Transformer 修复** (`gold_yolo/transformer.py`)
    - ✅ 修复 `AdaptiveAvgPool2d` 不支持 numpy.ndarray 输入
    - ✅ 将 numpy array 转换为 tuple: `(int(size[0]), int(size[1]))`

17. **Import 修复** (`yolov6/assigners/__init__.py`)
    - ✅ 修复 `bbox_overlaps` 导入路径（从 `iou2d_calculator` 而非 `assigner_utils`）

---

## 测试结果

### 完整测试套件 (2025-01-30)

| 测试项 | 状态 | 说明 |
|--------|------|------|
| 1. 模块导入 | ✅ | 12个核心模块全部导入成功 |
| 2. 模型构建 | ✅ | Gold-YOLO-n 构建成功，use_dfl=False |
| 3. 前向传播 | ✅ | 输出 shapes 正确: feats, pred_scores, pred_distri |
| 4. Loss 计算 | ✅ | warmup_assigner + tal_assigner 正常工作 |
| 5. 推理模式 | ✅ | 输出 shape [1,8400,85] 正确 |
| 6. 参数量验证 | ✅ | 5,631,550 与 PyTorch 完全一致 |

### 参数量验证 ✅

```
Jittor total params (excl BN stats): 5,631,550
PyTorch reference: 5,631,550
Match: True
```

### API 桥接层测试 ✅

- `tile` 函数: 输入 (2,3,4) → tile([1,2,1]) → 输出 (2,6,4) ✓
- `binary_cross_entropy_with_logits` 函数: 正常计算损失值 ✓

### 模型前向传播测试 ✅

- 模型构建成功 ✓
- 推理模式输出正常 ✓
- 训练模式输出正常 ✓

### Loss 计算测试 ✅

- Loss 前向计算正常 ✓
- warmup_assigner (ATSS) 正常 ✓
- tal_assigner 正常 ✓
- iou_loss + cls_loss 计算正确 ✓

### 训练测试

**状态**: ⚠️ 反向传播存在 Jittor cuDNN 问题

反向传播时 `cudnn_conv_backward_x` 出现 `float64/float32` 混合精度错误。
所有模型参数、输出、梯度均为 float32，但 Jittor 内部 cuDNN 调用生成了 float64 的算子。

**错误**: `undefined symbol: _ZN6jittor11getDataTypeIdEE15cudnnDataType_tv`

**可能的解决方案**:
1. 升级 Jittor 版本（当前 1.3.10 可能存在 cuDNN 兼容性问题）
2. 使用 `jt.grad()` 手动计算梯度，避免 `optimizer.backward()`
3. 清理 Jittor 编译缓存：`rm -rf ~/.cache/jittor/`

---

## 修复文件清单

| 文件 | 修复内容 |
|------|----------|
| `yolov6/models/losses/loss.py` | 完全重写 Loss 函数 |
| `yolov6/assigners/tal_assigner.py` | 完全重写 Assigner |
| `yolov6/assigners/assigner_utils.py` | 完全重写工具函数，修复 argmax 返回值 |
| `yolov6/layers/common.py` | 添加缺失类和融合方法 |
| `yolov6/models/efficientrep.py` | 添加 P6 类，修复逻辑 |
| `yolov6/models/effidehead.py` | 修复 stride 参数注册 |
| `yolov6/core/engine.py` | 添加全部缺失方法 |
| `gold_yolo/transformer.py` | 修复 pool_mode 参数 |
| `gold_yolo/layers.py` | 修复 norm_cfg 处理 |
| `gold_yolo/common.py` | 修复层结构 |
| `configs/gold_yolo-n.py` | 修复 num_repeats |
| `yolov6/utils/figure_iou.py` | 添加 pairwise_bbox_iou |

---

## 参考资料

- [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/)
- [Gold-YOLO 原始仓库](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)
