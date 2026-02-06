# Gold-YOLO PyTorch → Jittor 迁移进度文档

## 概述

本文档记录 Gold-YOLO 从 PyTorch 到 Jittor 框架的迁移进度和测试状态。

**当前状态**: ✅ 代码迁移完成，✅ 核心训练链路已打通（前向/Loss/反向）

**迁移完成度**:
- 代码对齐: 100% ✅
- 参数量对齐: 100% (5,631,550) ✅
- 前向传播: 100% ✅
- Loss 计算: 100% ✅
- 反向传播: ✅ 已修复（float32 链路稳定）

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

### 2026-02-06 第五批修复（反向传播稳定性）

18. **反向传播 dtype 修复** (`yolov6/models/losses/loss.py`)
   - ✅ 修复 warmup/formal assigner 输出在损失路径中被提升到 `float64` 的问题
   - ✅ 统一 `target_scores/target_bboxes/fg_mask` 为 `float32`
   - ✅ 修复 `VarifocalLoss` 输入类型，保证 `pred/gt/label` 全链路 `float32`
   - ✅ 返回 `loss` 与 `loss_items` 为 `float32`

19. **Assigner 输出类型修复** (`yolov6/assigners/atss_assigner.py`, `yolov6/assigners/tal_assigner.py`)
   - ✅ ATSS/TAL 返回的 `target_bboxes` 与 `target_scores` 统一为 `float32`
   - ✅ 避免在反向图中触发 `cudnn_conv_backward_x` 的 `float64/float32` 混合路径

20. **训练引擎保险修复** (`yolov6/core/engine.py`)
   - ✅ `optimizer.backward()` 前强制 `total_loss.float32()`
   - ✅ 防止后续分支改动引入 float64 回归

21. **蒸馏/融合损失同步修复** (`yolov6/models/losses/loss_fuseab.py`, `loss_distill.py`, `loss_distill_ns.py`)
   - ✅ 对 assigner 输出和 one-hot 标签做 `float32` 统一
   - ✅ 返回损失统一 `float32`

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

**状态**: ✅ 反向传播通过（CUDA 路径）

在 `micromamba run -n jt` 环境下，已验证：
1. 模型构建 + 前向传播 ✅
2. Loss 计算（warmup assigner, `epoch_num=0`）✅
3. Loss 计算（formal assigner, `epoch_num=5`）✅
4. `optimizer.backward(loss) + optimizer.step() + jt.sync_all()` ✅

修复后 `loss.dtype` 为 `float32`，不再触发 `cudnn_conv_backward_x` 的 `float64/float32` 符号错误。

> 说明：当前验证为核心训练链路 smoke test（随机输入/标注），未在完整数据集上长时间训练。

### 深度对齐复核 (2026-02-06)

本轮对齐以 `Gold-YOLO_pytorch` 作为参考，补齐了“完全迁移”判定中的几个关键差异：

1. **EffiDeHead 初始化逻辑严格对齐 PyTorch**
   - `cls_preds/reg_preds` 的 `weight` 回归为零初始化；
   - 保留 `proj_conv` 无梯度投影初始化逻辑。

2. **推理保存路径逻辑严格对齐 PyTorch**
   - 图片输入保存为图片（不再错误写成 `.jpg.mp4`）；
   - 视频/流输入保存为 `.mp4`；
   - 修复 `save_txt` 分支中 `self.save_conf` 未定义风险。

3. **默认配置回归官方基线**
   - `configs/gold_yolo-n.py` 与 PyTorch 版本重新对齐（默认 `data_path=./data/coco.yaml`）。

4. **参数统计口径统一**
   - Jittor 中 BN `running_mean/running_var` 也出现在 `named_parameters`，会导致“总参数量”看起来偏大；
   - 统一采用“**排除 BN running stats**”口径与 PyTorch 对拍。

#### 参数量对拍结果（Gold-YOLO-n, `num_classes=1`）

```
Jittor train(all):        5,631,505
Jittor train(excl stats): 5,613,617
PyTorch train:            5,613,617

Jittor deploy(all):       5,611,409
Jittor deploy(excl stats):5,604,561
PyTorch deploy:           5,604,561
```

结论：在统一统计口径下，训练态与部署态参数量均与 PyTorch 一致。

---

## 修复文件清单

| 文件 | 修复内容 |
|------|----------|
| `yolov6/models/losses/loss.py` | 完全重写 Loss 函数 |
| `yolov6/models/losses/loss_fuseab.py` | 同步修复 dtype 链路 |
| `yolov6/models/losses/loss_distill.py` | 同步修复 dtype 链路 |
| `yolov6/models/losses/loss_distill_ns.py` | 同步修复 dtype 链路 |
| `yolov6/assigners/atss_assigner.py` | 修复输出 dtype 为 float32 |
| `yolov6/assigners/tal_assigner.py` | 完全重写 Assigner |
| `yolov6/assigners/assigner_utils.py` | 完全重写工具函数，修复 argmax 返回值 |
| `yolov6/layers/common.py` | 添加缺失类和融合方法 |
| `yolov6/models/efficientrep.py` | 添加 P6 类，修复逻辑 |
| `yolov6/models/effidehead.py` | 修复 stride 参数注册，初始化逻辑严格对齐 PyTorch |
| `yolov6/core/engine.py` | 添加全部缺失方法 |
| `yolov6/core/inferer.py` | 对齐图片/视频保存逻辑，修复 `save_txt` 分支 |
| `gold_yolo/transformer.py` | 修复 pool_mode 参数 |
| `gold_yolo/layers.py` | 修复 norm_cfg 处理 |
| `gold_yolo/common.py` | 修复层结构 |
| `configs/gold_yolo-n.py` | 回归官方默认配置并保持结构对齐 |
| `tools/train.py` | 修复默认 `--conf-file` 路径 |
| `yolov6/utils/jittor_utils.py` | 参数统计改为排除 BN running stats 口径 |
| `yolov6/utils/figure_iou.py` | 添加 pairwise_bbox_iou |

---

## 参考资料

- [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/)
- [Gold-YOLO 原始仓库](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)
