# GOLD-YOLO Jittor版本 - 快速参考卡片

## 🚀 通道数速查表

### 基础配置
```python
# gold_yolo-n 缩放参数
depth_mul = 0.33
width_mul = 0.25

# 完整通道列表 (索引对应)
channels_list = [16, 32, 64, 128, 256, 64, 32, 32, 64, 64, 128]
#                0   1   2   3    4    5   6   7   8   9   10

# Extra配置
trans_channels = [16, 8, 16, 32]  # 全局信息通道数
```

### Backbone输出
| 特征图 | 索引 | 通道数 | 分辨率 |
|--------|------|--------|--------|
| c2 | 1 | 32 | 160×160 |
| c3 | 2 | 64 | 80×80 |
| c4 | 3 | 128 | 40×40 |
| c5 | 4 | 256 | 20×20 |

### Neck关键组件
```python
# LAF_p4输入: [c3, c4, c5_half] = [64, 128, 64]
# LAF_p3输入: [c2, c3, p4_half] = [32, 64, 32]

# Inject_p4: inp=64, oup=64, global_inp=16
# Inject_p3: inp=32, oup=32, global_inp=8

# 降维层
# c5 -> c5_half: 256 -> 64
# p4 -> p4_half: 64 -> 32
```

### Head输入/输出
```python
# 输入: [p3, p4, p5] = [32, 64, 128]
# 训练输出: (feats, cls_scores, reg_distri)
# 推理输出: yolo_format [batch, anchors, 5+num_classes]
```

---

## ⚡ 常用调试代码

### 1. 通道数检查
```python
def debug_channels(x, name):
    print(f"{name} shape: {x.shape}")
    if len(x.shape) == 4:
        print(f"  -> 通道数: {x.shape[1]}, 分辨率: {x.shape[2]}×{x.shape[3]}")

# 使用示例
debug_channels(c3, "c3")
debug_channels(c4, "c4") 
debug_channels(c5_half, "c5_half")
```

### 2. 内存监控
```python
import jittor as jt
jt.display_memory_info()
```

### 3. 梯度检查
```python
def check_gradients(model):
    total_norm = 0
    param_count = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm()
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    print(f"梯度范数: {total_norm:.6f}, 参数数量: {param_count}")
    return total_norm
```

### 4. 损失值检查
```python
def check_loss_values(loss, outputs):
    print(f"损失值: {loss:.6f}")
    if jt.isnan(loss):
        print("❌ 损失为NaN!")
        print(f"模型输出范围: {outputs.min():.6f} ~ {outputs.max():.6f}")
        print(f"包含NaN: {jt.isnan(outputs).any()}")
        print(f"包含Inf: {jt.isinf(outputs).any()}")
```

---

## 🔧 快速修复模板

### 通道数不匹配修复
```python
# 1. 找到出错的组件
# 2. 检查输入输出通道数
# 3. 对照channels_list修正

# LAF_p4修复示例
self.LAF_p4 = SimFusion_3in(
    in_channel_list=[
        channels_list[2],  # c3: 64
        channels_list[3],  # c4: 128
        channels_list[5]   # c5_half: 64
    ],
    out_channels=channels_list[5]  # 64
)

# Inject修复示例  
self.Inject_p4 = InjectionMultiSum_Auto_pool(
    inp=channels_list[5],              # 64
    oup=channels_list[5],              # 64
    global_inp=extra_cfg.trans_channels[0]  # 16
)
```

### 训练稳定性修复
```python
# 1. 降低学习率
lr = 0.01  # 从0.02降到0.01

# 2. 减小批次大小
batch_size = 8  # 从16降到8

# 3. 添加梯度裁剪
def clip_gradients(model, max_norm=10.0):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm()
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for param in model.parameters():
            if param.grad is not None:
                param.grad *= clip_coef
    
    return total_norm
```

---

## 📋 检查清单

### 架构对齐检查
- [ ] Backbone输出通道数正确
- [ ] LAF_p4输入通道数: [64, 128, 64]
- [ ] LAF_p3输入通道数: [32, 64, 32]  
- [ ] Inject_p4全局输入: 16通道
- [ ] Inject_p3全局输入: 8通道
- [ ] Head输入通道数: [32, 64, 128]
- [ ] 训练时输出格式: (feats, cls_scores, reg_distri)

### 训练稳定性检查
- [ ] 学习率合理 (0.01-0.02)
- [ ] 批次大小适中 (8-16)
- [ ] 梯度范数正常 (<10.0)
- [ ] 损失值收敛 (不为NaN/Inf)
- [ ] 内存使用正常 (<90%)

### 性能优化检查
- [ ] 数据加载效率
- [ ] 前向传播速度
- [ ] 内存使用优化
- [ ] GPU利用率

---

## 🆘 紧急故障处理

### 训练崩溃
1. **立即检查**: 损失值、梯度范数、内存使用
2. **降级策略**: 减小lr和batch_size
3. **回滚代码**: 恢复到最后一个稳定版本

### 内存不足
1. **立即操作**: 减小batch_size到4或更小
2. **清理内存**: 调用`jt.gc()`和`jt.display_memory_info()`
3. **检查泄漏**: 查看lived_vars数量

### 通道数错误
1. **定位错误**: 查看完整错误堆栈
2. **对照文档**: 检查ARCHITECTURE_DOCUMENTATION.md
3. **逐步验证**: 从Backbone开始逐层检查

---

## 📞 联系信息

**文档位置**: `Gold-YOLO_jittor/ARCHITECTURE_DOCUMENTATION.md`
**快速参考**: `Gold-YOLO_jittor/QUICK_REFERENCE.md` (本文件)
**训练脚本**: `Gold-YOLO_jittor/train_pytorch_aligned_stable.py`

**最后更新**: 2025-07-28
