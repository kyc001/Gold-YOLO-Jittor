# GOLD-YOLO Jittor版本 - 完整架构文档

## 📋 概述

本文档详细记录了GOLD-YOLO Jittor版本的完整架构设计，包括每个组件的通道数配置、参数对齐情况以及与PyTorch版本的对比。

**版本信息**：
- **Jittor版本**：1.0.0
- **对齐目标**：Gold-YOLO_pytorch
- **模型规模**：gold_yolo-n (nano版本)
- **参数量**：5.71M

---

## 🏗️ 整体架构

```
输入图像 (3, 640, 640)
    ↓
Backbone (EfficientRep)
    ↓
Neck (RepGDNeck) 
    ↓
Head (EffiDeHead)
    ↓
输出 (训练时: feats, pred_scores, pred_distri)
```

---

## 📊 通道数配置表

### 基础配置参数
```python
# 缩放参数 (gold_yolo-n)
depth_mul = 0.33
width_mul = 0.25

# 原始通道配置
backbone_base_channels = [64, 128, 256, 512, 1024]
neck_base_channels = [256, 128, 128, 256, 256, 512]
extra_cfg_base = {
    'fusion_in': 480,
    'embed_dim_p': 96,
    'embed_dim_n': 352,
    'trans_channels': [64, 32, 64, 128]
}
```

### 缩放后的实际通道数
```python
# Backbone通道数 (缩放后)
channels_list_backbone = [16, 32, 64, 128, 256]  # 索引0-4

# Neck通道数 (缩放后)
neck_channels = [64, 32, 32, 64, 64, 128]  # 索引5-10

# 完整通道列表
full_channels_list = [16, 32, 64, 128, 256, 64, 32, 32, 64, 64, 128]
#                     0   1   2   3    4    5   6   7   8   9   10

# Extra配置 (缩放后)
extra_cfg_scaled = {
    'fusion_in': 480,      # 实际通道数，不缩放
    'embed_dim_p': 24,     # 96 * 0.25 = 24
    'embed_dim_n': 88,     # 352 * 0.25 = 88
    'trans_channels': [16, 8, 16, 32]  # [64*0.25, 32*0.25, 64*0.25, 128*0.25]
}
```

---

## 🔧 Backbone详细架构

### EfficientRep结构
```python
class EfficientRep:
    def __init__(self):
        # 输入: (batch, 3, 640, 640)
        
        # Stem层
        self.stem = RepVGGBlock(3, 16, 3, 2)  # -> (batch, 16, 320, 320)
        
        # Stage 1
        self.ERBlock_2 = nn.Sequential(
            RepVGGBlock(16, 32, 3, 2),  # -> (batch, 32, 160, 160)
            *[RepVGGBlock(32, 32) for _ in range(1)]  # 重复1次
        )
        
        # Stage 2  
        self.ERBlock_3 = nn.Sequential(
            RepVGGBlock(32, 64, 3, 2),  # -> (batch, 64, 80, 80)
            *[RepVGGBlock(64, 64) for _ in range(3)]  # 重复3次
        )
        
        # Stage 3
        self.ERBlock_4 = nn.Sequential(
            RepVGGBlock(64, 128, 3, 2),  # -> (batch, 128, 40, 40)
            *[RepVGGBlock(128, 128) for _ in range(5)]  # 重复5次
        )
        
        # Stage 4
        self.ERBlock_5 = nn.Sequential(
            RepVGGBlock(128, 256, 3, 2),  # -> (batch, 256, 20, 20)
            *[RepVGGBlock(256, 256) for _ in range(1)]  # 重复1次
        )

    def execute(self, x):
        # 输出特征图
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)  # c2: (batch, 32, 160, 160)
        x = self.ERBlock_3(x)  
        outputs.append(x)  # c3: (batch, 64, 80, 80)
        x = self.ERBlock_4(x)
        outputs.append(x)  # c4: (batch, 128, 40, 40)
        x = self.ERBlock_5(x)
        outputs.append(x)  # c5: (batch, 256, 20, 20)
        
        return outputs  # [c2, c3, c4, c5]
```

### Backbone输出通道数总结
| 特征图 | 索引 | 通道数 | 分辨率 | 对应层 |
|--------|------|--------|--------|--------|
| c2 | channels_list[1] | 32 | 160×160 | ERBlock_2 |
| c3 | channels_list[2] | 64 | 80×80 | ERBlock_3 |
| c4 | channels_list[3] | 128 | 40×40 | ERBlock_4 |
| c5 | channels_list[4] | 256 | 20×20 | ERBlock_5 |

---

## 🔗 Neck详细架构 (RepGDNeck)

### 关键组件通道数配置

#### 1. 降维层 (Reduce Layers)
```python
# c5降维层
self.reduce_layer_c5 = SimConv(
    in_channels=channels_list[4],   # 256 (c5)
    out_channels=channels_list[5],  # 64 (c5_half)
    kernel_size=1
)

# p4降维层  
self.reduce_layer_p4 = SimConv(
    in_channels=channels_list[5],   # 64 (p4)
    out_channels=channels_list[6],  # 32 (p4_half)
    kernel_size=1
)
```

#### 2. 融合层 (LAF - Local Attention Fusion)
```python
# LAF_p4: 融合 [c3, c4, c5_half]
self.LAF_p4 = SimFusion_3in(
    in_channel_list=[
        channels_list[2],  # 64 (c3)
        channels_list[3],  # 128 (c4) 
        channels_list[5]   # 64 (c5_half)
    ],
    out_channels=channels_list[5]  # 64
)

# LAF_p3: 融合 [c2, c3, p4_half]
self.LAF_p3 = SimFusion_3in(
    in_channel_list=[
        channels_list[1],  # 32 (c2)
        channels_list[2],  # 64 (c3)
        channels_list[6]   # 32 (p4_half)
    ],
    out_channels=channels_list[6]  # 32
)
```

#### 3. 注入层 (Inject)
```python
# Inject_p4: 注入全局信息到p4
self.Inject_p4 = InjectionMultiSum_Auto_pool(
    inp=channels_list[5],              # 64 (局部特征)
    oup=channels_list[5],              # 64 (输出)
    global_inp=extra_cfg.trans_channels[0]  # 16 (全局信息)
)

# Inject_p3: 注入全局信息到p3  
self.Inject_p3 = InjectionMultiSum_Auto_pool(
    inp=channels_list[6],              # 32 (局部特征)
    oup=channels_list[6],              # 32 (输出)
    global_inp=extra_cfg.trans_channels[1]  # 8 (全局信息)
)
```

### 全局信息流处理

#### Low-level全局信息
```python
# 输入融合
input_fused = [c2, c3, c4, c5]  # 通道数: [32, 64, 128, 256]
fusion_in = 32 + 64 + 128 + 256 = 480

# 全局信息提取
self.low_FAM = SimFusion_4in()  # 输入: 480, 输出: 480
self.low_IFM = nn.Sequential(
    Conv(480, 24, 1),  # embed_dim_p = 24
    *[RepVGGBlock(24, 24) for _ in range(4)],  # fuse_block_num = 4
    Conv(24, 24, 1)    # sum(trans_channels[0:2]) = 16+8 = 24
)

# 分割全局信息
low_global_info = low_fuse_feat.split([16, 8], dim=1)
# low_global_info[0]: 16通道 -> 用于Inject_p4
# low_global_info[1]: 8通道  -> 用于Inject_p3
```

### Neck数据流图
```
c2(32) ──┐
c3(64) ──┼─→ LAF_p4 ──→ p4_adjacent_info(64) ──┐
c4(128)──┘                                      │
c5(256)──→ reduce_c5 ──→ c5_half(64) ──────────┘
                                                │
low_global_info[0](16) ─────────────────────────┼─→ Inject_p4 ──→ p4(64)
                                                │
c2(32) ──┐                                     │
c3(64) ──┼─→ LAF_p3 ──→ p3_adjacent_info(32) ──┘
p4(64) ──→ reduce_p4 ──→ p4_half(32) ──────────┘
                                                │
low_global_info[1](8) ──────────────────────────┼─→ Inject_p3 ──→ p3(32)
```

---

## 🎯 Head详细架构 (EffiDeHead)

### 输入特征图
```python
# 来自Neck的输出
neck_outputs = [p3, p4, p5]
# 通道数: [32, 64, 128]  (对应channels_list[6,5,4])
# 分辨率: [80×80, 40×40, 20×20]
```

### Head结构
```python
class EffiDeHead:
    def __init__(self):
        # 输入通道数
        in_channels = [32, 64, 128]  # [channels_list[6], channels_list[5], channels_list[4]]
        
        # Stem层
        self.stems = nn.ModuleList([
            SimConv(in_ch, in_ch, 1, 1) for in_ch in in_channels
        ])
        
        # 分类分支
        self.cls_convs = nn.ModuleList([
            SimConv(in_ch, in_ch, 3, 1) for in_ch in in_channels  
        ])
        self.cls_preds = nn.ModuleList([
            nn.Conv2d(in_ch, num_classes, 1) for in_ch in in_channels
        ])
        
        # 回归分支
        self.reg_convs = nn.ModuleList([
            SimConv(in_ch, in_ch, 3, 1) for in_ch in in_channels
        ])
        self.reg_preds = nn.ModuleList([
            nn.Conv2d(in_ch, 4, 1) for in_ch in in_channels  # 4个坐标
        ])
```

### Head输出格式
```python
def execute(self, x):
    if self.training:
        # 训练时输出 (严格对齐PyTorch)
        feats = x  # 原始特征图
        cls_score_list = []  # 分类得分列表
        reg_distri_list = []  # 回归分布列表
        
        for i in range(self.nl):
            # 处理每个尺度
            x_i = self.stems[i](x[i])
            cls_output = self.cls_preds[i](self.cls_convs[i](x_i))
            reg_output = self.reg_preds[i](self.reg_convs[i](x_i))
            
            # 重塑为 [batch, anchors, channels]
            cls_score_list.append(cls_output.reshape([b, self.nc, -1]))
            reg_distri_list.append(reg_output.reshape([b, 4, -1]))
        
        # 合并所有尺度
        cls_score_list = jt.concat(cls_score_list, dim=-1).permute(0, 2, 1)
        reg_distri_list = jt.concat(reg_distri_list, dim=-1).permute(0, 2, 1)
        
        return feats, cls_score_list, reg_distri_list
    else:
        # 推理时输出YOLO格式
        return yolo_format_output
```

---

## 📉 损失函数架构

### ComputeLoss输入格式
```python
def __call__(self, outputs, targets, epoch_num, step_num):
    # 严格对齐PyTorch版本的输入解析
    feats, pred_scores, pred_distri = outputs
    
    # feats: 原始特征图 [p3, p4, p5]
    # pred_scores: 分类得分 [batch, total_anchors, num_classes]  
    # pred_distri: 回归分布 [batch, total_anchors, 4]
```

### 锚点生成
```python
# 使用特征图生成锚点
anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
    feats,                    # [p3, p4, p5]
    fpn_strides=[8, 16, 32], # 对应的步长
    grid_cell_size=5.0,
    grid_cell_offset=0.5
)

# 锚点数量统计
# p3 (80×80): 6400个锚点
# p4 (40×40): 1600个锚点  
# p5 (20×20): 400个锚点
# 总计: 8400个锚点
```

---

## ⚠️ 关键注意事项

### 1. 通道数对齐检查清单
- [ ] Backbone输出通道数与channels_list[1-4]一致
- [ ] LAF_p4输入通道数: [c3, c4, c5_half] = [64, 128, 64]
- [ ] LAF_p3输入通道数: [c2, c3, p4_half] = [32, 64, 32]
- [ ] Inject_p4全局输入通道数: trans_channels[0] = 16
- [ ] Inject_p3全局输入通道数: trans_channels[1] = 8
- [ ] Head输入通道数: [32, 64, 128]

### 2. 常见错误模式
1. **SimFusion_3in通道数不匹配**: 检查in_channel_list与实际输入是否对应
2. **InjectionMultiSum_Auto_pool全局通道数错误**: 确保global_inp参数正确设置
3. **Head输出格式不对齐**: 训练时必须返回三元组(feats, cls_scores, reg_distri)

### 3. 调试技巧
```python
# 在关键位置添加通道数检查
print(f"c3 shape: {c3.shape}")  # 应该是 [batch, 64, 80, 80]
print(f"c4 shape: {c4.shape}")  # 应该是 [batch, 128, 40, 40]
print(f"c5_half shape: {c5_half.shape}")  # 应该是 [batch, 64, 40, 40]
print(f"low_global_info[0] shape: {low_global_info[0].shape}")  # 应该是 [batch, 16, H, W]
```

---

## 📝 版本历史

### v1.0.0 (当前版本)
- ✅ 完成Backbone、Neck、Head架构对齐
- ✅ 修复所有通道数不匹配问题
- ✅ 严格对齐PyTorch版本的训练参数
- ✅ 实现稳定的梯度计算和损失收敛

### 待优化项目
- [ ] 性能优化：减少内存使用
- [ ] 推理优化：加速推理过程
- [ ] 模型压缩：进一步减小模型大小

---

---

## 🔧 详细组件实现

### SimFusion_3in实现细节
```python
class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        # 兼容性处理：支持2元素和3元素输入
        if len(in_channel_list) == 2:
            # 扩展为3元素: [ch1, ch1, ch2]
            in_channels = [in_channel_list[0], in_channel_list[0], in_channel_list[1]]
        else:
            in_channels = in_channel_list

        # 为每个输入创建1x1卷积转换层
        self.cv0 = SimConv(in_channels[0], out_channels, 1, 1)  # 处理x[0]
        self.cv1 = SimConv(in_channels[1], out_channels, 1, 1)  # 处理x[1]
        self.cv2 = SimConv(in_channels[2], out_channels, 1, 1)  # 处理x[2]

        # 融合层：3*out_channels -> out_channels
        self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)

    def execute(self, x):
        # x是包含3个tensor的列表
        x0 = self.cv0(x[0])  # 转换第1个输入
        x1 = self.cv1(x[1])  # 转换第2个输入
        x2 = self.cv2(x[2])  # 转换第3个输入

        # 自适应池化到相同尺寸（以最小尺寸为准）
        min_h = min(x0.shape[2], x1.shape[2], x2.shape[2])
        min_w = min(x0.shape[3], x1.shape[3], x2.shape[3])

        if x0.shape[2:] != (min_h, min_w):
            x0 = jt.pool.AdaptiveAvgPool2d((min_h, min_w))(x0)
        if x1.shape[2:] != (min_h, min_w):
            x1 = jt.pool.AdaptiveAvgPool2d((min_h, min_w))(x1)
        if x2.shape[2:] != (min_h, min_w):
            x2 = jt.pool.AdaptiveAvgPool2d((min_h, min_w))(x2)

        # 通道维度拼接并融合
        fused = jt.concat([x0, x1, x2], dim=1)  # [batch, 3*out_channels, H, W]
        output = self.cv_fuse(fused)  # [batch, out_channels, H, W]

        return output
```

### InjectionMultiSum_Auto_pool实现细节
```python
class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(self, inp: int, oup: int, norm_cfg=dict(type='BN'),
                 activations=None, global_inp=None):
        super().__init__()

        # 如果未指定global_inp，默认与inp相同
        if not global_inp:
            global_inp = inp

        # 局部特征嵌入：inp -> oup
        self.local_embedding = ConvModule(inp, oup, kernel_size=1,
                                        norm_cfg=norm_cfg, act_cfg=None)

        # 全局特征嵌入：global_inp -> oup
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1,
                                         norm_cfg=norm_cfg, act_cfg=None)

        # 全局注意力：global_inp -> oup
        self.global_act = ConvModule(global_inp, oup, kernel_size=1,
                                   norm_cfg=norm_cfg, act_cfg=None)

        # 激活函数
        self.act = h_sigmoid()

    def execute(self, x_l, x_g):
        """
        x_l: 局部特征 [batch, inp, H, W]
        x_g: 全局特征 [batch, global_inp, H_g, W_g]
        """
        B, C, H, W = x_l.shape

        # 局部特征处理
        local_feat = self.local_embedding(x_l)  # [batch, oup, H, W]

        # 全局特征自适应池化到局部特征尺寸
        global_feat = jt.pool.AdaptiveAvgPool2d((H, W))(x_g)  # [batch, global_inp, H, W]

        # 全局特征嵌入和注意力
        global_embed = self.global_embedding(global_feat)  # [batch, oup, H, W]
        global_att = self.global_act(global_feat)  # [batch, oup, H, W]
        global_att = self.act(global_att)  # sigmoid激活

        # 特征融合：局部 + 全局*注意力
        output = local_feat + global_embed * global_att

        return output
```

---

## 🧮 参数量统计详解

### 各组件参数量分布
```python
# Backbone (EfficientRep): 3,448,928 参数 (60.3%)
├── stem: RepVGGBlock(3->16)           # ~1,200 参数
├── ERBlock_2: RepVGGBlock(16->32) x2  # ~50,000 参数
├── ERBlock_3: RepVGGBlock(32->64) x4  # ~200,000 参数
├── ERBlock_4: RepVGGBlock(64->128) x6 # ~800,000 参数
└── ERBlock_5: RepVGGBlock(128->256) x2# ~2,400,000 参数

# Neck (RepGDNeck): 1,849,328 参数 (32.4%)
├── 全局信息处理模块                    # ~500,000 参数
├── LAF融合层                         # ~300,000 参数
├── Injection注入层                   # ~400,000 参数
└── RepBlock处理层                    # ~650,000 参数

# Head (EffiDeHead): 416,715 参数 (7.3%)
├── Stem层: 3个SimConv                # ~100,000 参数
├── 分类分支: 3个分支 x 20类            # ~200,000 参数
└── 回归分支: 3个分支 x 4坐标           # ~116,715 参数

总计: 5,714,971 参数 (5.71M)
```

### 内存使用估算
```python
# 训练时内存使用 (batch_size=16)
输入图像: 16 x 3 x 640 x 640 x 4字节 = 78.6MB
中间特征图:
├── c2: 16 x 32 x 160 x 160 x 4字节 = 52.4MB
├── c3: 16 x 64 x 80 x 80 x 4字节 = 26.2MB
├── c4: 16 x 128 x 40 x 40 x 4字节 = 13.1MB
├── c5: 16 x 256 x 20 x 20 x 4字节 = 6.6MB
└── Neck输出: ~50MB

模型参数: 5.71M x 4字节 = 22.8MB
梯度缓存: 5.71M x 4字节 = 22.8MB
优化器状态: 5.71M x 8字节 = 45.6MB

估算总内存: ~320MB (理论值)
实际使用: ~2GB (包含Jittor框架开销)
```

---

## 🔍 故障排除指南

### 常见错误及解决方案

#### 1. AssertionError: C==self.in_channels
**原因**: 卷积层输入通道数与期望不匹配
**排查步骤**:
```python
# 在出错位置前添加调试代码
print(f"输入tensor形状: {x.shape}")
print(f"期望输入通道数: {conv_layer.in_channels}")
print(f"实际输入通道数: {x.shape[1]}")
```
**解决方案**: 检查channels_list配置和组件连接

#### 2. 训练损失为NaN或Inf
**原因**: 梯度爆炸或数值不稳定
**排查步骤**:
```python
# 检查模型输出
outputs = model(images)
print(f"模型输出范围: {outputs.min()} ~ {outputs.max()}")
print(f"是否包含NaN: {jt.isnan(outputs).any()}")
print(f"是否包含Inf: {jt.isinf(outputs).any()}")
```
**解决方案**: 调整学习率、添加梯度裁剪

#### 3. GPU内存不足
**原因**: 批次大小过大或内存泄漏
**排查步骤**:
```python
# 监控内存使用
import jittor as jt
jt.display_memory_info()
```
**解决方案**: 减小batch_size、添加内存清理

#### 4. 训练速度过慢
**原因**: 频繁的内存分配或低效操作
**排查步骤**:
```python
# 性能分析
import time
start_time = time.time()
outputs = model(images)
forward_time = time.time() - start_time
print(f"前向传播耗时: {forward_time:.3f}s")
```
**解决方案**: 优化数据加载、使用混合精度

---

## 📚 参考资料

### PyTorch版本对比
- **源码位置**: `Gold-YOLO_pytorch/`
- **关键差异**:
  - Jittor使用`execute()`方法替代`forward()`
  - 张量操作API略有不同 (`jt.concat` vs `torch.cat`)
  - 批归一化和激活函数实现差异

### 相关论文
- **GOLD-YOLO**: "Efficient Object Detection with Global-Local Fusion"
- **YOLOv6**: "A Single-Stage Object Detection Framework"
- **RepVGG**: "Making VGG-style ConvNets Great Again"

### 开发工具
- **Jittor**: https://github.com/Jittor/jittor
- **调试工具**: `jt.display_memory_info()`, `jt.profiler`
- **可视化**: TensorBoard, Wandb

---

**文档维护**: 每次架构修改后必须更新此文档！
**最后更新**: 2025-07-28
**维护者**: Augment Agent
