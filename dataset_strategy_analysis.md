# Gold-YOLO 数据集策略分析
新芽第二阶段：选择最适合的训练数据集

## 🎯 实验目标
- 建立PyTorch和Jittor版本的对比基准
- 计算资源有限（约800-1000张训练图片）
- 验证模型实现的正确性，而非追求最高精度

## 📊 数据集选择分析

### ❌ 当前方案：COCO 80类
- **数据量**: 800张图片，80个类别
- **平均每类**: 10张图片
- **问题**: 
  - 数据严重不足
  - 类别不平衡
  - 无法学习有效特征
  - 训练效果极差

### ✅ 推荐方案1：Pascal VOC 2012（最佳选择）

**优势：**
- **类别数**: 20个类别（减少4倍）
- **数据分布**: 平均每类40张图片（800÷20=40）
- **类别平衡**: VOC数据集类别分布相对均匀
- **训练效果**: 足够建立有效的对比基准
- **兼容性**: Gold-YOLO原生支持VOC格式

**VOC 20类别：**
```
person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat,
bus, car, motorbike, train, bottle, chair, dining table, potted plant,
sofa, tv/monitor
```

### ✅ 推荐方案2：单类别人像检测

**优势：**
- **类别数**: 1个类别（person）
- **数据分布**: 800张全部用于人像检测
- **训练效果**: 数据充足，效果最好
- **实用性**: 人像检测是常见应用场景

### ✅ 推荐方案3：COCO子集（5-10类）

**优势：**
- **类别数**: 选择5-10个常见类别
- **数据分布**: 每类80-160张图片
- **类别选择**: person, car, bicycle, dog, cat, chair, bottle, bird, horse, sheep

## 🎯 最终推荐：Pascal VOC 2012

### 为什么选择VOC？

1. **数据充足性**: 20类 × 40张/类 = 合理的训练数据量
2. **类别平衡**: VOC数据集经过精心设计，类别分布相对均匀
3. **训练效果**: 足够建立有效的PyTorch vs Jittor对比基准
4. **技术成熟**: VOC是经典数据集，有大量参考实现
5. **计算友好**: 20类别降低了计算复杂度

### VOC数据集获取方案

**方案A：使用现有COCO数据，筛选VOC类别**
- 从800张COCO图片中筛选包含VOC 20类的图片
- 重新标注为VOC格式
- 优点：无需下载新数据
- 缺点：可能某些类别数据仍然不足

**方案B：下载VOC 2012训练集子集**
- 从每个类别中随机选择40-50张图片
- 总计约800-1000张图片
- 优点：数据质量高，类别平衡好
- 缺点：需要下载额外数据

## 🚀 实施建议

### 立即行动方案：
1. **转换为VOC格式**: 将现有COCO数据转换为VOC 20类
2. **类别映射**: 建立COCO 80类到VOC 20类的映射关系
3. **数据筛选**: 保留包含VOC类别的图片，过滤其他图片
4. **重新训练**: 使用VOC 20类进行PyTorch基准训练

### COCO到VOC类别映射：
```python
COCO_TO_VOC_MAPPING = {
    # COCO类别ID -> VOC类别名
    0: 'person',        # person
    1: 'bicycle',       # bicycle  
    2: 'car',          # car
    3: 'motorbike',    # motorcycle
    4: 'aeroplane',    # airplane
    5: 'bus',          # bus
    6: 'train',        # train
    7: 'boat',         # boat
    15: 'bird',        # bird
    16: 'cat',         # cat
    17: 'dog',         # dog
    18: 'horse',       # horse
    19: 'sheep',       # sheep
    20: 'cow',         # cow
    39: 'bottle',      # bottle
    56: 'chair',       # chair
    60: 'dining table', # dining table
    61: 'potted plant', # potted plant
    62: 'sofa',        # couch
    63: 'tv/monitor'   # tv
}
```

## 📈 预期效果对比

| 数据集方案 | 类别数 | 每类图片数 | 预期训练效果 | 对比基准质量 |
|------------|--------|------------|--------------|--------------|
| COCO 80类  | 80     | ~10        | ❌ 很差      | ❌ 无效      |
| VOC 20类   | 20     | ~40        | ✅ 良好      | ✅ 有效      |
| 人像单类   | 1      | 800        | ✅ 优秀      | ✅ 有效      |
| COCO 5类   | 5      | ~160       | ✅ 优秀      | ✅ 有效      |

## 🎯 结论

**强烈推荐使用Pascal VOC 20类方案**，因为：
1. 数据充足性和类别平衡性最佳
2. 能够建立有效的PyTorch vs Jittor对比基准
3. 技术实现相对简单
4. 符合新芽第二阶段的实验目标

这样可以确保我们的Jittor实现能够与PyTorch版本进行有意义的对比！
