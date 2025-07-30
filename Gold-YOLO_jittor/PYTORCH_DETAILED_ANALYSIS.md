# PyTorch版本详细分析结果

## 模型参数统计
- 总参数: 5,617,930 (5.62M)
- 可训练参数: 5,617,896
- Backbone参数: 3,138,624
- Neck参数: 2,063,904
- Head参数: 0

## 特征图尺寸
- backbone_out0: torch.Size([1, 32, 125, 125])
- backbone_out1: torch.Size([1, 64, 63, 63])
- backbone_out2: torch.Size([1, 128, 32, 32])
- backbone_out3: torch.Size([1, 256, 16, 16])
- neck_out0: torch.Size([1, 32, 63, 63])
- neck_out1: torch.Size([1, 64, 32, 32])
- neck_out2: torch.Size([1, 128, 16, 16])

## 输出信息
- 输出类型: <class 'list'>
- 输出0: torch.Size([1, 5249, 25])
