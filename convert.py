from jittor.utils.pytorch_converter import convert

pytorch_code="""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 简化的损失函数
class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_scores, pred_distri, targets):
        # 分类损失
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        num_classes = pred_scores.shape[2]

        # 创建目标标签
        target_labels = torch.zeros_like(pred_scores)

        # 简单分配策略
        if 'cls' in targets:
            gt_classes = targets['cls'][0]
            for i, cls_id in enumerate(gt_classes):
                start_idx = i * 1000
                end_idx = min(start_idx + 1000, num_anchors)
                target_labels[0, start_idx:end_idx, cls_id] = 1.0

        # 分类损失
        cls_loss = self.bce_loss(pred_scores, target_labels)

        # 回归损失（简化）
        reg_loss = self.mse_loss(pred_distri, torch.zeros_like(pred_distri))

        total_loss = cls_loss + 0.1 * reg_loss
        return total_loss
"""

jittor_code = convert(pytorch_code)
print(jittor_code)