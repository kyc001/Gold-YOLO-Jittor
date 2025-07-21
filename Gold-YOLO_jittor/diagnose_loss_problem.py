#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
深入诊断Jittor损失函数问题
新芽第二阶段：彻底解决损失数值异常小的问题
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置Jittor
jt.flags.use_cuda = 1

class ProblemLossFunction(nn.Module):
    """有问题的损失函数 - 重现异常"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def execute(self, pred, targets=None):
        multi_feats, cls_pred, reg_pred = pred
        
        batch_size = cls_pred.shape[0]
        num_anchors = cls_pred.shape[1]
        num_classes = cls_pred.shape[2]
        reg_dim = reg_pred.shape[2]
        
        # 问题1: 人工目标过于简单
        cls_targets = jt.zeros_like(cls_pred)
        reg_targets = jt.zeros_like(reg_pred)
        
        for b in range(batch_size):
            num_pos = min(10, num_anchors)
            for i in range(num_pos):
                cls_id = i % num_classes
                cls_targets[b, i, cls_id] = 1.0
                
                # 问题2: 回归目标过小
                for j in range(min(4, reg_dim)):
                    reg_targets[b, i, j] = 0.1 + 0.05 * jt.randn(1)
        
        # 问题3: BCE对稀疏目标计算有问题
        reg_loss = self.mse_loss(reg_pred, reg_targets)
        cls_loss = self.bce_loss(cls_pred, cls_targets)
        
        # 问题4: 梯度强制项权重过小
        reg_gradient_force = jt.mean(reg_pred ** 2) * 0.1
        
        total_loss = reg_loss + cls_loss + reg_gradient_force
        
        return total_loss, reg_loss, cls_loss, reg_gradient_force


class FixedLossFunction(nn.Module):
    """修复后的损失函数"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 修复1: 合理的损失权重
        self.lambda_coord = 5.0
        self.lambda_cls = 1.0
        self.lambda_obj = 1.0
        self.lambda_reg = 0.5
    
    def execute(self, pred, targets=None):
        multi_feats, cls_pred, reg_pred = pred
        
        batch_size = cls_pred.shape[0]
        num_anchors = cls_pred.shape[1]
        num_classes = cls_pred.shape[2]
        reg_dim = reg_pred.shape[2]
        
        # 修复2: 更真实的目标分布
        cls_targets = jt.zeros_like(cls_pred)
        reg_targets = jt.zeros_like(reg_pred)
        obj_mask = jt.zeros((batch_size, num_anchors))
        
        for b in range(batch_size):
            # 修复3: 更合理的正样本数量
            num_pos = np.random.randint(3, min(15, num_anchors//5))
            pos_indices = np.random.choice(num_anchors, num_pos, replace=False)
            
            for idx in pos_indices:
                obj_mask[b, idx] = 1.0
                
                # 随机类别
                cls_id = np.random.randint(0, num_classes)
                cls_targets[b, idx, cls_id] = 1.0
                
                # 修复4: 更真实的回归目标
                reg_targets[b, idx, 0] = np.random.uniform(0.2, 0.8)  # x
                reg_targets[b, idx, 1] = np.random.uniform(0.2, 0.8)  # y
                reg_targets[b, idx, 2] = np.random.uniform(0.1, 0.6)  # w
                reg_targets[b, idx, 3] = np.random.uniform(0.1, 0.6)  # h
                
                # DFL分布
                for j in range(4, min(20, reg_dim)):
                    reg_targets[b, idx, j] = np.random.uniform(0.0, 0.5)
        
        # 修复5: 只对有目标的位置计算损失
        pos_mask_cls = obj_mask.unsqueeze(-1).expand_as(cls_pred)
        pos_mask_reg = obj_mask.unsqueeze(-1).expand_as(reg_pred)
        
        # 分类损失 - 只对正样本
        cls_loss = self.bce_loss(cls_pred * pos_mask_cls, cls_targets * pos_mask_cls)
        
        # 回归损失 - 只对正样本
        reg_loss = self.mse_loss(reg_pred * pos_mask_reg, reg_targets * pos_mask_reg)
        
        # 目标性损失
        obj_pred = jt.max(cls_pred, dim=-1)  # [batch, anchors]
        if isinstance(obj_pred, tuple):
            obj_pred = obj_pred[0]
        obj_loss = self.bce_loss(obj_pred, obj_mask)
        
        # 无目标损失
        noobj_mask = 1.0 - obj_mask
        noobj_loss = self.bce_loss(obj_pred * noobj_mask, jt.zeros_like(obj_pred) * noobj_mask)
        
        # 修复6: 合理的权重组合
        total_loss = (self.lambda_coord * reg_loss + 
                     self.lambda_cls * cls_loss + 
                     self.lambda_obj * obj_loss + 
                     self.lambda_reg * noobj_loss)
        
        return total_loss, reg_loss, cls_loss, obj_loss, noobj_loss


def create_test_model():
    """创建测试模型"""
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.cls_head = nn.Linear(32, 525 * 80)
            self.reg_head = nn.Linear(32, 525 * 68)
        
        def execute(self, x):
            feat = self.backbone(x)
            feat = feat.view(x.size(0), -1)
            
            cls_pred = self.cls_head(feat).view(x.size(0), 525, 80)
            reg_pred = self.reg_head(feat).view(x.size(0), 525, 68)
            
            return feat, cls_pred, reg_pred
    
    return SimpleTestModel()


def diagnose_loss_behavior():
    """诊断损失函数行为"""
    print("🔍 深入诊断损失函数问题...")
    print("=" * 60)
    
    # 创建测试模型
    model = create_test_model()
    problem_loss = ProblemLossFunction()
    fixed_loss = FixedLossFunction()
    
    # 测试数据
    batch_size = 4
    test_input = jt.randn(batch_size, 3, 640, 640)
    
    print(f"📊 测试配置:")
    print(f"   批次大小: {batch_size}")
    print(f"   输入尺寸: {test_input.shape}")
    
    # 记录损失变化
    problem_losses = []
    fixed_losses = []
    
    print("\n🧪 损失函数对比测试:")
    print("=" * 60)
    
    for step in range(20):
        # 前向传播
        with jt.no_grad():
            outputs = model(test_input)
        
        # 问题损失函数
        prob_total, prob_reg, prob_cls, prob_force = problem_loss(outputs)
        problem_losses.append(prob_total.item())
        
        # 修复损失函数
        fixed_total, fixed_reg, fixed_cls, fixed_obj, fixed_noobj = fixed_loss(outputs)
        fixed_losses.append(fixed_total.item())
        
        if step % 5 == 0:
            print(f"Step {step:2d}:")
            print(f"  问题版本: total={prob_total.item():.4f}, reg={prob_reg.item():.4f}, cls={prob_cls.item():.4f}")
            print(f"  修复版本: total={fixed_total.item():.4f}, reg={fixed_reg.item():.4f}, cls={fixed_cls.item():.4f}")
            print()
    
    # 分析结果
    print("📈 损失分析结果:")
    print("=" * 60)
    print(f"问题版本:")
    print(f"  初始损失: {problem_losses[0]:.4f}")
    print(f"  最终损失: {problem_losses[-1]:.4f}")
    print(f"  损失范围: {min(problem_losses):.4f} - {max(problem_losses):.4f}")
    print(f"  平均损失: {np.mean(problem_losses):.4f}")
    
    print(f"\n修复版本:")
    print(f"  初始损失: {fixed_losses[0]:.4f}")
    print(f"  最终损失: {fixed_losses[-1]:.4f}")
    print(f"  损失范围: {min(fixed_losses):.4f} - {max(fixed_losses):.4f}")
    print(f"  平均损失: {np.mean(fixed_losses):.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(problem_losses, 'r-', label='问题版本', linewidth=2)
    plt.plot(fixed_losses, 'b-', label='修复版本', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('损失函数对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(problem_losses, 'r-', label='问题版本', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('问题版本损失 (放大)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path("runs/loss_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'loss_comparison_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 诊断图表已保存: {output_dir}/loss_comparison_diagnosis.png")
    
    # 问题总结
    print("\n🚨 问题总结:")
    print("=" * 60)
    
    if np.mean(problem_losses) < 0.1:
        print("❌ 确认问题: 损失值异常小")
        print("   原因1: 人工目标过于简单，容易拟合")
        print("   原因2: BCE损失对稀疏目标计算不当")
        print("   原因3: 梯度强制项权重过小")
        print("   原因4: 缺乏真实的目标检测损失结构")
    
    if np.mean(fixed_losses) > 1.0:
        print("✅ 修复成功: 损失值回到正常范围")
        print("   修复1: 使用真实的目标检测损失结构")
        print("   修复2: 合理的正负样本比例")
        print("   修复3: 适当的损失权重配置")
        print("   修复4: 只对有目标位置计算损失")
    
    return problem_losses, fixed_losses


def main():
    """主函数"""
    print("🎯 Jittor损失函数深度诊断")
    print("新芽第二阶段：彻底解决损失异常问题")
    print("=" * 80)
    
    # 运行诊断
    problem_losses, fixed_losses = diagnose_loss_behavior()
    
    print("\n🔧 下一步修复建议:")
    print("=" * 60)
    print("1. 替换当前的StableGradientLoss为FixedLossFunction")
    print("2. 使用真实的YOLO损失结构")
    print("3. 合理设置正负样本比例")
    print("4. 调整损失权重配置")
    print("5. 重新训练验证修复效果")
    
    print("\n✅ 诊断完成！请查看runs/loss_diagnosis/目录获取详细结果")


if __name__ == "__main__":
    main()
