# RTX 4060 8GB 优化训练配置
# 针对算力有限的情况优化

import os

class TrainConfig:
    def __init__(self):
        # 硬件配置
        self.device = 'cuda'
        self.gpu_memory_limit = 8  # GB
        
        # 数据集配置
        self.dataset_type = 'coco_subset'
        self.data_root = './data'
        self.selected_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'stop sign', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'bottle', 'chair', 'laptop', 'cell phone', 'book'
        ]
        self.num_classes = len(self.selected_classes)
        
        # 模型配置
        self.model_name = 'gold_yolo_s'
        self.input_size = 512  # 降低到512节省显存
        self.pretrained = True
        self.pretrained_path = './weights/gold_yolo_s_coco.pth'
        
        # 训练配置（显存优化）
        self.batch_size = 6  # 8GB显存的最佳选择
        self.num_workers = 4  # 充分利用32GB RAM
        self.pin_memory = True
        self.persistent_workers = True
        
        # 优化器配置
        self.optimizer = 'SGD'
        self.lr = 0.01
        self.momentum = 0.937
        self.weight_decay = 0.0005
        self.nesterov = True
        
        # 学习率调度
        self.lr_scheduler = 'cosine'
        self.warmup_epochs = 5
        self.warmup_lr = 0.001
        self.min_lr = 0.0001
        
        # 训练策略
        self.epochs = 150
        self.gradient_accumulation_steps = 2  # 模拟batch_size=12
        self.mixed_precision = True  # 必须开启！
        self.gradient_clip_norm = 10.0
        
        # 数据增强（显存友好）
        self.mosaic_prob = 0.5  # 降低mosaic概率
        self.mixup_prob = 0.1   # 减少mixup
        self.copy_paste_prob = 0.0  # 关闭copy-paste
        self.hsv_h = 0.015
        self.hsv_s = 0.7
        self.hsv_v = 0.4
        self.degrees = 0.0
        self.translate = 0.1
        self.scale = 0.9
        self.shear = 0.0
        self.flipud = 0.0
        self.fliplr = 0.5
        
        # 验证配置
        self.val_interval = 10  # 每10个epoch验证一次
        self.save_interval = 20  # 每20个epoch保存一次
        self.early_stop_patience = 30
        
        # 输出配置
        self.work_dir = './work_dirs/gold_yolo_s_4060'
        self.log_interval = 50
        self.save_best_only = True


class QuickTestConfig(TrainConfig):
    """快速测试配置（Pascal VOC）"""
    def __init__(self):
        super().__init__()
        self.dataset_type = 'voc'
        self.num_classes = 20
        self.input_size = 416
        self.batch_size = 8
        self.epochs = 50
        self.val_interval = 5
        self.work_dir = './work_dirs/quick_test'


class FullTrainConfig(TrainConfig):
    """完整训练配置"""
    def __init__(self):
        super().__init__()
        self.input_size = 640  # 最终阶段使用更大尺寸
        self.batch_size = 4    # 相应减少batch size
        self.epochs = 200
        self.gradient_accumulation_steps = 3  # 模拟batch_size=12


# Jittor特定优化
def setup_jittor_for_4060():
    """为RTX 4060设置Jittor优化"""
    import jittor as jt
    
    # 基础设置
    jt.flags.use_cuda = 1
    jt.flags.lazy_execution = 1
    
    # 显存优化
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 编译优化
    jt.flags.nvcc_flags += ' -O3'
    
    print("✅ Jittor optimized for RTX 4060 8GB")


# 使用示例
def get_config(stage='normal'):
    """获取对应阶段的配置"""
    if stage == 'quick':
        return QuickTestConfig()
    elif stage == 'full':
        return FullTrainConfig()
    else:
        return TrainConfig()


if __name__ == "__main__":
    # 测试配置
    config = get_config('normal')
    print(f"Dataset: {config.dataset_type}")
    print(f"Classes: {config.num_classes}")
    print(f"Input size: {config.input_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Mixed precision: {config.mixed_precision}")
