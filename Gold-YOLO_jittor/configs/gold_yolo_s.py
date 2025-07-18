# Gold-YOLO-s configuration for Jittor
# Migrated from PyTorch version

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.solver = SolverConfig()
        self.data_aug = DataAugConfig()


class ModelConfig:
    def __init__(self):
        self.type = 'GoldYOLO-s'
        self.pretrained = None
        self.depth_multiple = 0.33
        self.width_multiple = 0.50
        self.backbone = BackboneConfig()
        self.neck = NeckConfig()
        self.head = HeadConfig()


class BackboneConfig:
    def __init__(self):
        self.type = 'EfficientRep'
        self.num_repeats = [1, 6, 12, 18, 6]
        self.out_channels = [64, 128, 256, 512, 1024]
        self.fuse_P2 = True
        self.cspsppf = True

    def get(self, key, default=None):
        return getattr(self, key, default)


class NeckConfig:
    def __init__(self):
        self.type = 'RepGDNeck'
        self.num_repeats = [12, 12, 12, 12]
        self.out_channels = [256, 128, 128, 256, 256, 512]
        self.extra_cfg = ExtraConfig()

    def get(self, key, default=None):
        return getattr(self, key, default)


class ExtraConfig:
    def __init__(self):
        self.norm_cfg = {'type': 'SyncBN', 'requires_grad': True}
        self.depths = 2
        self.fusion_in = 1088  # Actual backbone output: 64+128+256+512+128
        self.ppa_in = 704
        self.fusion_act = {'type': 'ReLU6'}
        self.fuse_block_num = 3
        self.embed_dim_p = 64  # 128 * 0.5 for width_multiple
        self.embed_dim_n = 352  # 704 * 0.5 for width_multiple
        self.key_dim = 8
        self.num_heads = 4
        self.mlp_ratios = 1
        self.attn_ratios = 2
        self.c2t_stride = 2
        self.drop_path_rate = 0.1
        self.trans_channels = [64, 32, 64, 128]  # Apply width_multiple=0.5
        self.pool_mode = 'jittor'


class HeadConfig:
    def __init__(self):
        self.type = 'EffiDeHead'
        self.in_channels = [128, 256, 512]
        self.num_layers = 3
        self.begin_indices = 24
        self.anchors = 3
        self.anchors_init = [
            [10, 13, 19, 19, 33, 23],
            [30, 61, 59, 59, 59, 119],
            [116, 90, 185, 185, 373, 326]
        ]
        self.out_indices = [17, 20, 23]
        self.strides = [8, 16, 32]
        self.atss_warmup_epoch = 0
        self.iou_type = 'giou'
        self.use_dfl = True  # set to True if you want to further train with distillation
        self.reg_max = 16  # set to 16 if you want to further train with distillation
        self.distill_weight = {
            'class': 1.0,
            'dfl': 1.0,
        }


class SolverConfig:
    def __init__(self):
        self.optim = 'SGD'
        self.lr_scheduler = 'Cosine'
        self.lr0 = 0.01
        self.lrf = 0.01
        self.momentum = 0.937
        self.weight_decay = 0.0005
        self.warmup_epochs = 3.0
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1


class DataAugConfig:
    def __init__(self):
        self.hsv_h = 0.015
        self.hsv_s = 0.7
        self.hsv_v = 0.4
        self.degrees = 0.0
        self.translate = 0.1
        self.scale = 0.9
        self.shear = 0.0
        self.flipud = 0.0
        self.fliplr = 0.5
        self.mosaic = 1.0
        self.mixup = 0.1


# Create default config instance
def get_config():
    return Config()
