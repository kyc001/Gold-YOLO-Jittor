#!/bin/bash
# GOLD-YOLO Jittor版本 - 改进的训练脚本
# 完全对齐PyTorch版本的训练配置

echo "🚀 开始改进的Gold-YOLO-n训练 (Jittor版本)"
echo "================================"

# 激活Jittor环境
conda activate jt

# 训练参数 - 完全对齐PyTorch版本
BATCH_SIZE=16          # 增加批次大小
EPOCHS=50             # 大幅增加训练轮数
IMG_SIZE=640
WORKERS=4
CONF_FILE="configs/gold_yolo-n.py"
DATA_PATH="data/voc_subset_improved.yaml"
NAME="gold_yolo_n_improved_jittor"
OUTPUT_DIR="runs/train"

# 学习率和优化器参数 - 对齐PyTorch版本
LR_INITIAL=0.01        # 初始学习率
LR_FINAL=0.001         # 最终学习率
MOMENTUM=0.937
WEIGHT_DECAY=0.0005

# 数据增强参数 - 对齐PyTorch版本
MOSAIC_PROB=1.0        # Mosaic增强概率
MIXUP_PROB=0.1         # Mixup增强概率

echo "📊 训练配置:"
echo "   批次大小: $BATCH_SIZE"
echo "   训练轮数: $EPOCHS"
echo "   图像尺寸: $IMG_SIZE"
echo "   初始学习率: $LR_INITIAL"
echo "   数据增强: Mosaic + Mixup"
echo "   框架: Jittor"

# 检查GPU
echo "🔍 GPU状态:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# 开始训练
echo "🚀 开始训练..."
python tools/train.py \
    --batch-size $BATCH_SIZE \
    --conf-file $CONF_FILE \
    --data-path $DATA_PATH \
    --epochs $EPOCHS \
    --img-size $IMG_SIZE \
    --name $NAME \
    --workers $WORKERS \
    --eval-interval 10 \
    --output-dir $OUTPUT_DIR \
    --lr-initial $LR_INITIAL \
    --lr-final $LR_FINAL \
    --momentum $MOMENTUM \
    --weight-decay $WEIGHT_DECAY \
    --mosaic-prob $MOSAIC_PROB \
    --mixup-prob $MIXUP_PROB \
    --save-interval 50 \
    --log-interval 10

echo "✅ 训练完成!"
echo "📊 检查结果: $OUTPUT_DIR/$NAME"
