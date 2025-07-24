#!/bin/bash
# Gold-YOLO-N Jittor版本训练启动脚本
# 严格对齐PyTorch版本的所有训练参数

echo "🚀 Gold-YOLO-N Jittor版本训练"
echo "================================"
echo "严格对齐PyTorch版本的训练参数配置"
echo ""

# 激活Jittor环境
echo "🔧 激活Jittor环境..."
conda activate jt

# 检查环境状态
echo "🔍 环境检查:"
echo "   Python版本: $(python --version)"
echo "   Jittor版本: $(python -c 'import jittor as jt; print(jt.__version__)')"
echo "   CUDA可用: $(python -c 'import jittor as jt; print(jt.has_cuda)')"

# 检查GPU状态
echo ""
echo "🔍 GPU状态:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

# 训练参数 - 严格对齐PyTorch版本improved_train.py
BATCH_SIZE=16          # 对齐PyTorch版本
EPOCHS=200             # 对齐PyTorch版本
IMG_SIZE=640           # 对齐PyTorch版本
DEVICE=0
WORKERS=4              # 对齐PyTorch版本
CONF_FILE="configs/gold_yolo-n.py"
NAME="gold_yolo_n_jittor_improved"
OUTPUT_DIR="runs/train"
EVAL_INTERVAL=10       # 对齐PyTorch版本

echo ""
echo "📊 训练配置 (对齐PyTorch版本):"
echo "   批次大小: $BATCH_SIZE"
echo "   训练轮数: $EPOCHS"
echo "   图像尺寸: $IMG_SIZE"
echo "   评估间隔: $EVAL_INTERVAL"
echo "   实验名称: $NAME"
echo "   输出目录: $OUTPUT_DIR"

# 学习率和优化器参数 (在Python脚本中设置，对齐PyTorch版本)
echo "   初始学习率: 0.01"
echo "   最终学习率: 0.001"
echo "   动量: 0.937"
echo "   权重衰减: 0.0005"

# 数据增强参数 (在Python脚本中设置，对齐PyTorch版本)
echo "   Mosaic增强: 1.0"
echo "   Mixup增强: 0.1"

echo ""
echo "🚀 开始训练..."
echo "================================"

# 开始训练 - 使用与PyTorch版本完全对齐的参数
python train_gold_yolo_jittor.py \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --img-size $IMG_SIZE \
    --device $DEVICE \
    --workers $WORKERS \
    --conf-file $CONF_FILE \
    --name $NAME \
    --output-dir $OUTPUT_DIR \
    --eval-interval $EVAL_INTERVAL \
    --verbose

echo ""
echo "✅ 训练脚本执行完成!"
echo ""
echo "📋 训练结果查看:"
echo "   训练日志: $OUTPUT_DIR/$NAME/"
echo "   最佳模型: $OUTPUT_DIR/$NAME/best.pkl"
echo "   最新模型: $OUTPUT_DIR/$NAME/latest.pkl"
echo ""
echo "🎯 下一步操作:"
echo "   1. 查看训练损失曲线"
echo "   2. 使用最佳模型进行推理测试"
echo "   3. 与PyTorch版本进行性能对比"
echo "   4. 分析模型性能指标"
