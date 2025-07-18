#!/bin/bash
# Gold-YOLO Jittor vs PyTorch 对齐验证实验自动化脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置参数
COCO_PATH="/path/to/coco/dataset"  # 请修改为实际COCO数据集路径
NUM_IMAGES=1000
NUM_CLASSES=10
EPOCHS=100
BATCH_SIZE=6
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 目录设置
PROJECT_ROOT=$(pwd)
DATA_DIR="$PROJECT_ROOT/data/alignment_dataset"
JITTOR_EXP_DIR="$PROJECT_ROOT/experiments/jittor_${TIMESTAMP}"
PYTORCH_EXP_DIR="$PROJECT_ROOT/experiments/pytorch_${TIMESTAMP}"
COMPARISON_DIR="$PROJECT_ROOT/experiments/comparison_${TIMESTAMP}"

print_info "🚀 开始Gold-YOLO对齐验证实验"
print_info "📅 实验时间戳: $TIMESTAMP"
print_info "📁 项目根目录: $PROJECT_ROOT"

# 检查环境
check_environment() {
    print_info "🔍 检查实验环境..."
    
    # 检查conda环境
    if ! command -v conda &> /dev/null; then
        print_error "Conda未安装，请先安装Anaconda或Miniconda"
        exit 1
    fi
    
    # 检查Jittor环境
    if ! conda env list | grep -q "jt"; then
        print_warning "Jittor环境(jt)不存在，请先创建"
        print_info "运行: conda create -n jt python=3.7"
        exit 1
    fi
    
    # 检查PyTorch环境
    if ! conda env list | grep -q "yolo_py"; then
        print_warning "PyTorch环境(yolo_py)不存在，请先创建"
        print_info "运行: conda create -n yolo_py python=3.8"
        exit 1
    fi
    
    # 检查COCO数据集路径
    if [ ! -d "$COCO_PATH" ]; then
        print_error "COCO数据集路径不存在: $COCO_PATH"
        print_info "请修改脚本中的COCO_PATH变量"
        exit 1
    fi
    
    print_success "环境检查通过"
}

# 准备数据集
prepare_dataset() {
    print_info "📊 准备对齐实验数据集..."
    
    # 激活Jittor环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate jt
    
    # 准备数据集
    python scripts/prepare_data.py \
        --source "$COCO_PATH" \
        --target "$DATA_DIR" \
        --num_images $NUM_IMAGES \
        --seed 42 \
        --split
    
    if [ $? -eq 0 ]; then
        print_success "数据集准备完成: $DATA_DIR"
    else
        print_error "数据集准备失败"
        exit 1
    fi
}

# Jittor训练
train_jittor() {
    print_info "🎯 开始Jittor训练..."
    
    # 激活Jittor环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate jt
    
    # 创建实验目录
    mkdir -p "$JITTOR_EXP_DIR"
    
    # 开始训练
    python scripts/train.py \
        --data "$DATA_DIR/dataset.yaml" \
        --num_classes $NUM_CLASSES \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr 0.01 \
        --output_dir "$JITTOR_EXP_DIR" \
        2>&1 | tee "$JITTOR_EXP_DIR/train_output.log"
    
    if [ $? -eq 0 ]; then
        print_success "Jittor训练完成: $JITTOR_EXP_DIR"
    else
        print_error "Jittor训练失败"
        exit 1
    fi
}

# PyTorch训练
train_pytorch() {
    print_info "🔥 开始PyTorch训练..."
    
    # 激活PyTorch环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate yolo_py
    
    # 创建实验目录
    mkdir -p "$PYTORCH_EXP_DIR"
    
    # 切换到PyTorch项目目录
    cd "../Gold-YOLO_pytorch"
    
    # 开始训练 (这里需要根据实际的PyTorch训练脚本调整)
    python tools/train.py \
        --data "$DATA_DIR/dataset.yaml" \
        --cfg configs/gold_yolo-s.py \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --device 0 \
        --project "$PYTORCH_EXP_DIR" \
        2>&1 | tee "$PYTORCH_EXP_DIR/train_output.log"
    
    # 返回Jittor项目目录
    cd "$PROJECT_ROOT"
    
    if [ $? -eq 0 ]; then
        print_success "PyTorch训练完成: $PYTORCH_EXP_DIR"
    else
        print_error "PyTorch训练失败"
        exit 1
    fi
}

# Jittor测试
test_jittor() {
    print_info "🧪 开始Jittor测试..."
    
    # 激活Jittor环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate jt
    
    # 运行测试
    python scripts/test.py \
        --weights "$JITTOR_EXP_DIR/best.pkl" \
        --num_classes $NUM_CLASSES \
        --data "$DATA_DIR" \
        --output_dir "$JITTOR_EXP_DIR/test_results" \
        2>&1 | tee "$JITTOR_EXP_DIR/test_output.log"
    
    if [ $? -eq 0 ]; then
        print_success "Jittor测试完成"
    else
        print_error "Jittor测试失败"
        exit 1
    fi
}

# 生成对比报告
generate_comparison() {
    print_info "📋 生成对比报告..."
    
    # 激活Jittor环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate jt
    
    # 创建对比目录
    mkdir -p "$COMPARISON_DIR"
    
    # 生成对比报告
    python scripts/test.py \
        --weights "$JITTOR_EXP_DIR/best.pkl" \
        --pytorch_results "$PYTORCH_EXP_DIR/test_results.json" \
        --output_dir "$COMPARISON_DIR" \
        2>&1 | tee "$COMPARISON_DIR/comparison_output.log"
    
    if [ $? -eq 0 ]; then
        print_success "对比报告生成完成: $COMPARISON_DIR"
    else
        print_error "对比报告生成失败"
        exit 1
    fi
}

# 主函数
main() {
    print_info "🎯 Gold-YOLO Jittor vs PyTorch 对齐验证实验"
    print_info "================================================"
    
    # 检查环境
    check_environment
    
    # 准备数据集
    prepare_dataset
    
    # 并行训练 (可选)
    if [ "$1" = "--parallel" ]; then
        print_info "🔄 并行训练模式"
        train_jittor &
        JITTOR_PID=$!
        train_pytorch &
        PYTORCH_PID=$!
        
        wait $JITTOR_PID
        wait $PYTORCH_PID
    else
        print_info "🔄 顺序训练模式"
        train_jittor
        train_pytorch
    fi
    
    # 测试和对比
    test_jittor
    generate_comparison
    
    # 实验总结
    print_success "🎉 对齐验证实验完成!"
    print_info "📊 实验结果:"
    print_info "  - Jittor实验: $JITTOR_EXP_DIR"
    print_info "  - PyTorch实验: $PYTORCH_EXP_DIR"
    print_info "  - 对比报告: $COMPARISON_DIR"
    print_info "📋 查看详细报告: $COMPARISON_DIR/alignment_summary.html"
}

# 运行主函数
main "$@"
