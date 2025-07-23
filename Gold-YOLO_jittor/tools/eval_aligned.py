#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完全对齐PyTorch版本的Gold-YOLO评估脚本
严格按照PyTorch版本的评估逻辑和指标计算
"""

import argparse
import os
import sys
import time
import yaml
import json
import os.path as osp
from pathlib import Path

import jittor as jt
import jittor.nn as nn
import numpy as np
from tqdm import tqdm

# 设置Jittor
jt.flags.use_cuda = 1

# 添加项目路径
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入Gold-YOLO组件
from gold_yolo.models.gold_yolo import create_gold_yolo
from gold_yolo.data.coco_dataset import COCODataset
from gold_yolo.utils.general import increment_name
from gold_yolo.utils.events import LOGGER
from gold_yolo.utils.metrics import COCOEvaluator


def boolean_string(s):
    """布尔字符串转换 - 对齐PyTorch版本"""
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args_parser(add_help=True):
    """完全对齐PyTorch版本的评估参数解析器"""
    parser = argparse.ArgumentParser(description='Gold-YOLO Jittor Evaluation', add_help=add_help)
    
    # 数据和模型参数
    parser.add_argument('--data', type=str, default='./data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='./weights/gold_yolo_s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    
    # 推理参数
    parser.add_argument('--conf-thres', type=float, default=0.03, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    
    # 任务和设备
    parser.add_argument('--task', default='val', help='val, test, or speed')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', default=False, action='store_true', help='whether to use fp16 infer')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='runs/val/', help='evaluation save dir')
    parser.add_argument('--name', type=str, default='exp', help='save evaluation results to save_dir/name')
    
    # 评估控制参数
    parser.add_argument('--test_load_size', type=int, default=640, help='load img resize when test')
    parser.add_argument('--letterbox_return_int', default=False, action='store_true', help='return int offset for letterbox')
    parser.add_argument('--scale_exact', default=False, action='store_true', help='use exact scale size to scale coords')
    parser.add_argument('--force_no_pad', default=False, action='store_true', help='for no extra pad in letterbox')
    parser.add_argument('--not_infer_on_rect', default=False, action='store_true', help='default to use rect image size to boost infer')
    
    # 指标计算参数
    parser.add_argument('--do_coco_metric', default=True, type=boolean_string, help='whether to use pycocotool to metric')
    parser.add_argument('--do_pr_metric', default=False, type=boolean_string, help='whether to calculate precision, recall and F1')
    parser.add_argument('--plot_curve', default=True, type=boolean_string, help='whether to save plots in savedir')
    parser.add_argument('--plot_confusion_matrix', default=False, action='store_true', help='whether to save confusion matrix plots')
    parser.add_argument('--verbose', default=False, action='store_true', help='whether to print metric on each class')
    
    # 配置文件
    parser.add_argument('--config-file', default='', type=str, help='experiments description file')
    
    return parser


class AlignedEvaler:
    """完全对齐PyTorch版本的评估器"""
    
    def __init__(self, args):
        self.args = args
        
        # 设置设备
        if args.device == 'cpu':
            jt.flags.use_cuda = 0
        else:
            jt.flags.use_cuda = 1
            
        # 创建保存目录
        self.save_dir = str(increment_name(osp.join(args.save_dir, args.name)))
        os.makedirs(self.save_dir, exist_ok=True)
        
        LOGGER.info(f'🎯 对齐评估器初始化完成')
        LOGGER.info(f'   保存目录: {self.save_dir}')
        LOGGER.info(f'   设备: {"CUDA" if jt.flags.use_cuda else "CPU"}')
        
    def load_model(self):
        """加载模型 - 对齐PyTorch版本"""
        # 从权重文件名推断模型版本
        model_version = 's'  # 默认
        if 'gold_yolo_n' in self.args.weights or 'gold-yolo-n' in self.args.weights:
            model_version = 'n'
        elif 'gold_yolo_m' in self.args.weights or 'gold-yolo-m' in self.args.weights:
            model_version = 'm'
        elif 'gold_yolo_l' in self.args.weights or 'gold-yolo-l' in self.args.weights:
            model_version = 'l'
            
        # 创建模型
        self.model = create_gold_yolo(
            model_version,
            num_classes=80,  # COCO数据集
            use_pytorch_components=False
        )
        
        # 加载权重
        if os.path.exists(self.args.weights):
            checkpoint = jt.load(self.args.weights)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            LOGGER.info(f'✅ 权重加载完成: {self.args.weights}')
        else:
            LOGGER.warning(f'⚠️ 权重文件不存在: {self.args.weights}，使用随机初始化权重')
            
        self.model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(f'✅ 模型加载完成: Gold-YOLO-{model_version.upper()}')
        LOGGER.info(f'   参数量: {total_params:,} ({total_params/1e6:.2f}M)')
        
        return self.model
        
    def setup_data(self):
        """设置数据加载器 - 对齐PyTorch版本"""
        # 加载数据配置
        with open(self.args.data, 'r') as f:
            data_cfg = yaml.safe_load(f)
            
        # 根据任务选择数据集
        if self.args.task == 'val':
            img_dir = data_cfg['val']
            ann_file = data_cfg.get('val_ann', None)
        elif self.args.task == 'test':
            img_dir = data_cfg.get('test', data_cfg['val'])
            ann_file = data_cfg.get('test_ann', data_cfg.get('val_ann', None))
        else:
            raise ValueError(f'Unsupported task: {self.args.task}')
            
        # 创建数据集
        dataset = COCODataset(
            img_dir=img_dir,
            ann_file=ann_file,
            img_size=self.args.img_size,
            augment=False,
            cache=False,
            rect=not self.args.not_infer_on_rect,
            stride=32
        )
        
        # 创建数据加载器
        self.dataloader = dataset.set_attrs(
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        
        LOGGER.info(f'✅ 数据加载器创建完成')
        LOGGER.info(f'   数据集: {len(dataset)} 张图片')
        LOGGER.info(f'   批次大小: {self.args.batch_size}')
        
        return self.dataloader, dataset
        
    def setup_evaluator(self):
        """设置评估器 - 对齐PyTorch版本"""
        self.evaluator = COCOEvaluator(
            save_dir=self.save_dir,
            conf_thres=self.args.conf_thres,
            iou_thres=self.args.iou_thres,
            max_det=self.args.max_det,
            do_coco_metric=self.args.do_coco_metric,
            do_pr_metric=self.args.do_pr_metric,
            plot_curve=self.args.plot_curve,
            plot_confusion_matrix=self.args.plot_confusion_matrix,
            verbose=self.args.verbose
        )
        
        LOGGER.info(f'✅ 评估器创建完成')
        LOGGER.info(f'   置信度阈值: {self.args.conf_thres}')
        LOGGER.info(f'   NMS IoU阈值: {self.args.iou_thres}')
        LOGGER.info(f'   最大检测数: {self.args.max_det}')
        
        return self.evaluator
        
    def run_evaluation(self):
        """运行评估 - 对齐PyTorch版本"""
        LOGGER.info(f'🚀 开始评估...')
        
        # 加载组件
        model = self.load_model()
        dataloader, dataset = self.setup_data()
        evaluator = self.setup_evaluator()
        
        # 评估循环
        start_time = time.time()
        all_predictions = []
        all_targets = []
        
        with jt.no_grad():
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluating')
            
            for batch_idx, (images, targets, paths, shapes) in pbar:
                # 前向传播
                predictions = model(images)
                
                # 后处理
                processed_preds = evaluator.postprocess(
                    predictions, images.shape, shapes
                )
                
                # 收集结果
                all_predictions.extend(processed_preds)
                all_targets.extend(targets)
                
                # 更新进度条
                pbar.set_postfix({
                    'batch': f'{batch_idx+1}/{len(dataloader)}'
                })
                
        eval_time = time.time() - start_time
        
        # 计算指标
        LOGGER.info(f'📊 计算评估指标...')
        metrics = evaluator.compute_metrics(all_predictions, all_targets)
        
        # 保存结果
        results_file = osp.join(self.save_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        LOGGER.info(f'🎉 评估完成!')
        LOGGER.info(f'   用时: {eval_time:.2f}s')
        LOGGER.info(f'   结果保存至: {results_file}')
        
        # 打印主要指标
        if 'mAP_0.5:0.95' in metrics:
            LOGGER.info(f'   mAP@0.5:0.95: {metrics["mAP_0.5:0.95"]:.4f}')
        if 'mAP_0.5' in metrics:
            LOGGER.info(f'   mAP@0.5: {metrics["mAP_0.5"]:.4f}')
            
        return metrics


def main(args):
    """主评估函数 - 完全对齐PyTorch版本"""
    LOGGER.info(f'🎯 开始Gold-YOLO Jittor评估')
    LOGGER.info(f'评估参数: {args}\n')
    
    # 创建评估器
    evaler = AlignedEvaler(args)
    
    # 运行评估
    metrics = evaler.run_evaluation()
    
    return metrics


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
