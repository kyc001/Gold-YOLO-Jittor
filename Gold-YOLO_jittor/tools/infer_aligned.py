#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完全对齐PyTorch版本的Gold-YOLO推理脚本
严格按照PyTorch版本的推理逻辑和后处理
"""

import argparse
import os
import sys
import time
import yaml
import os.path as osp
from pathlib import Path

import jittor as jt
import jittor.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 设置Jittor
jt.flags.use_cuda = 1

# 添加项目路径
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 导入Gold-YOLO组件
from gold_yolo.models.gold_yolo import create_gold_yolo
from gold_yolo.utils.general import increment_name, letterbox, scale_coords
from gold_yolo.utils.events import LOGGER
from gold_yolo.utils.postprocess import non_max_suppression


def get_args_parser(add_help=True):
    """完全对齐PyTorch版本的推理参数解析器"""
    parser = argparse.ArgumentParser(description='Gold-YOLO Jittor Inference', add_help=add_help)
    
    # 模型和数据参数
    parser.add_argument('--weights', type=str, default='weights/gold_yolo_s.pt', help='model path(s) for inference')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help='data yaml file')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image-size(h,w) in inference size')
    
    # 推理参数
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold for inference')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for inference')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image')
    
    # 设备参数
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference')
    
    # 输出参数
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in')
    parser.add_argument('--view-img', action='store_true', help='show inference results')
    
    # 过滤参数
    parser.add_argument('--classes', nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    
    # 项目参数
    parser.add_argument('--project', default='runs/inference', help='save inference results to project/name')
    parser.add_argument('--name', default='exp', help='save inference results to project/name')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    
    return parser


class AlignedInferer:
    """完全对齐PyTorch版本的推理器"""
    
    def __init__(self, args):
        self.args = args
        
        # 设置设备
        if args.device == 'cpu':
            jt.flags.use_cuda = 0
        else:
            jt.flags.use_cuda = 1
            
        # 处理图像尺寸
        if len(args.img_size) == 1:
            self.img_size = [args.img_size[0], args.img_size[0]]
        else:
            self.img_size = args.img_size[:2]
            
        # 创建保存目录
        if args.save_dir:
            self.save_dir = args.save_dir
        else:
            self.save_dir = str(increment_name(osp.join(args.project, args.name)))
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 加载类别名称
        self.class_names = self.load_class_names()
        
        LOGGER.info(f'🎯 对齐推理器初始化完成')
        LOGGER.info(f'   保存目录: {self.save_dir}')
        LOGGER.info(f'   设备: {"CUDA" if jt.flags.use_cuda else "CPU"}')
        LOGGER.info(f'   图像尺寸: {self.img_size}')
        
    def load_class_names(self):
        """加载类别名称 - 对齐PyTorch版本"""
        if os.path.exists(self.args.yaml):
            with open(self.args.yaml, 'r') as f:
                data_cfg = yaml.safe_load(f)
                return data_cfg.get('names', [f'class_{i}' for i in range(80)])
        else:
            # COCO默认类别
            return [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            
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
            num_classes=len(self.class_names),
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
        LOGGER.info(f'   类别数: {len(self.class_names)}')
        
        return self.model
        
    def preprocess_image(self, img_path):
        """图像预处理 - 对齐PyTorch版本"""
        # 读取图像
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            raise ValueError(f'Cannot read image: {img_path}')
            
        # 记录原始尺寸
        h0, w0 = img0.shape[:2]
        
        # Letterbox resize
        img, ratio, (dw, dh) = letterbox(img0, self.img_size, stride=32, auto=False)
        
        # 转换格式
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = jt.array(img).float() / 255.0
        
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
            
        return img, img0, (h0, w0), ratio, (dw, dh)
        
    def postprocess(self, predictions, img_shape, orig_shape, ratio, pad):
        """后处理 - 对齐PyTorch版本"""
        # NMS
        predictions = non_max_suppression(
            predictions,
            conf_thres=self.args.conf_thres,
            iou_thres=self.args.iou_thres,
            classes=self.args.classes,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det
        )
        
        # 处理每个图像的预测结果
        results = []
        for i, pred in enumerate(predictions):
            if len(pred):
                # 坐标缩放回原图
                pred[:, :4] = scale_coords(img_shape[2:], pred[:, :4], orig_shape).round()
                
                # 转换为列表格式
                for *xyxy, conf, cls in pred:
                    results.append({
                        'bbox': [float(x) for x in xyxy],
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': self.class_names[int(cls)]
                    })
                    
        return results
        
    def draw_results(self, img, results):
        """绘制结果 - 对齐PyTorch版本"""
        for result in results:
            x1, y1, x2, y2 = [int(x) for x in result['bbox']]
            conf = result['confidence']
            cls = result['class']
            class_name = result['class_name']
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            if not self.args.hide_labels:
                label = f'{class_name}'
                if not self.args.hide_conf:
                    label += f' {conf:.2f}'
                    
                # 计算文本尺寸
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # 绘制背景
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # 绘制文本
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return img
        
    def save_results(self, results, img_path, save_txt=False):
        """保存结果 - 对齐PyTorch版本"""
        if save_txt:
            # 保存为txt格式
            txt_path = osp.join(self.save_dir, Path(img_path).stem + '.txt')
            with open(txt_path, 'w') as f:
                for result in results:
                    x1, y1, x2, y2 = result['bbox']
                    conf = result['confidence']
                    cls = result['class']
                    f.write(f'{cls} {x1} {y1} {x2} {y2} {conf}\n')
                    
    def run_inference(self):
        """运行推理 - 对齐PyTorch版本"""
        LOGGER.info(f'🚀 开始推理...')
        
        # 加载模型
        model = self.load_model()
        
        # 获取输入文件列表
        source_path = Path(self.args.source)
        if source_path.is_file():
            img_paths = [source_path]
        elif source_path.is_dir():
            img_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')) + list(source_path.glob('*.jpeg'))
        else:
            raise ValueError(f'Invalid source: {self.args.source}')
            
        LOGGER.info(f'   找到 {len(img_paths)} 张图片')
        
        # 推理循环
        start_time = time.time()
        total_results = []
        
        with jt.no_grad():
            for img_path in img_paths:
                # 预处理
                img, img0, orig_shape, ratio, pad = self.preprocess_image(img_path)
                
                # 推理
                predictions = model(img)
                
                # 后处理
                results = self.postprocess(predictions, img.shape, orig_shape, ratio, pad)
                
                # 保存结果
                if not self.args.not_save_img:
                    # 绘制结果
                    img_with_results = self.draw_results(img0.copy(), results)
                    
                    # 保存图片
                    save_path = osp.join(self.save_dir, Path(img_path).name)
                    cv2.imwrite(save_path, img_with_results)
                    
                if self.args.save_txt:
                    self.save_results(results, img_path, save_txt=True)
                    
                # 显示结果
                if self.args.view_img:
                    cv2.imshow('Results', img_with_results)
                    cv2.waitKey(1)
                    
                total_results.append({
                    'image': str(img_path),
                    'detections': results
                })
                
                LOGGER.info(f'   {Path(img_path).name}: {len(results)} 个检测结果')
                
        inference_time = time.time() - start_time
        
        LOGGER.info(f'🎉 推理完成!')
        LOGGER.info(f'   用时: {inference_time:.2f}s')
        LOGGER.info(f'   平均: {inference_time/len(img_paths):.3f}s/张')
        LOGGER.info(f'   结果保存至: {self.save_dir}')
        
        return total_results


@jt.no_grad()
def main(args):
    """主推理函数 - 完全对齐PyTorch版本"""
    LOGGER.info(f'🎯 开始Gold-YOLO Jittor推理')
    LOGGER.info(f'推理参数: {args}\n')
    
    # 创建推理器
    inferer = AlignedInferer(args)
    
    # 运行推理
    results = inferer.run_inference()
    
    return results


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
