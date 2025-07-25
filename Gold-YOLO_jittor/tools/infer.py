#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 推理脚本
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import argparse
import os
import sys
import os.path as osp
import jittor as jt

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='GOLD-YOLO Jittor Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, default='weights/goldyolo_n.pkl', help='model path(s) for inference.')
    parser.add_argument('--source', type=str, default='data/images', help='the source path, e.g. image-file/dir.')
    parser.add_argument('--yaml', type=str, default='data/coco.yaml', help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', action='store_true', help='show inference results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default='runs/inference', help='save inference results to project/name.')
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    args = parser.parse_args()
    LOGGER.info(args)
    return args


def run(weights=osp.join(ROOT, 'goldyolo_n.pkl'),
        source=osp.join(ROOT, 'data/images'),
        yaml=None,
        img_size=640,
        conf_thres=0.4,
        iou_thres=0.45,
        max_det=1000,
        device='',
        save_txt=False,
        not_save_img=False,
        save_dir=None,
        view_img=True,
        classes=None,
        agnostic_nms=False,
        project=osp.join(ROOT, 'runs/inference'),
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False,
        ):
    """ Inference process, supporting inference on one image file or directory which containing images.
    Args:
        weights: The path of model.pkl, e.g. goldyolo_n.pkl
        source: Source path, supporting image files or dirs containing images.
        yaml: Data yaml file, .
        img_size: Inference image-size, e.g. 640
        conf_thres: Confidence threshold in inference, e.g. 0.25
        iou_thres: NMS IOU threshold in inference, e.g. 0.45
        max_det: Maximal detections per image, e.g. 1000
        device: Cuda device, e.e. 0, or 0,1,2,3 or cpu
        save_txt: Save results to *.txt
        not_save_img: Do not save visualized inference results
        classes: Filter by class: --class 0, or --class 0 2 3
        agnostic_nms: Class-agnostic NMS
        project: Save results to project/name
        name: Save results to project/name
        hide_labels: Hide labels
        hide_conf: Hide confidences
        half: Use FP16 half-precision inference
    """
    
    # Jittor设置
    if jt.has_cuda and device != 'cpu':
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    
    # 处理图像尺寸
    if isinstance(img_size, list):
        assert len(img_size) <= 2
        if len(img_size) == 2:
            img_size = max(img_size)
        else:
            img_size = img_size[0]
    
    # 检查源路径
    if not osp.exists(source):
        LOGGER.warning(f'Source path {source} does not exist')
        return
    
    # 检查权重文件
    if not osp.exists(weights):
        LOGGER.warning(f'Weights file {weights} does not exist')
        return
    
    # 检查yaml文件
    if yaml and not osp.exists(yaml):
        LOGGER.warning(f'YAML file {yaml} does not exist')
        yaml = 'data/coco.yaml'  # 使用默认yaml
    
    # 创建推理器
    inferer = Inferer(source, False, '', weights, device, yaml, img_size, half)
    
    # 设置保存目录
    if save_dir is None:
        save_dir = osp.join(project, name)
    
    # 开始推理
    inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, 
                 save_txt, not not_save_img, hide_labels, hide_conf, view_img)
    
    LOGGER.info(f'Inference completed. Results saved to {save_dir}')


def main():
    args = get_args_parser()
    
    # 运行推理
    run(**vars(args))


if __name__ == '__main__':
    main()
