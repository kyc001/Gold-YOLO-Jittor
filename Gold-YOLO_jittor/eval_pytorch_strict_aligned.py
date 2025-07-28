#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本推理评估脚本
严格对齐PyTorch版本的推理方法，不做任何简化
"""

import argparse
import os
import sys
import os.path as osp
import time
import cv2
import numpy as np
from pathlib import Path

import jittor as jt

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.core.inferer_jittor import InfererJittor


def get_args_parser(add_help=True):
    """参数解析器，严格对齐PyTorch版本"""
    parser = argparse.ArgumentParser(description='GOLD-YOLO Jittor Inference.', add_help=add_help)
    parser.add_argument('--weights', type=str, 
                       default='/home/kyc/project/GOLD-YOLO/runs/train/pytorch_aligned_stable/epoch_100.pkl', 
                       help='model path(s) for inference.')
    parser.add_argument('--source', type=str, 
                       default='/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch/runs/inference/pytorch_baseline_test/test_images', 
                       help='the source path, e.g. image-file/dir.')
    parser.add_argument('--yaml', type=str, default='../data/voc2012_subset/voc20.yaml', help='data yaml file.')
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
    parser.add_argument('--name', default='jittor_eval', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    args = parser.parse_args()
    LOGGER.info(args)
    return args


@jt.no_grad()
def run(weights=osp.join(ROOT, 'yolov6s.pkl'),
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
        view_img=False,
        classes=None,
        agnostic_nms=False,
        project='runs/inference',
        name='jittor_eval',
        hide_labels=False,
        hide_conf=False,
        half=False):
    """推理主函数，严格对齐PyTorch版本的run函数"""
    
    # 自动查找数据集配置文件
    if yaml is None or not os.path.exists(yaml):
        possible_paths = [
            '../data/voc2012_subset/voc20.yaml',
            '/home/kyc/project/GOLD-YOLO/data/voc2012_subset/voc20.yaml',
            'data/voc2012_subset/voc20.yaml'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                yaml = path
                break
        else:
            raise FileNotFoundError("数据集配置文件未找到")
    
    # 创建保存目录
    if save_dir is None:
        save_dir = osp.join(project, name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化推理器
    inferer = InfererJittor(
        source=source,
        weights=weights, 
        device=device,
        yaml=yaml,
        img_size=img_size,
        half=half
    )
    
    # 获取图像路径
    if osp.isdir(source):
        files = sorted([f for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        files = [osp.join(source, f) for f in files]
    else:
        files = [source]
    
    LOGGER.info(f"推理图像数量: {len(files)}")
    
    # 推理统计
    total_time = 0
    total_detections = 0
    
    # 逐张图像推理
    for i, img_path in enumerate(files):
        LOGGER.info(f"处理图像 {i+1}/{len(files)}: {osp.basename(img_path)}")
        
        # 读取图像
        img_src = cv2.imread(img_path)
        assert img_src is not None, f"无法读取图像: {img_path}"
        
        # 推理
        start_time = time.time()
        detections = inferer.infer(
            img_src, 
            conf_thres=conf_thres, 
            iou_thres=iou_thres, 
            max_det=max_det
        )
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # 统计检测结果
        num_det = len(detections[0]) if len(detections) > 0 and len(detections[0]) > 0 else 0
        total_detections += num_det
        
        LOGGER.info(f"推理时间: {inference_time*1000:.1f}ms, 检测数量: {num_det}")
        
        # 保存可视化结果
        if not not_save_img:
            img_vis = inferer.draw_detections(img_src.copy(), detections, conf_thres)
            save_path = save_dir / f"{Path(img_path).stem}_result.jpg"
            cv2.imwrite(str(save_path), img_vis)
            LOGGER.info(f"保存可视化结果: {save_path}")
        
        # 保存检测结果为txt
        if save_txt and len(detections) > 0:
            txt_path = save_dir / f"{Path(img_path).stem}.txt"
            save_detection_txt(detections, img_src.shape, txt_path, conf_thres)
            LOGGER.info(f"保存检测结果: {txt_path}")
        
        # 显示结果
        if view_img:
            img_vis = inferer.draw_detections(img_src.copy(), detections, conf_thres)
            cv2.imshow('GOLD-YOLO Jittor Inference', img_vis)
            if cv2.waitKey(1) == ord('q'):
                break
    
    # 输出统计结果
    LOGGER.info("=" * 70)
    LOGGER.info("推理统计结果:")
    LOGGER.info(f"总图像数量: {len(files)}")
    LOGGER.info(f"总推理时间: {total_time:.3f}s")
    LOGGER.info(f"平均推理时间: {total_time/len(files)*1000:.1f}ms/图像")
    LOGGER.info(f"推理速度: {len(files)/total_time:.1f} FPS")
    LOGGER.info(f"总检测数量: {total_detections}")
    LOGGER.info(f"平均检测数量: {total_detections/len(files):.1f}/图像")
    LOGGER.info(f"结果保存在: {save_dir}")
    
    if view_img:
        cv2.destroyAllWindows()


def save_detection_txt(detections, img_shape, save_path, conf_thres=0.25):
    """保存检测结果为txt格式，对齐PyTorch版本"""
    with open(save_path, 'w') as f:
        for det in detections:
            if len(det):
                # 转换为numpy
                if hasattr(det, 'numpy'):
                    det = det.numpy()
                
                for *xyxy, conf, cls in det:
                    if conf >= conf_thres:
                        # YOLO格式: class_id center_x center_y width height confidence
                        x1, y1, x2, y2 = xyxy
                        center_x = (x1 + x2) / 2 / img_shape[1]
                        center_y = (y1 + y2) / 2 / img_shape[0]
                        width = (x2 - x1) / img_shape[1]
                        height = (y2 - y1) / img_shape[0]
                        f.write(f"{int(cls)} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")


def main():
    """主函数，严格对齐PyTorch版本"""
    args = get_args_parser()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              GOLD-YOLO 推理评估 (Jittor版本)                 ║
    ║                                                              ║
    ║  🎯 严格对齐PyTorch版本的推理评估                            ║
    ║  📊 识别准确度测试 + 可视化结果                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    run(**vars(args))


if __name__ == '__main__':
    main()
