# 2023.09.18-Changed for checkpoint load implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 评估器模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
from tqdm import tqdm
import numpy as np
import json
import jittor as jt
import yaml
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from gold_yolo.switch_tool import switch_to_deploy
from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER, NCOLS
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.checkpoint import load_checkpoint, load_checkpoint_2
from yolov6.utils.jittor_utils import time_sync, get_model_info

'''
python tools/eval.py --task 'train'/'val'/'speed'
'''


class Evaler:
    def __init__(self,
                 data,
                 batch_size=32,
                 img_size=640,
                 conf_thres=0.03,
                 iou_thres=0.65,
                 device='',
                 half=True,
                 save_dir='',
                 test_load_size=640,
                 letterbox_return_int=False,
                 force_no_pad=False,
                 not_infer_on_rect=False,
                 scale_exact=False,
                 verbose=False,
                 do_coco_metric=True,
                 do_pr_metric=False,
                 plot_curve=True,
                 plot_confusion_matrix=False
                 ):
        assert do_pr_metric or do_coco_metric, 'ERROR: at least set one val metric'
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.half = half
        self.save_dir = save_dir
        self.test_load_size = test_load_size
        self.letterbox_return_int = letterbox_return_int
        self.force_no_pad = force_no_pad
        self.not_infer_on_rect = not_infer_on_rect
        self.scale_exact = scale_exact
        self.verbose = verbose
        self.do_coco_metric = do_coco_metric
        self.do_pr_metric = do_pr_metric
        self.plot_curve = plot_curve
        self.plot_confusion_matrix = plot_confusion_matrix
    
    def init_model(self, model, weights, task, use_cfg=False):
        if task != 'train':
            
            if not use_cfg:
                model = load_checkpoint(weights, map_location=self.device)
            else:
                model = load_checkpoint_2(model, weights, map_location=self.device)
            
            self.stride = int(model.stride.max())
            # Jittor模型预热
            if jt.has_cuda:
                model(jt.zeros(1, 3, self.img_size, self.img_size))
            # switch to deploy
            model = switch_to_deploy(model)
            
            LOGGER.info("Switch model to deploy modality.")
            LOGGER.info("Model Summary: {}".format(get_model_info(model, self.img_size)))
        
        # Jittor的半精度处理
        if self.half and jt.has_cuda:
            model.half()
        else:
            model.float()
        return model
    
    def init_data(self, dataloader, task):
        '''Initialize dataloader.
        Returns a dataloader for task val or speed.
        '''
        self.is_coco = self.data.get("is_coco", False)
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        if task != 'train':
            pad = 0.0 if task == 'speed' else 0.5
            eval_hyp = {
                    "test_load_size": self.test_load_size,
                    "letterbox_return_int": self.letterbox_return_int,
            }
            if self.force_no_pad:
                eval_hyp["force_no_pad"] = self.force_no_pad
            if self.not_infer_on_rect:
                eval_hyp["not_infer_on_rect"] = self.not_infer_on_rect
            if self.scale_exact:
                eval_hyp["scale_exact"] = self.scale_exact
            
            dataloader = create_dataloader(self.data[task if task in ('train', 'val', 'test') else 'val'],
                                           self.img_size, self.batch_size, self.stride, hyp=eval_hyp, rect=True,
                                           rank=-1, pad=pad, data_dict=self.data, task=task)[0]
        return dataloader
    
    def predict_model(self, model, dataloader, task):
        '''Model prediction
        Predicting the model in validation dataset

        Returns:
            Prediction results
        '''
        self.speed_result = jt.zeros(4)  # 使用jt.zeros替代torch.zeros
        self.ap_result = None
        model.eval()
        pred_results = []
        targets = []
        for batch_i, (imgs, targets_batch, paths, shapes) in enumerate(tqdm(dataloader, ncols=NCOLS)):
            
            # preprocess
            targets_batch[:, 2:] *= jt.array([self.img_size, self.img_size, self.img_size, self.img_size])  # 使用jt.array
            
            # Inference
            t1 = time_sync()
            # Jittor不需要torch.no_grad()
            outputs = model(imgs)
            t2 = time_sync()
            
            # post-process
            outputs = non_max_suppression(outputs, self.conf_thres, self.iou_thres, multi_label=True, max_det=300)
            t3 = time_sync()
            
            # Statistics per image
            for si, pred in enumerate(outputs):
                labels = targets_batch[targets_batch[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                path, shape = Path(paths[si]), shapes[si][0]
                
                if len(pred) == 0:
                    if nl:
                        pred_results.append((jt.zeros(0, 6), jt.array(tcls), jt.array([])))
                    continue
                
                # Predictions
                if self.is_coco:
                    pred[:, 5] = pred[:, 5] + 1  # COCO class offset
                
                pred_results.append((pred[:, :6], jt.array(tcls), pred[:, 4]))
                targets.append(labels)
            
            # Speed statistics
            self.speed_result[0] += t2 - t1  # inference time
            self.speed_result[1] += t3 - t2  # NMS time
            self.speed_result[2] += t3 - t1  # total time
            self.speed_result[3] += len(outputs)  # number of images
        
        return pred_results, targets, self.speed_result
    
    def eval_model(self, pred_results, model, dataloader, task):
        '''Evaluate models
        task: 'train' for training, 'val' for validation, 'speed' for speed test
        '''
        LOGGER.info(f'\nEvaluating speed.')
        
        # Print speeds
        self.print_metric(task, self.speed_result)
        
        # Evaluate
        if self.do_coco_metric:
            return self.eval_coco(pred_results, model, dataloader, task)
        elif self.do_pr_metric:
            return self.eval_pr(pred_results, model, dataloader, task)
    
    def print_metric(self, task, speed_result):
        '''Print metric'''
        # Print speeds
        shape = (self.batch_size, 3, self.img_size, self.img_size)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' %
                    tuple(speed_result[:3] / speed_result[3] * 1E3))
        if task != 'train':
            LOGGER.info(f"Results saved to {self.save_dir}")
    
    def box_convert(self, x):
        # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y
    
    def coco80_to_coco91_class(self):  # converts 80-index (val2014) to 91-index (paper)
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x
