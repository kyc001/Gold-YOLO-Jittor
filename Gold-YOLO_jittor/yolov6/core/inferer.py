#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 推理器模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
import cv2
import time
import math
import jittor as jt
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

from gold_yolo.switch_tool import switch_to_deploy
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.jittor_utils import get_model_info


class Inferer:
    def __init__(self, source, webcam, webcam_addr, weights, device, yaml, img_size, half):
        
        self.__dict__.update(locals())
        
        # Init model
        self.device = device
        self.img_size = img_size
        # Jittor自动处理CUDA可用性
        self.device = 'cuda' if jt.has_cuda and device != 'cpu' else 'cpu'
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size
        self.half = half
        
        # Switch model to deploy status
        self.model.model = self.model_switch(self.model.model, self.img_size)
        
        # Half precision
        if self.half and jt.has_cuda:
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False
        
        # 模型预热
        if jt.has_cuda:
            self.model(jt.zeros(1, 3, *self.img_size))  # warmup
        
        # Load data
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        self.files = LoadData(source, webcam, webcam_addr)
        self.source = source
    
    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        model = switch_to_deploy(model)
        LOGGER.info("Switch model to deploy modality.")
        
        return model
    
    def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels,
              hide_conf, view_img=True):
        ''' Model Inference and results visualization '''
        vid_path, vid_writer, windows = None, None, []
        fps_calculator = CalcFPS()
        for img_src, img_path, vid_cap in tqdm(self.files):
            img, img_src = self.precess_image(img_src, self.img_size, self.stride, self.half)
            # Jittor自动处理设备转换
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            t1 = time.time()
            pred_results = self.model(img)
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            t2 = time.time()
            
            if self.webcam:
                save_path = osp.join(save_dir, self.webcam_addr)
                txt_path = osp.join(save_dir, self.webcam_addr)
            else:
                # Create output files in nested dirs that mirrors the structure of the images' dirs
                rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.source))
                save_path = osp.join(save_dir, rel_path, osp.basename(img_path))  # im.jpg
                txt_path = osp.join(save_dir, rel_path, osp.splitext(osp.basename(img_path))[0])
                os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)
            
            gn = jt.array(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh，使用jt.array
            img_ori = img_src.copy()
            
            # check image and font
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            self.font_check()
            
            # FPS counter
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()
            
            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    LOGGER.info(f"{n} {self.class_names[int(c)]}{'s' * (n > 1)}, ")  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (self.box_convert(jt.array(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if save_img or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (self.class_names[c] if hide_conf else f'{self.class_names[c]} {conf:.2f}')
                        self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label,
                                                color=self.generate_colors(c, True))
                
                img_src = np.asarray(img_ori)
            
            # Stream results
            if view_img:
                if self.webcam:
                    cv2.imshow(str(self.webcam_addr), img_src)
                    cv2.waitKey(1)  # 1 millisecond
                else:
                    cv2.imshow(str(img_path), img_src)
                    cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(img_src)
            
            # Print time (inference + NMS)
            LOGGER.info(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({avg_fps:.1f} FPS)')
    
    @staticmethod
    def precess_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = jt.array(image)  # 使用jt.array替代torch.from_numpy
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0
        
        return image, img_src
    
    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        gain = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        pad = (ori_shape[1] - target_shape[1] * gain) / 2, (ori_shape[0] - target_shape[0] * gain) / 2
        
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= gain
        
        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2
        
        return boxes
    
    def check_img_size(self, img_size, s=32):
        """Check image size and make it divisible by stride s"""
        def make_divisible(x, divisor):
            # Returns x evenly divisible by divisor
            return math.ceil(x / divisor) * divisor
        
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(make_divisible(img_size, int(s)), int(s))
        else:  # list i.e. img_size=[640, 480]
            new_size = [max(make_divisible(x, int(s)), int(s)) for x in img_size]
        
        if new_size != img_size:
            LOGGER.warning(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size, list) else [new_size, new_size]
    
    def font_check(self, font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)
    
    @staticmethod
    def box_convert(x):
        # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y
    
    def generate_colors(self, i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = [tuple(int(hex[i % len(hex)][j:j + 2], 16) for j in (0, 2, 4)) for i in range(len(hex))]
        n = len(palette)
        c = palette[int(i) % n]
        return (c[2], c[1], c[0]) if bgr else c


class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)
    
    def update(self, duration: float):
        self.framerate.append(duration)
    
    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
