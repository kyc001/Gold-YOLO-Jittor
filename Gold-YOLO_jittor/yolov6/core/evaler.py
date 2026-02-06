# 2023.09.18-Changed for checkpoint load implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 评估器模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
import json
from pathlib import Path

import jittor as jt
import numpy as np
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from gold_yolo.switch_tool import switch_to_deploy
from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER, NCOLS
from yolov6.utils.nms import non_max_suppression, xywh2xyxy
from yolov6.utils.checkpoint import load_checkpoint, load_checkpoint_2
from yolov6.utils.jittor_utils import time_sync, get_model_info
from yolov6.utils.metrics import ap_per_class, process_batch, ConfusionMatrix

"""
python tools/eval.py --task 'train'/'val'/'speed'
"""


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
        self.stride = 32
        self.pr_metric_result = (0.0, 0.0)

    @staticmethod
    def _to_jt_var(x):
        if isinstance(x, jt.Var):
            return x
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            x = x.numpy()
        return jt.array(x)

    def init_model(self, model, weights, task, use_cfg=False):
        if task != 'train':
            if not use_cfg:
                model = load_checkpoint(weights, map_location=self.device)
            else:
                model = load_checkpoint_2(model, weights, map_location=self.device)

            if hasattr(model, 'stride'):
                stride = model.stride
                self.stride = int(stride.max()) if hasattr(stride, "max") else int(stride)
            if jt.has_cuda:
                model(jt.zeros((1, 3, self.img_size, self.img_size), dtype=jt.float32))

            model = switch_to_deploy(model)
            LOGGER.info("Switch model to deploy modality.")
            LOGGER.info("Model Summary: {}".format(get_model_info(model, self.img_size)))
        else:
            if model is None:
                raise ValueError("`model` must be provided when task='train'.")
            if hasattr(model, 'stride'):
                stride = model.stride
                self.stride = int(stride.max()) if hasattr(stride, "max") else int(stride)

        if self.half and jt.has_cuda:
            if hasattr(model, "half"):
                try:
                    model.half()
                except Exception as e:
                    LOGGER.warning(f"Disable fp16 eval due to model precision cast issue: {e}")
                    self.half = False
            else:
                LOGGER.warning("Disable fp16 eval because current model path does not expose .half().")
                self.half = False

        if not self.half and hasattr(model, "float"):
            model.float()

        return model

    def init_data(self, dataloader, task):
        """Initialize dataloader for validation/speed or reuse training dataloader."""
        self.is_coco = self.data.get("is_coco", False)
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))

        if task == 'train':
            if dataloader is None:
                raise ValueError("`dataloader` must be provided when task='train'.")
            return dataloader

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

        rect = not self.not_infer_on_rect
        dataloader = create_dataloader(
            self.data[task if task in ('train', 'val', 'test') else 'val'],
            self.img_size,
            self.batch_size,
            self.stride,
            hyp=eval_hyp,
            check_labels=True,
            rect=rect,
            rank=-1,
            pad=pad,
            data_dict=self.data,
            task=task
        )[0]
        return dataloader

    def predict_model(self, model, dataloader, task):
        """Run model prediction for the whole dataloader."""
        self.speed_result = jt.zeros((4,), dtype=jt.float32)  # [num_images, pre, infer, nms]
        pred_results = []
        vis_outputs, vis_paths = [], []
        if self.do_pr_metric:
            self.pr_stats = []
            self.iouv = np.linspace(0.5, 0.95, 10).astype(np.float32)
            self.niou = self.iouv.shape[0]
            self.pr_seen = 0
            self.confusion_matrix = ConfusionMatrix(nc=model.nc) if self.plot_confusion_matrix else None
        pbar = tqdm(dataloader, desc=f"Inferencing model in {task} datasets.", ncols=NCOLS)

        for i, (imgs, targets, paths, shapes) in enumerate(pbar):
            t1 = time_sync()
            imgs = self._to_jt_var(imgs).float32() / 255.0
            targets = self._to_jt_var(targets).float32()
            self.speed_result[1] += time_sync() - t1

            t2 = time_sync()
            outputs = model(imgs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            self.speed_result[2] += time_sync() - t2

            t3 = time_sync()
            # Keep single-label NMS for stability in current jittor NMS implementation.
            outputs = non_max_suppression(outputs, self.conf_thres, self.iou_thres, multi_label=False, max_det=300)
            self.speed_result[3] += time_sync() - t3
            self.speed_result[0] += len(outputs)

            pred_results.extend(self.convert_to_coco_format(outputs, imgs, paths, shapes, self.ids))

            if i == 0:
                vis_num = min(len(outputs), 8)
                vis_outputs = outputs[:vis_num]
                vis_paths = paths[:vis_num]

            if self.do_pr_metric:
                for si, pred in enumerate(outputs):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl = int(labels.shape[0]) if len(labels.shape) else 0
                    tcls = labels[:, 0].numpy().tolist() if nl else []
                    self.pr_seen += 1

                    if len(pred) == 0:
                        if nl:
                            self.pr_stats.append(
                                (np.zeros((0, self.niou), dtype=bool), np.array([]), np.array([]), np.array(tcls))
                            )
                        continue
                    if len(pred.shape) != 2 or int(pred.shape[1]) < 6:
                        continue

                    predn = pred.clone()
                    self.scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

                    correct = np.zeros((predn.shape[0], self.niou), dtype=bool)
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5].clone())
                        tbox[:, [0, 2]] *= int(imgs[si].shape[2])
                        tbox[:, [1, 3]] *= int(imgs[si].shape[1])
                        self.scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                        labelsn = jt.concat((labels[:, 0:1], tbox), 1)
                        correct = process_batch(predn, labelsn, self.iouv)
                        if self.confusion_matrix is not None:
                            self.confusion_matrix.process_batch(predn, labelsn)

                    self.pr_stats.append((correct, pred[:, 4].numpy(), pred[:, 5].numpy(), np.array(tcls)))

        if self.do_pr_metric:
            if len(self.pr_stats):
                stats = [np.concatenate(x, 0) for x in zip(*self.pr_stats)]
            else:
                stats = []

            if len(stats) and stats[0].size and stats[0].any():
                metric_names = model.names if hasattr(model, 'names') else []
                if isinstance(metric_names, dict):
                    metric_names = [metric_names[k] for k in sorted(metric_names.keys())]
                p, r, ap, f1, ap_class = ap_per_class(*stats, plot=self.plot_curve, save_dir=self.save_dir,
                                                      names=metric_names)
                ap50, ap = ap[:, 0], ap.mean(1)
                AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
                mp = p[:, AP50_F1_max_idx].mean()
                mr = r[:, AP50_F1_max_idx].mean()
                map50 = ap50.mean()
                map_ = ap.mean()
                nc = int(model.nc) if hasattr(model, 'nc') else 0
                nt = np.bincount(stats[3].astype(np.int64), minlength=nc) if nc > 0 else np.array([])

                s = ('%-16s' + '%12s' * 7) % (
                    'Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
                LOGGER.info(s)
                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5
                LOGGER.info(pf % ('all', self.pr_seen, nt.sum() if nt.size else 0,
                                  mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map_))

                if self.verbose and nc > 1:
                    for i, c in enumerate(ap_class):
                        class_name = model.names[c] if isinstance(model.names, (list, tuple)) else str(c)
                        labels_count = nt[c] if c < len(nt) else 0
                        LOGGER.info(pf % (class_name, self.pr_seen, labels_count,
                                          p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                          f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

                if self.confusion_matrix is not None:
                    cm_names = metric_names if isinstance(metric_names, list) else []
                    self.confusion_matrix.plot(save_dir=self.save_dir, names=cm_names)
                self.pr_metric_result = (float(map50), float(map_))
            else:
                LOGGER.info("Calculate PR metric failed, might check dataset.")
                self.pr_metric_result = (0.0, 0.0)

        return pred_results, vis_outputs, vis_paths

    def eval_model(self, pred_results, model, dataloader, task):
        """Evaluate model by speed and metrics."""
        LOGGER.info('\nEvaluating speed.')
        self.eval_speed(task)

        if not self.do_coco_metric and self.do_pr_metric:
            return self.eval_pr(pred_results, model, dataloader, task)

        LOGGER.info('\nEvaluating mAP by pycocotools.')
        if self.do_coco_metric:
            return self.eval_coco(pred_results, model, dataloader, task)
        if self.do_pr_metric:
            return self.eval_pr(pred_results, model, dataloader, task)
        return (0.0, 0.0)

    def eval_coco(self, pred_results, model, dataloader, task):
        if task == 'speed' or len(pred_results) == 0:
            return (0.0, 0.0)

        if 'anno_path' in self.data:
            anno_json = self.data['anno_path']
        else:
            task_name = 'val' if task == 'train' else task
            dataset_root = os.path.dirname(os.path.dirname(self.data[task_name]))
            base_name = os.path.basename(self.data[task_name])
            anno_json = os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json')

        pred_json = os.path.join(self.save_dir, "predictions.json")
        LOGGER.info(f'Saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(pred_results, f)

        anno = COCO(anno_json)
        pred = anno.loadRes(pred_json)
        coco_eval = COCOeval(anno, pred, 'bbox')
        if self.is_coco:
            img_ids = []
            for x in dataloader.dataset.img_paths:
                stem = os.path.basename(x).split(".")[0]
                if stem.isnumeric():
                    img_ids.append(int(stem))
            if img_ids:
                coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map_, map50 = coco_eval.stats[:2]
        if hasattr(model, "float"):
            model.float()
        if task != 'train':
            LOGGER.info(f"Results saved to {self.save_dir}")
        return (float(map50), float(map_))

    def eval_pr(self, pred_results, model, dataloader, task):
        return self.pr_metric_result

    def eval_speed(self, task):
        n_samples = float(self.speed_result[0].item())
        if n_samples <= 0:
            return
        pre_time = 1000.0 * float(self.speed_result[1].item()) / n_samples
        inf_time = 1000.0 * float(self.speed_result[2].item()) / n_samples
        nms_time = 1000.0 * float(self.speed_result[3].item()) / n_samples
        for name, value in zip(["pre-process", "inference", "NMS"], [pre_time, inf_time, nms_time]):
            LOGGER.info(f"Average {name} time: {value:.2f} ms")
        if task != 'train':
            LOGGER.info(f"Results saved to {self.save_dir}")

    @staticmethod
    def box_convert(x):
        # Convert boxes from [x1, y1, x2, y2] to [x, y, w, h]
        y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """Rescale coords (xyxy) from img1_shape to img0_shape."""
        if ratio_pad is None:
            gain = [min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])]
            if self.scale_exact:
                gain = [img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]]
            pad = (
                (img1_shape[1] - img0_shape[1] * gain[0]) / 2,
                (img1_shape[0] - img0_shape[0] * gain[0]) / 2
            )
        else:
            gain = ratio_pad[0]
            if isinstance(gain, (tuple, list, np.ndarray)):
                if len(gain) == 1:
                    gain = [float(gain[0]), float(gain[0])]
                else:
                    gain = [float(gain[0]), float(gain[1])]
            else:
                gain = [float(gain), float(gain)]

            pad = ratio_pad[1]
            if isinstance(pad, (tuple, list, np.ndarray)):
                pad = (float(pad[0]), float(pad[1]))
            else:
                pad = (float(pad), float(pad))

        coords[:, [0, 2]] -= pad[0]
        if self.scale_exact:
            coords[:, [0, 2]] /= gain[1]
        else:
            coords[:, [0, 2]] /= gain[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, [1, 3]] /= gain[0]

        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        return coords

    def convert_to_coco_format(self, outputs, imgs, paths, shapes, ids):
        pred_results = []
        for i, pred in enumerate(outputs):
            if pred is None or len(pred) == 0:
                continue
            if len(pred.shape) != 2 or int(pred.shape[1]) < 6:
                continue

            path, shape = Path(paths[i]), shapes[i][0]
            self.scale_coords(imgs[i].shape[1:], pred[:, :4], shape, shapes[i][1])

            image_id = int(path.stem) if self.is_coco and path.stem.isnumeric() else path.stem
            pred_np = pred.numpy()
            boxes = pred_np[:, :4]
            scores = pred_np[:, 4]
            cls = pred_np[:, 5].astype(np.int64)

            bboxes = np.zeros_like(boxes)
            bboxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
            bboxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
            bboxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            bboxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            bboxes[:, :2] -= bboxes[:, 2:] / 2

            for ind in range(pred_np.shape[0]):
                cls_id = int(cls[ind])
                category_id = ids[cls_id] if cls_id < len(ids) else cls_id
                bbox = [round(float(x), 3) for x in bboxes[ind].tolist()]
                score = round(float(scores[ind]), 5)
                pred_results.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score
                })
        return pred_results

    @staticmethod
    def check_task(task):
        if task not in ['train', 'val', 'test', 'speed']:
            raise Exception("task argument error: only support 'train' / 'val' / 'test' / 'speed' task.")

    @staticmethod
    def check_thres(conf_thres, iou_thres, task):
        """Check whether confidence and IoU thresholds are reasonable for val/speed tasks."""
        if task != 'train':
            if task in ('val', 'test'):
                if conf_thres > 0.03:
                    LOGGER.warning(
                        f'The best conf_thresh when evaluating is usually <= 0.03, while you set {conf_thres}')
                if iou_thres != 0.65:
                    LOGGER.warning(
                        f'The best iou_thresh when evaluating is usually 0.65, while you set {iou_thres}')
            if task == 'speed' and conf_thres < 0.4:
                LOGGER.warning(
                    f'The best conf_thresh for speed test is usually >= 0.4, while you set {conf_thres}')

    @staticmethod
    def reload_dataset(data, task='val'):
        with open(data, errors='ignore') as yaml_file:
            data = yaml.safe_load(yaml_file)
        task_name = 'test' if task == 'test' else 'val'
        path = data.get(task_name, 'val')
        if not os.path.exists(path):
            raise Exception(f'Dataset not found: {path}')
        return data

    @staticmethod
    def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
        # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32,
             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
             60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x
