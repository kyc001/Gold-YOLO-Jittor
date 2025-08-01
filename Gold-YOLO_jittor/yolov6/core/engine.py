# 2023.09.18-Changed for engine implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com
# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 训练引擎模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
import time
from copy import deepcopy
import os.path as osp

from tqdm import tqdm

import cv2
import numpy as np
import math
import jittor as jt
from jittor import nn

import tools.eval as eval
from yolov6.data.data_load import create_dataloader
from yolov6.models.yolo import build_model

# 百分百对齐PyTorch版本 - 完整的损失函数导入
from yolov6.models.losses.loss import ComputeLoss as ComputeLoss
from yolov6.models.losses.loss_fuseab import ComputeLoss as ComputeLoss_ab
from yolov6.models.losses.loss_distill import ComputeLoss as ComputeLoss_distill
from yolov6.models.losses.loss_distill_ns import ComputeLoss as ComputeLoss_distill_ns

from yolov6.utils.events import LOGGER, NCOLS, load_yaml, write_tblog, write_tbimg
from yolov6.utils.ema import ModelEMA, de_parallel
from yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from yolov6.solver.build import build_optimizer, build_lr_scheduler
from yolov6.utils.RepOptimizer import extract_scales, RepVGGOptimizer
from yolov6.utils.nms import xywh2xyxy


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device
        
        if args.resume:
            self.ckpt = jt.load(args.resume)  # Jittor使用jt.load
        
        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir
        # get data loader
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']
        self.train_loader, self.val_loader = self.get_data_loader(args, cfg, self.data_dict)
        # get model and optimizer
        self.distill_ns = True if self.args.distill and self.cfg.model.type in ['YOLOv6n', 'YOLOv6s', 'GoldYOLO-n', 'GoldYOLO-s'] else False
        model = self.get_model(args, cfg, self.num_classes, device)
        if self.args.distill:
            if self.args.fuse_ab:
                LOGGER.error('ERROR in: Distill models should turn off the fuse_ab.\n')
                exit()
            self.teacher_model = self.get_teacher_model(args, cfg, self.num_classes, device)
        if self.args.quant:
            self.quant_setup(model, cfg, device)
        if cfg.training_mode == 'repopt':
            scales = self.load_scale_from_pretrained_models(cfg, device)
            reinit = False if cfg.model.pretrained is not None else True
            self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=reinit)
        else:
            self.optimizer = self.get_optimizer(args, cfg, model)
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
        self.ema = ModelEMA(model) if self.main_process else None
        # tensorboard - Jittor可以使用TensorBoard或其他日志工具
        try:
            from jittor.utils.tensorboard import SummaryWriter
            self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
        except ImportError:
            # 如果没有tensorboard，使用简单的日志记录
            self.tblogger = None
            if self.main_process:
                LOGGER.warning("TensorBoard not available, using simple logging")
        
        self.start_epoch = 0
        # resume
        if hasattr(self, "ckpt"):
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict)  # load，Jittor不需要strict参数
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
                self.ema.updates = self.ckpt['updates']
        self.model = self.parallel_model(args, model, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']
        
        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.vis_imgs_list = []
        self.write_trainbatch_tb = args.write_trainbatch_tb
        # set color for classnames
        self.color = [tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)]
        
        self.loss_num = 3
        self.loss_info = ['Epoch', 'iou_loss', 'dfl_loss', 'cls_loss']
        if self.args.distill:
            self.loss_num += 1
            self.loss_info += ['cwd_loss']
    
    # Training Process
    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop(self.epoch)
            self.strip_model()
        
        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()
    
    # Training loop for each epoch
    def train_in_loop(self, epoch_num):
        try:
            self.prepare_for_steps()
            for self.step, self.batch_data in self.pbar:
                try:
                    self.train_in_steps(epoch_num, self.step)
                except Exception as e:
                    LOGGER.error(f'ERROR in training steps: {e}')
                self.print_details()
        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise
        try:
            self.eval_and_save()
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise
    
    # Training loop for batchdata
    def train_in_steps(self, epoch_num, step_num):
        images, targets = self.prepro_data(self.batch_data, self.device)
        # plot train_batch and save to tensorboard once an epoch
        if self.write_trainbatch_tb and self.main_process and self.step == 0:
            self.plot_train_batch(images, targets)
            if self.tblogger:
                write_tbimg(self.tblogger, self.vis_train_batch, self.step + self.max_stepnum * self.epoch, type='train')
        
        # forward - Jittor不需要amp.autocast
        preds, s_featmaps = self.model(images)
        if self.args.distill:
            # Jittor不需要torch.no_grad()，使用jt.no_grad()
            with jt.no_grad():
                t_preds, t_featmaps = self.teacher_model(images)
            temperature = self.args.temperature
            total_loss, loss_items = self.compute_loss_distill(preds, t_preds, s_featmaps, t_featmaps, targets, \
                                                               epoch_num, self.max_epoch, temperature, step_num)
        
        elif self.args.fuse_ab:
            total_loss, loss_items = self.compute_loss((preds[0], preds[3], preds[4]), targets, epoch_num,
                                                       step_num)  # YOLOv6_af
            total_loss_ab, loss_items_ab = self.compute_loss_ab(preds[:3], targets, epoch_num,
                                                                step_num)  # YOLOv6_ab
            total_loss += total_loss_ab
            loss_items += loss_items_ab
        else:
            total_loss, loss_items = self.compute_loss(preds, targets, epoch_num, step_num)  # YOLOv6_af
        if self.rank != -1:
            total_loss *= self.world_size
        
        # backward - Jittor自动处理梯度缩放
        self.optimizer.backward(total_loss)  # Jittor的反向传播方式
        self.loss_items = loss_items
        self.update_optimizer()
    
    def eval_and_save(self):
        remaining_epochs = self.max_epoch - self.epoch
        eval_interval = self.args.eval_interval if remaining_epochs > self.args.heavy_eval_range else 3
        is_val_epoch = (not self.args.eval_final_only or (remaining_epochs == 1)) and (self.epoch % eval_interval == 0)
        if self.main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride'])  # update attributes for ema model
            if is_val_epoch:
                self.eval_model()
                self.ap = self.evaluate_results[1]
                self.best_ap = max(self.ap, self.best_ap)
            # save ckpt
            ckpt = {
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'ema': deepcopy(self.ema.ema).half(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
            }
            
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
            if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
                save_checkpoint(ckpt, False, save_ckpt_dir, model_name=f'{self.epoch}_ckpt')
            
            # default save best ap ckpt in stop strong aug epochs
            if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
                if self.best_stop_strong_aug_ap < self.ap:
                    self.best_stop_strong_aug_ap = max(self.ap, self.best_stop_strong_aug_ap)
                    save_checkpoint(ckpt, False, save_ckpt_dir, model_name='best_stop_aug_ckpt')
            
            del ckpt
            # log for learning rate
            lr = [x['lr'] for x in self.optimizer.param_groups]
            self.evaluate_results = list(self.evaluate_results) + lr

            # log for tensorboard
            if self.tblogger:
                write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)
                # save validation predictions to tensorboard
                write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type='val')

    def eval_model(self):
        if not hasattr(self.cfg, "eval_params"):
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                       batch_size=self.batch_size // self.world_size * 2,
                                                       img_size=self.img_size,
                                                       model=self.ema.ema if self.args.calib is False else self.model,
                                                       conf_thres=0.03,
                                                       dataloader=self.val_loader,
                                                       save_dir=self.save_dir,
                                                       task='train')
        else:
            def get_cfg_value(cfg_dict, value_str, default_value):
                if value_str in cfg_dict:
                    if isinstance(cfg_dict[value_str], list):
                        return cfg_dict[value_str][0] if cfg_dict[value_str][0] is not None else default_value
                    else:
                        return cfg_dict[value_str] if cfg_dict[value_str] is not None else default_value
                else:
                    return default_value

            eval_img_size = get_cfg_value(self.cfg.eval_params, "img_size", self.img_size)
            results, vis_outputs, vis_paths = eval.run(self.data_dict,
                                                       batch_size=get_cfg_value(self.cfg.eval_params, "batch_size",
                                                                                self.batch_size // self.world_size * 2),
                                                       img_size=eval_img_size,
                                                       model=self.ema.ema if self.args.calib is False else self.model,
                                                       conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres",
                                                                                0.03),
                                                       dataloader=self.val_loader,
                                                       save_dir=self.save_dir,
                                                       task='train',
                                                       test_load_size=get_cfg_value(self.cfg.eval_params,
                                                                                    "test_load_size", eval_img_size),
                                                       letterbox_return_int=get_cfg_value(self.cfg.eval_params,
                                                                                          "letterbox_return_int",
                                                                                          False),
                                                       force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad",
                                                                                  False),
                                                       not_infer_on_rect=get_cfg_value(self.cfg.eval_params,
                                                                                       "not_infer_on_rect", False),
                                                       scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact",
                                                                                 False),
                                                       verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                                                       do_coco_metric=get_cfg_value(self.cfg.eval_params,
                                                                                    "do_coco_metric", True),
                                                       do_pr_metric=get_cfg_value(self.cfg.eval_params, "do_pr_metric",
                                                                                  False),
                                                       plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve",
                                                                                False),
                                                       plot_confusion_matrix=get_cfg_value(self.cfg.eval_params,
                                                                                           "plot_confusion_matrix",
                                                                                           False),
                                                       )

        LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]
        # plot validation predictions
        self.plot_val_pred(vis_outputs, vis_paths)

    def train_before_loop(self):
        LOGGER.info('Training start...')
        self.start_time = time.time()
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum),
                                  1000) if self.args.quant is False else 0
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        # Jittor不需要GradScaler，自动处理梯度缩放

        self.best_ap, self.ap = 0.0, 0.0
        self.best_stop_strong_aug_ap = 0.0
        self.evaluate_results = (0, 0)  # AP50, AP50_95

        self.compute_loss = ComputeLoss(num_classes=self.data_dict['nc'],
                                        ori_img_size=self.img_size,
                                        warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                        use_dfl=self.cfg.model.head.use_dfl,
                                        reg_max=self.cfg.model.head.reg_max,
                                        iou_type=self.cfg.model.head.iou_type,
                                        fpn_strides=self.cfg.model.head.strides)

        if self.args.fuse_ab:
            self.compute_loss_ab = ComputeLoss_ab(num_classes=self.data_dict['nc'],
                                                  ori_img_size=self.img_size,
                                                  warmup_epoch=0,
                                                  use_dfl=False,
                                                  reg_max=0,
                                                  iou_type=self.cfg.model.head.iou_type,
                                                  fpn_strides=self.cfg.model.head.strides)
        if self.args.distill:
            if self.cfg.model.type in ['YOLOv6n', 'YOLOv6s', 'GoldYOLO-n', 'GoldYOLO-s']:
                self.compute_loss_distill = ComputeLoss_distill_ns(num_classes=self.data_dict['nc'],
                                                                   ori_img_size=self.img_size,
                                                                   warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                                                   use_dfl=self.cfg.model.head.use_dfl,
                                                                   reg_max=self.cfg.model.head.reg_max,
                                                                   iou_type=self.cfg.model.head.iou_type,
                                                                   fpn_strides=self.cfg.model.head.strides,
                                                                   distill_weight=self.args.distill_weight)
            else:
                self.compute_loss_distill = ComputeLoss_distill(num_classes=self.data_dict['nc'],
                                                                ori_img_size=self.img_size,
                                                                warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                                                                use_dfl=self.cfg.model.head.use_dfl,
                                                                reg_max=self.cfg.model.head.reg_max,
                                                                iou_type=self.cfg.model.head.iou_type,
                                                                fpn_strides=self.cfg.model.head.strides,
                                                                distill_weight=self.args.distill_weight)

    def prepare_for_steps(self):
        if self.epoch > self.start_epoch:
            self.scheduler.step()
        # stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            self.train_loader, self.val_loader = self.get_data_loader(self.args, self.cfg, self.data_dict)
        self.model.train()
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = jt.zeros(self.loss_num)  # 使用jt.zeros替代torch.zeros
        self.optimizer.zero_grad()

        LOGGER.info(('\n' + '%10s' * (self.loss_num + 1)) % (*self.loss_info,))
        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # Print loss after each steps
    def print_details(self):
        if self.main_process:
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
            self.pbar.set_description(('%10s' + '%10.4g' * self.loss_num) % (f'{self.epoch}/{self.max_epoch - 1}', \
                                                                             *(self.mean_loss)))

    def strip_model(self):
        if self.main_process:
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            strip_optimizer(save_ckpt_dir, self.epoch)  # strip optimizers for saved pt model

    # Empty cache if training finished
    def train_after_loop(self):
        # Jittor自动管理内存，不需要手动清空缓存
        jt.gc()  # 可选的垃圾回收

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                        [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum],
                                                  [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
        if curr_step - self.last_opt_step >= self.accumulate:
            # Jittor不需要scaler，直接step
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        train_path, val_path = data_dict['train'], data_dict['val']
        # check data
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, data_dict=data_dict, task='train')[0]
        # create val dataloader
        val_loader = None
        if args.rank in [-1, 0]:
            val_loader = create_dataloader(val_path, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                           hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                           workers=args.workers, check_images=args.check_images,
                                           check_labels=args.check_labels, data_dict=data_dict, task='val')[0]

        return train_loader, val_loader

    @staticmethod
    def prepro_data(batch_data, device):
        # Jittor自动处理设备转换，不需要.to(device)
        # 修复：确保图像数据类型正确，避免int进入卷积层
        images = batch_data[0]
        if images.dtype != 'float32':
            images = images.float32()  # 强制转换为float32
        images = images / 255.0  # 归一化

        targets = batch_data[1]
        if targets.dtype != 'float32':
            targets = targets.float32()  # 确保targets也是float32

        return images, targets
