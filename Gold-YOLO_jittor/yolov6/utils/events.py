#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 事件和日志工具
从PyTorch版本迁移到Jittor框架
"""

import os
import yaml
import logging
import shutil


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def write_tblog(tblogger, epoch, results, losses):
    """Display mAP and loss information to log."""
    if tblogger is not None:
        tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
        tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)

        tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
        tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
        tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)

        tblogger.add_scalar("x/lr0", results[2], epoch + 1)
        tblogger.add_scalar("x/lr1", results[3], epoch + 1)
        tblogger.add_scalar("x/lr2", results[4], epoch + 1)


def write_tbimg(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    if tblogger is not None:
        if type == 'train':
            tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
        elif type == 'val':
            tblogger.add_image(f'val_pred', imgs, step + 1, dataformats='HWC')


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorstr(*input):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')"""
    *args, string = input if len(input) > 1 else ('blue', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
