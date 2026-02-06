# Model validation metrics
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py

from pathlib import Path
import warnings

import numpy as np
import jittor as jt

from . import general

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, jt.Var):
        return x.numpy()
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """Compute AP given prediction correctness, confidence, and class ids."""
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]

    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap = np.zeros((nc, tp.shape[1]))
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + 1e-16)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))

    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """Compute average precision from recall and precision curves."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    if plt is None or len(py) == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')
    else:
        ax.plot(px, py, linewidth=1, color='grey')

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close(fig)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    if plt is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close(fig)


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix.
    detections: [N, 6] -> x1, y1, x2, y2, conf, class
    labels: [M, 5] -> class, x1, y1, x2, y2
    iouv: [K]
    """
    detections = _to_numpy(detections).astype(np.float32)
    labels = _to_numpy(labels).astype(np.float32)
    iouv = _to_numpy(iouv).astype(np.float32).reshape(-1)

    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return correct

    iou = general.box_iou(jt.array(labels[:, 1:]), jt.array(detections[:, :4])).numpy()
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i, iou_thr in enumerate(iouv):
        x = np.where((iou >= iou_thr) & correct_class)
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        detections = _to_numpy(detections).astype(np.float32)
        labels = _to_numpy(labels).astype(np.float32)

        if detections.shape[0] == 0:
            if labels.shape[0] > 0:
                for gc in labels[:, 0].astype(int):
                    self.matrix[self.nc, gc] += 1
            return

        detections = detections[detections[:, 4] > self.conf]
        if detections.shape[0] == 0:
            if labels.shape[0] > 0:
                for gc in labels[:, 0].astype(int):
                    self.matrix[self.nc, gc] += 1
            return

        gt_classes = labels[:, 0].astype(int)
        detection_classes = detections[:, 5].astype(int)
        iou = general.box_iou(jt.array(labels[:, 1:]), jt.array(detections[:, :4])).numpy()

        x = np.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        if n:
            m0, m1, _ = matches.transpose().astype(int)
        else:
            m0, m1 = np.array([], dtype=int), np.array([], dtype=int)

        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and j.sum() == 1:
                self.matrix[detection_classes[m1[j]][0], gc] += 1
            else:
                self.matrix[self.nc, gc] += 1

        if n:
            for i, dc in enumerate(detection_classes):
                if not np.any(m1 == i):
                    self.matrix[dc, self.nc] += 1

    def tp_fp(self):
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        return tp[:-1], fp[:-1]

    def plot(self, normalize=True, save_dir='', names=()):
        if plt is None:
            return
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)
            array[array < 0.005] = np.nan

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            nc, nn = self.nc, len(names)
            sn.set(font_scale=1.0 if nc < 50 else 0.8)
            labels = (0 < nn < 99) and (nn == nc)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sn.heatmap(array,
                           annot=nc < 30,
                           annot_kws={"size": 8},
                           cmap='Blues',
                           fmt='.2f',
                           square=True,
                           vmin=0.0,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close(fig)
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

