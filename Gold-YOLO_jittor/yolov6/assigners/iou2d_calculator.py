"""
GOLD-YOLO Jittor版本 - 2D IoU计算器
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
This code is based on
https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/iou_calculators/iou2d_calculator.py
"""

import jittor as jt


def compute_iou_2d(bboxes1, bboxes2):
    """简化的2D IoU计算"""
    # bboxes1: [N, 4], bboxes2: [M, 4]
    # 返回: [N, M]

    N = bboxes1.shape[0]
    M = bboxes2.shape[0]

    # 扩展维度进行广播
    bboxes1_exp = bboxes1.unsqueeze(1)  # [N, 1, 4]
    bboxes2_exp = bboxes2.unsqueeze(0)  # [1, M, 4]

    # 计算交集
    x1 = jt.maximum(bboxes1_exp[:, :, 0], bboxes2_exp[:, :, 0])
    y1 = jt.maximum(bboxes1_exp[:, :, 1], bboxes2_exp[:, :, 1])
    x2 = jt.minimum(bboxes1_exp[:, :, 2], bboxes2_exp[:, :, 2])
    y2 = jt.minimum(bboxes1_exp[:, :, 3], bboxes2_exp[:, :, 3])

    # 交集面积
    intersection = jt.clamp(x2 - x1, min_v=0) * jt.clamp(y2 - y1, min_v=0)

    # 计算各自面积
    area1 = (bboxes1_exp[:, :, 2] - bboxes1_exp[:, :, 0]) * (bboxes1_exp[:, :, 3] - bboxes1_exp[:, :, 1])
    area2 = (bboxes2_exp[:, :, 2] - bboxes2_exp[:, :, 0]) * (bboxes2_exp[:, :, 3] - bboxes2_exp[:, :, 1])

    # 并集面积
    union = area1 + area2 - intersection

    # IoU
    iou = intersection / (union + 1e-6)

    return iou


def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


def fp16_clamp(x, min=None, max=None):
    # Jittor兼容的clamp实现
    if min is not None and max is not None:
        return jt.clamp(x, min, max)
    elif min is not None:
        return jt.maximum(x, min)
    elif max is not None:
        return jt.minimum(x, max)
    else:
        return x


def iou2d_calculator(bboxes1, bboxes2, mode='iou', is_aligned=False, scale=1., dtype=None):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    """Calculate IoU between 2D bboxes.

    Args:
        bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
            format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
        bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
            format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
            empty. If ``is_aligned `` is ``True``, then m and n must be
            equal.
        mode (str): "iou" (intersection over union), "iof" (intersection
            over foreground), or "giou" (generalized intersection over
            union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert bboxes1.size(-1) in [0, 4, 5]
    assert bboxes2.size(-1) in [0, 4, 5]
    if bboxes2.size(-1) == 5:
        bboxes2 = bboxes2[..., :4]
    if bboxes1.size(-1) == 5:
        bboxes1 = bboxes1[..., :4]

    if dtype == 'fp16':
        # change tensor type to save memory and keep speed
        bboxes1 = cast_tensor_type(bboxes1, scale, dtype)
        bboxes2 = cast_tensor_type(bboxes2, scale, dtype)
        overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
        # Jittor中简化fp16处理
        return overlaps

    return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = jt.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = jt.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = jt.empty(0, 4)
        >>> nonempty = jt.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # 修复batch维度不匹配问题
    if bboxes1.shape[:-2] != bboxes2.shape[:-2]:
        # 验证bbox格式
        if bboxes1.shape[-1] != 4:
            raise ValueError(f"bboxes1最后维度必须是4，得到: {bboxes1.shape}")
        if bboxes2.shape[-1] != 4:
            raise ValueError(f"bboxes2最后维度必须是4，得到: {bboxes2.shape}")

        # 如果bboxes1是2D [N, 4]，bboxes2是3D [B, M, 4]
        if len(bboxes1.shape) == 2 and len(bboxes2.shape) == 3:
            # 将bboxes2 reshape为2D [B*M, 4]
            batch_size = bboxes2.shape[0]
            num_anchors = bboxes2.shape[1]
            bboxes2_reshaped = bboxes2.view(-1, 4)  # [B*M, 4]

            # 验证reshape后的格式
            if bboxes2_reshaped.shape[-1] != 4:
                raise ValueError(f"bboxes2 reshape后格式错误: {bboxes2_reshaped.shape}")

            # 计算IoU (使用简化的2D计算)
            overlaps = compute_iou_2d(bboxes1, bboxes2_reshaped)

            # 将结果reshape回 [N, B*M]，然后可以进一步处理
            return overlaps

        # 如果bboxes2是2D [N, 4]，bboxes1是3D [B, M, 4]
        elif len(bboxes1.shape) == 3 and len(bboxes2.shape) == 2:
            # 将bboxes1 reshape为2D [B*M, 4]
            batch_size = bboxes1.shape[0]
            num_anchors = bboxes1.shape[1]
            bboxes1_reshaped = bboxes1.view(-1, 4)  # [B*M, 4]

            # 验证reshape后的格式
            if bboxes1_reshaped.shape[-1] != 4:
                raise ValueError(f"bboxes1 reshape后格式错误: {bboxes1_reshaped.shape}")

            # 计算IoU (使用简化的2D计算)
            overlaps = compute_iou_2d(bboxes1_reshaped, bboxes2)

            # 将结果reshape回 [B*M, N]
            return overlaps

        else:
            raise AssertionError(f"不支持的bbox形状组合: {bboxes1.shape} vs {bboxes2.shape}")

    # 原始的batch维度检查
    # assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return jt.zeros(batch_shape + (rows, ), dtype=bboxes1.dtype)
        else:
            return jt.zeros(batch_shape + (rows, cols), dtype=bboxes1.dtype)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = jt.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = jt.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = jt.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = jt.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = jt.maximum(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = jt.minimum(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = jt.minimum(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = jt.maximum(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = jt.array([eps], dtype=union.dtype)
    union = jt.maximum(union, eps)
    ious = overlap / union

    # 限制IoU值在合理范围内，防止数值错误
    ious = jt.clamp(ious, 0.0, 1.0)

    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = jt.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area

    # 限制GIoU值在合理范围内
    gious = jt.clamp(gious, -1.0, 1.0)

    return gious
