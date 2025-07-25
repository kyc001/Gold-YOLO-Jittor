"""
GOLD-YOLO Jittor版本 - 锚点生成器
从PyTorch版本迁移到Jittor框架
"""

import jittor as jt


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu', is_eval=False, mode='af'):
    '''Generate anchors from features.'''
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = jt.arange(end=w) + grid_cell_offset  # Jittor不需要指定device
            shift_y = jt.arange(end=h) + grid_cell_offset
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
            anchor_point = jt.stack([shift_x, shift_y], dim=-1).float()
            
            if mode == 'af':  # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                    jt.full((h * w, 1), stride, dtype=jt.float32))
            elif mode == 'ab':  # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
                stride_tensor.append(
                    jt.full((h * w, 1), stride, dtype=jt.float32).repeat(3, 1))
        
        anchor_points = jt.concat(anchor_points)
        stride_tensor = jt.concat(stride_tensor)
        return anchor_points, stride_tensor
    
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (jt.arange(end=w) + grid_cell_offset) * stride
            shift_y = (jt.arange(end=h) + grid_cell_offset) * stride
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
            
            anchor = jt.stack([
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ], dim=-1).clone().astype(feats[0].dtype)
            
            anchor_point = jt.stack([shift_x, shift_y], dim=-1).clone().astype(feats[0].dtype)

            if mode == 'af':  # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab':  # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3, 1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
            
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                jt.full([num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        
        anchors = jt.concat(anchors)
        anchor_points = jt.concat(anchor_points)
        stride_tensor = jt.concat(stride_tensor)
        return anchors, anchor_points, num_anchors_list, stride_tensor


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = jt.arange(end=w, dtype=dtype) + grid_cell_offset  # shift x
        sy = jt.arange(end=h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = jt.meshgrid(sy, sx)
        anchor_points.append(jt.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(jt.full((h * w, 1), stride, dtype=dtype))
    return jt.concat(anchor_points), jt.concat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return jt.concat((c_xy, wh), dim)  # xywh bbox
    return jt.concat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return jt.concat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)
