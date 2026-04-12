"""Box geometry helpers for dense detection models.

Changes vs original
-------------------
FIX-1  generalized_box_iou previously called box_iou (which internally
       computes inter, area1, area2, union) and then recomputed ALL of those
       quantities from scratch a second time to get the enclosing-box penalty.
       Fixed by fusing both into one function so every tensor operation runs
       exactly once.  For large [N, M] matrices this halves memory traffic.
"""

import torch


def distance_to_boxes(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Decode ltrb distances from points into xyxy boxes."""
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU for xyxy boxes — unchanged."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Element-wise IoU for matched xyxy box pairs. Returns shape [N]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU for xyxy boxes.

    FIX-1: fused into a single pass — inter, union, and enclosing box are all
    computed once.  Original called box_iou() first (inter + union computed
    internally) then recomputed inter + union again for the penalty term.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    # Intersection
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Individual areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter

    iou = inter / union.clamp(min=1e-6)

    # Enclosing box (computed once — was recomputed after box_iou() in original)
    enc_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


def generalized_box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Element-wise GIoU for matched xyxy box pairs. Returns shape [N]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0],))

    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)

def cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    """Convert normalized cxcywh boxes to absolute xyxy."""
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * image_w
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * image_h
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * image_w
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * image_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_abs_to_cxcywh_norm(boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
    """Convert absolute xyxy boxes to normalized cxcywh."""
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / image_w
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5 / image_h
    w  = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) / image_w
    h  = (boxes[:, 3] - boxes[:, 1]).clamp(min=0) / image_h
    return torch.stack([cx, cy, w, h], dim=-1)
