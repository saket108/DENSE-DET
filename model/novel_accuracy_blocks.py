"""Accuracy-oriented lightweight blocks for DenseDet."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from model.dense_blocks import ConvBNAct


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def elongation_weighted_giou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    max_weight: float = 8.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sum of GIoU loss terms weighted by target-box elongation."""
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    gt_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps)
    gt_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=eps)
    elongation = torch.maximum(gt_w, gt_h) / torch.minimum(gt_w, gt_h).clamp(min=eps)
    weight = elongation.clamp(1.0, max_weight) / max_weight

    ix1 = torch.maximum(pred_boxes[:, 0], target_boxes[:, 0])
    iy1 = torch.maximum(pred_boxes[:, 1], target_boxes[:, 1])
    ix2 = torch.minimum(pred_boxes[:, 2], target_boxes[:, 2])
    iy2 = torch.minimum(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0.0) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    ).clamp(min=0.0)
    area_t = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0.0) * (
        target_boxes[:, 3] - target_boxes[:, 1]
    ).clamp(min=0.0)
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=eps)

    ex1 = torch.minimum(pred_boxes[:, 0], target_boxes[:, 0])
    ey1 = torch.minimum(pred_boxes[:, 1], target_boxes[:, 1])
    ex2 = torch.maximum(pred_boxes[:, 2], target_boxes[:, 2])
    ey2 = torch.maximum(pred_boxes[:, 3], target_boxes[:, 3])
    enc = (ex2 - ex1).clamp(min=0.0) * (ey2 - ey1).clamp(min=0.0)

    giou = iou - (enc - union) / enc.clamp(min=eps)
    return ((1.0 - giou) * weight).sum()


class SpectralChannelAttention(nn.Module):
    """Learn a per-channel spectral gate over neck features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.mix = ConvBNAct(channels, channels, kernel_size=1, act=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        energy = xf.abs().pow(2).mean(dim=(-2, -1))
        gate = self.gate_mlp(energy).unsqueeze(-1).unsqueeze(-1)
        x_gated = torch.fft.irfft2(xf * gate, s=(height, width), norm="ortho")
        return F.silu(self.norm(self.mix(x_gated)) + x, inplace=False)


def apply_uncertainty_weighted_varifocal(
    cls_logits: torch.Tensor,
    cls_targets: torch.Tensor,
    uncertainty: torch.Tensor | None,
    alpha: float = 0.75,
    gamma: float = 2.0,
    unc_floor: float = 0.05,
) -> torch.Tensor:
    """Varifocal loss reweighted by evidential uncertainty."""
    pred_sigmoid = cls_logits.sigmoid()
    weight = alpha * pred_sigmoid.pow(gamma) * (cls_targets <= 0).to(cls_logits.dtype)
    weight = weight + cls_targets * (cls_targets > 0).to(cls_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(cls_logits, cls_targets, reduction="none") * weight

    if uncertainty is not None:
        confidence = (1.0 - 2.0 * uncertainty).clamp(min=unc_floor)
        loss = loss * confidence.unsqueeze(-1)
    return loss


class ClassConditionalGroupNorm(nn.Module):
    """GroupNorm with affine offsets conditioned on class predictions."""

    def __init__(self, channels: int, num_classes: int, num_groups: int = 8) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(_group_count(channels, max_groups=num_groups), channels, affine=True)
        self.cls_proj = nn.Linear(num_classes, channels * 2, bias=False)
        nn.init.zeros_(self.cls_proj.weight)

    def forward(self, x: torch.Tensor, cls_logits: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        cls_prob = cls_logits.detach().sigmoid().mean(dim=(-2, -1))
        delta = self.cls_proj(cls_prob)
        delta_gamma = delta[:, :channels].view(batch_size, channels, 1, 1)
        delta_beta = delta[:, channels:].view(batch_size, channels, 1, 1)
        x_norm = self.gn(x)
        return (1.0 + delta_gamma) * x_norm + delta_beta


class CCGNConvBlock(nn.Module):
    """Depthwise-pointwise block using class-conditional GroupNorm."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = ClassConditionalGroupNorm(channels, num_classes)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cls_logits: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x)), cls_logits))


__all__ = [
    "CCGNConvBlock",
    "ClassConditionalGroupNorm",
    "SpectralChannelAttention",
    "apply_uncertainty_weighted_varifocal",
    "elongation_weighted_giou_loss",
]
