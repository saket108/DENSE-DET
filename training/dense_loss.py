"""Loss for the lightweight PRISM-based dense detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils.box_ops import (
    box_iou,
    box_iou_pairwise,
    cxcywh_norm_to_xyxy_abs,
    distance_to_boxes,
    generalized_box_iou,
    generalized_box_iou_pairwise,
)
from utils.points import build_points


def varifocal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.55,
    gamma: float = 2.0,
) -> torch.Tensor:
    pred_sigmoid = logits.sigmoid()
    weight = alpha * pred_sigmoid.pow(gamma) * (targets <= 0).to(logits.dtype)
    weight = weight + targets * (targets > 0).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return loss * weight


@dataclass
class LossOutput:
    total: torch.Tensor
    cls: torch.Tensor
    box: torch.Tensor
    qual: torch.Tensor
    positives: int


class DenseDetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        strides: tuple[int, ...] = (4, 8, 16, 32),
        size_ranges: tuple[tuple[float, float], ...] | None = None,
        assigner: str = "atss",
        center_radius: float = 1.5,
        topk_candidates: int = 0,
        atss_topk: int = 9,
        atss_anchor_scale: float = 4.0,
        quality_loss_weight: float = 1.0,
        vfl_alpha: float = 0.55,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = tuple(int(value) for value in strides)
        self.assigner = str(assigner)
        self.center_radius = float(center_radius)
        self.topk_candidates = int(topk_candidates)
        self.atss_topk = int(atss_topk)
        self.atss_anchor_scale = float(atss_anchor_scale)
        self.quality_loss_weight = float(quality_loss_weight)
        self.vfl_alpha = float(vfl_alpha)

        if self.assigner not in {"fcos", "atss"}:
            raise ValueError("assigner must be either 'fcos' or 'atss'.")

        if size_ranges is None:
            size_ranges = (
                (0.0, 64.0),
                (64.0, 128.0),
                (128.0, 256.0),
                (256.0, 1e8),
            )
        self.size_ranges = tuple(size_ranges)
        if len(self.size_ranges) != len(self.strides):
            raise ValueError("size_ranges must match the number of strides.")

    def _size_ranges_for_strides(self, strides: tuple[int, ...]) -> tuple[tuple[float, float], ...]:
        active = []
        for stride in strides:
            try:
                index = self.strides.index(int(stride))
            except ValueError as exc:
                raise ValueError(
                    f"Stride {stride} is not present in the configured loss strides {self.strides}."
                ) from exc
            active.append(self.size_ranges[index])
        return tuple(active)

    def forward(
        self,
        outputs: dict[str, list[torch.Tensor] | tuple[int, ...]],
        targets: list[dict[str, torch.Tensor]],
    ) -> LossOutput:
        cls_levels = outputs["cls"]  # type: ignore[index]
        box_levels = outputs["box"]  # type: ignore[index]
        quality_levels = outputs.get("quality")  # type: ignore[assignment]
        image_h, image_w = outputs["image_size"]  # type: ignore[index]
        strides = tuple(int(value) for value in outputs.get("strides", self.strides))  # type: ignore[arg-type]
        size_ranges = self._size_ranges_for_strides(strides)

        batch_size = cls_levels[0].shape[0]
        device = cls_levels[0].device
        loss_dtype = torch.float32

        flat_cls = []
        flat_box = []
        flat_qual = []
        flat_points = []
        flat_ranges = []
        flat_strides = []

        level_iter = zip(
            cls_levels,
            box_levels,
            strides,
            size_ranges,
            quality_levels if quality_levels is not None else [None] * len(cls_levels),
        )
        for cls_map, box_map, stride, size_range, qual_map in level_iter:
            cls_map = cls_map.float()
            box_map = box_map.float()
            _, _, feat_h, feat_w = cls_map.shape
            points = build_points(feat_h, feat_w, stride, device, loss_dtype)

            flat_cls.append(cls_map.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes))
            flat_box.append(box_map.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) * stride)
            flat_points.append(points)
            flat_ranges.append(
                torch.tensor(size_range, device=device, dtype=loss_dtype).expand(points.shape[0], 2)
            )
            flat_strides.append(torch.full((points.shape[0],), stride, device=device, dtype=loss_dtype))
            if qual_map is not None:
                flat_qual.append(qual_map.float().reshape(batch_size, -1, 1))

        pred_cls = torch.cat(flat_cls, dim=1)
        pred_box = torch.cat(flat_box, dim=1)
        pred_qual = torch.cat(flat_qual, dim=1) if flat_qual else None
        points = torch.cat(flat_points, dim=0)
        size_ranges_t = torch.cat(flat_ranges, dim=0)
        strides_t = torch.cat(flat_strides, dim=0)

        cls_loss = pred_cls.new_tensor(0.0)
        box_loss = pred_cls.new_tensor(0.0)
        qual_loss = pred_cls.new_tensor(0.0)
        total_pos = 0

        # Process each batch sample: assign targets and compute losses
        # Note: Future optimization opportunity - vectorize _assign_targets to process batch dimension
        # by stacking all gt_boxes/labels with batch indices instead of per-sample loop
        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes"]
            if gt_boxes.numel() > 0:
                gt_boxes = cxcywh_norm_to_xyxy_abs(
                    gt_boxes.to(device=device, dtype=loss_dtype),
                    image_h=image_h,
                    image_w=image_w,
                )
            else:
                gt_boxes = torch.zeros((0, 4), device=device, dtype=loss_dtype)
            gt_labels = target["labels"].to(device=device)

            assigned = self._assign_targets(points, size_ranges_t, strides_t, gt_boxes, gt_labels)
            cls_targets = pred_cls.new_zeros((points.shape[0], self.num_classes))
            pos_mask = assigned["labels"] >= 0

            if pos_mask.any():
                pred_boxes_pos = distance_to_boxes(points[pos_mask], pred_box[batch_index][pos_mask])
                target_boxes = assigned["boxes"][pos_mask]
                iou_values = box_iou_pairwise(pred_boxes_pos, target_boxes).detach().clamp_(0.0, 1.0)
                cls_targets[pos_mask, assigned["labels"][pos_mask]] = iou_values

                giou_values = generalized_box_iou_pairwise(pred_boxes_pos, target_boxes)
                box_loss = box_loss + (1.0 - giou_values).sum()
                total_pos += int(pos_mask.sum().item())

                if pred_qual is not None:
                    qual_loss = qual_loss + F.binary_cross_entropy_with_logits(
                        pred_qual[batch_index][pos_mask].squeeze(-1),
                        iou_values,
                        reduction="sum",
                    )

            cls_loss = cls_loss + varifocal_loss(
                pred_cls[batch_index],
                cls_targets,
                alpha=self.vfl_alpha,
            ).sum()

        normalizer = max(total_pos, batch_size)
        cls_loss = cls_loss / normalizer
        box_loss = box_loss / normalizer
        qual_loss = qual_loss / normalizer
        total = cls_loss + box_loss + self.quality_loss_weight * qual_loss
        return LossOutput(total=total, cls=cls_loss, box=box_loss, qual=qual_loss, positives=total_pos)

    def _assign_targets(
        self,
        points: torch.Tensor,
        size_ranges: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.assigner == "atss":
            return self._assign_targets_atss(points, strides, gt_boxes, gt_labels)
        return self._assign_targets_fcos(points, size_ranges, strides, gt_boxes, gt_labels)

    def _assign_targets_fcos(
        self,
        points: torch.Tensor,
        size_ranges: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = points.device
        num_points = points.shape[0]
        labels = torch.full((num_points,), -1, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((num_points, 4), device=device, dtype=points.dtype)
        if gt_boxes.numel() == 0:
            return {"labels": labels, "boxes": assigned_boxes}

        x = points[:, 0][:, None]
        y = points[:, 1][:, None]

        left = x - gt_boxes[:, 0]
        top = y - gt_boxes[:, 1]
        right = gt_boxes[:, 2] - x
        bottom = gt_boxes[:, 3] - y
        reg_targets = torch.stack([left, top, right, bottom], dim=-1)

        inside_box = reg_targets.min(dim=-1).values > 0
        max_reg = reg_targets.max(dim=-1).values
        inside_range = (max_reg >= size_ranges[:, 0:1]) & (max_reg <= size_ranges[:, 1:2])

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        radius = strides[:, None] * self.center_radius
        center_x1 = torch.maximum(gt_boxes[:, 0], gt_centers[:, 0] - radius)
        center_y1 = torch.maximum(gt_boxes[:, 1], gt_centers[:, 1] - radius)
        center_x2 = torch.minimum(gt_boxes[:, 2], gt_centers[:, 0] + radius)
        center_y2 = torch.minimum(gt_boxes[:, 3], gt_centers[:, 1] + radius)
        inside_center = (x >= center_x1) & (x <= center_x2) & (y >= center_y1) & (y <= center_y2)

        matches = inside_box & inside_range & inside_center

        if self.topk_candidates > 0:
            candidate_mask = inside_box & inside_range
            gt_widths = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
            gt_heights = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)
            center_dist = (
                ((x - gt_centers[:, 0]) / gt_widths) ** 2
                + ((y - gt_centers[:, 1]) / gt_heights) ** 2
            )
            center_dist = center_dist.masked_fill(~candidate_mask, float("inf"))

            topk = min(self.topk_candidates, num_points)
            _, topk_idx = center_dist.topk(topk, dim=0, largest=False)
            topk_matches = torch.zeros_like(candidate_mask)
            gt_indices = torch.arange(gt_boxes.shape[0], device=device).unsqueeze(0).expand(topk, -1)
            valid = torch.isfinite(center_dist[topk_idx, gt_indices])
            topk_matches[topk_idx[valid], gt_indices[valid]] = True
            matches = matches | topk_matches

        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))[None, :]
        areas = areas.expand(num_points, -1).clone()
        areas[~matches] = float("inf")

        min_areas, matched_gt = areas.min(dim=1)
        pos_mask = torch.isfinite(min_areas)
        if not pos_mask.any():
            return {"labels": labels, "boxes": assigned_boxes}

        labels[pos_mask] = gt_labels[matched_gt[pos_mask]]
        assigned_boxes[pos_mask] = gt_boxes[matched_gt[pos_mask]].to(dtype=assigned_boxes.dtype)
        return {"labels": labels, "boxes": assigned_boxes}

    def _assign_targets_atss(
        self,
        points: torch.Tensor,
        strides: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        device = points.device
        num_points = points.shape[0]
        labels = torch.full((num_points,), -1, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((num_points, 4), device=device, dtype=points.dtype)
        if gt_boxes.numel() == 0:
            return {"labels": labels, "boxes": assigned_boxes}

        num_gt = gt_boxes.shape[0]
        x = points[:, 0][:, None]
        y = points[:, 1][:, None]
        reg = torch.stack(
            [x - gt_boxes[:, 0], y - gt_boxes[:, 1], gt_boxes[:, 2] - x, gt_boxes[:, 3] - y],
            dim=-1,
        )
        inside_box = reg.min(dim=-1).values > 0

        half_size = strides * (self.atss_anchor_scale * 0.5)
        anchors = torch.stack(
            [
                points[:, 0] - half_size,
                points[:, 1] - half_size,
                points[:, 0] + half_size,
                points[:, 1] + half_size,
            ],
            dim=-1,
        )
        anchor_ious = box_iou(anchors, gt_boxes)

        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        center_dist = (
            (points[:, None, 0] - gt_centers[:, 0]) ** 2
            + (points[:, None, 1] - gt_centers[:, 1]) ** 2
        )

        unique_strides = torch.unique(strides, sorted=True)
        candidate_mask = torch.zeros(num_points, num_gt, dtype=torch.bool, device=device)

        for stride_value in unique_strides.tolist():
            level_idx = torch.nonzero(strides == float(stride_value), as_tuple=False).squeeze(1)
            if level_idx.numel() == 0:
                continue
            topk = min(self.atss_topk, level_idx.numel())
            level_dist = center_dist[level_idx]
            _, top_idx = level_dist.topk(topk, dim=0, largest=False)
            global_idx = level_idx[top_idx]
            gt_indices = torch.arange(num_gt, device=device).unsqueeze(0).expand(topk, num_gt)
            candidate_mask[global_idx.reshape(-1), gt_indices.reshape(-1)] = True

        candidate_ious = anchor_ious.clone()
        candidate_ious[~candidate_mask] = 0.0
        iou_sum = candidate_ious.sum(dim=0)
        iou_count = candidate_mask.float().sum(dim=0).clamp(min=1)
        iou_mean = iou_sum / iou_count
        iou_sq_sum = (candidate_ious ** 2).sum(dim=0)
        iou_var = (iou_sq_sum / iou_count - iou_mean ** 2).clamp(min=0)
        
        iou_thresh = (iou_mean + iou_var.sqrt()).clamp(min=0.1).unsqueeze(0)

        positive_mask = candidate_mask & (anchor_ious >= iou_thresh) & inside_box

        no_pos_gt = ~positive_mask.any(dim=0)
        if no_pos_gt.any():
            inside_ious = anchor_ious.clone()
            inside_ious[~inside_box] = -1.0
            best_points = inside_ious[:, no_pos_gt].argmax(dim=0)
            gt_without_pos = torch.nonzero(no_pos_gt, as_tuple=False).squeeze(1)
            positive_mask[best_points, gt_without_pos] = True

        matched_iou = torch.full((num_points,), -1.0, device=device)
        matched_gt = torch.full((num_points,), -1, dtype=torch.long, device=device)
        for gt_index in range(num_gt):
            pos_idx = torch.nonzero(positive_mask[:, gt_index], as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue
            pos_ious = anchor_ious[pos_idx, gt_index]
            better = pos_ious > matched_iou[pos_idx]
            chosen = pos_idx[better]
            matched_gt[chosen] = gt_index
            matched_iou[chosen] = pos_ious[better]

        pos_mask = matched_gt >= 0
        if not pos_mask.any():
            return {"labels": labels, "boxes": assigned_boxes}

        labels[pos_mask] = gt_labels[matched_gt[pos_mask]]
        assigned_boxes[pos_mask] = gt_boxes[matched_gt[pos_mask]].to(dtype=assigned_boxes.dtype)
        return {"labels": labels, "boxes": assigned_boxes}
