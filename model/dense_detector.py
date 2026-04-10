"""Lightweight PRISM-based dense detector."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from model.dense_blocks import ContextAwareFusion, ConvBNAct, DilatedContextBridge, Scale, SpatialChannelGate
from model.prism_backbone import PRISMBackbone
from utils.box_ops import distance_to_boxes, xyxy_abs_to_cxcywh_norm
from utils.points import build_points


@dataclass(frozen=True)
class VariantConfig:
    head_channels: int


VARIANTS = {
    "tiny": VariantConfig(head_channels=96),
    "small": VariantConfig(head_channels=128),
}


class CAFPNNeck(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("CAFPNNeck expects four backbone feature levels.")

        self.lateral = nn.ModuleList(
            [ConvBNAct(channels, out_channels, kernel_size=1) for channels in in_channels]
        )
        self.downsample = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, stride=2) for _ in range(3)]
        )
        self.context_inject = nn.ModuleList(
            [SpatialChannelGate(out_channels) for _ in range(4)]
        )

        self.topdown_p4 = ContextAwareFusion(out_channels, inputs=2)
        self.topdown_p3 = ContextAwareFusion(out_channels, inputs=2)
        self.topdown_p2 = ContextAwareFusion(out_channels, inputs=2)
        self.bottomup_p3 = ContextAwareFusion(out_channels, inputs=2)
        self.bottomup_p4 = ContextAwareFusion(out_channels, inputs=2)
        self.bottomup_p5 = ContextAwareFusion(out_channels, inputs=2)
        self.refine = nn.ModuleList(
            [DilatedContextBridge(out_channels) for _ in range(4)]
        )

    def _apply_context(self, feature: torch.Tensor, context: torch.Tensor, index: int) -> torch.Tensor:
        context = F.interpolate(context, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return feature + self.context_inject[index](context) * feature

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        p2, p3, p4, p5 = [layer(feature) for layer, feature in zip(self.lateral, features)]
        global_context = p5

        p4_td = self.topdown_p4([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")])
        p4_td = self._apply_context(p4_td, global_context, 0)

        p3_td = self.topdown_p3([p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")])
        p3_td = self._apply_context(p3_td, global_context, 1)

        p2_td = self.topdown_p2([p2, F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")])
        p2_td = self._apply_context(p2_td, global_context, 2)

        p3_out = self.bottomup_p3([p3_td, self.downsample[0](p2_td)])
        p4_out = self.bottomup_p4([p4_td, self.downsample[1](p3_out)])
        p5_out = self.bottomup_p5([p5, self.downsample[2](p4_out)])
        p5_out = self._apply_context(p5_out, global_context, 3)

        outputs = [p2_td, p3_out, p4_out, p5_out]
        return tuple(block(feature) for block, feature in zip(self.refine, outputs))


class HeadTower(nn.Sequential):
    def __init__(self, channels: int, depth: int = 2) -> None:
        layers = []
        for _ in range(depth):
            layers.append(ConvBNAct(channels, channels, groups=channels))
            layers.append(ConvBNAct(channels, channels, kernel_size=1))
        super().__init__(*layers)


class DenseHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        levels: int,
        depth: int = 2,
        use_quality_head: bool = True,
    ) -> None:
        super().__init__()
        self.use_quality_head = bool(use_quality_head)
        self.cls_tower = HeadTower(channels, depth=depth)
        self.reg_tower = HeadTower(channels, depth=depth)
        self.cls_pred = nn.Conv2d(channels, num_classes, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        self.scales = nn.ModuleList([Scale() for _ in range(levels)])

        if self.use_quality_head:
            self.task_align = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            self.quality_pred = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self._init_biases()

    def _init_biases(self, prior_prob: float = 0.01) -> None:
        cls_bias = math.log(prior_prob / (1.0 - prior_prob))
        nn.init.constant_(self.cls_pred.bias, cls_bias)
        nn.init.constant_(self.box_pred.bias, 1.0)
        if self.use_quality_head:
            nn.init.constant_(self.quality_pred.bias, 0.0)

    def forward(self, features: tuple[torch.Tensor, ...]) -> dict[str, list[torch.Tensor]]:
        outputs: dict[str, list[torch.Tensor]] = {"cls": [], "box": []}
        if self.use_quality_head:
            outputs["quality"] = []

        for feature, scale in zip(features, self.scales):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)

            outputs["cls"].append(self.cls_pred(cls_feat))
            with torch.autocast(device_type=feature.device.type, enabled=False):
                reg_logits = self.box_pred(reg_feat.float())
                reg_distances = F.softplus(scale(reg_logits)).clamp(max=1e4)
            outputs["box"].append(reg_distances)

            if self.use_quality_head:
                aligned = self.task_align(torch.cat([cls_feat, reg_feat], dim=1))
                outputs["quality"].append(self.quality_pred(aligned))

        return outputs


class DenseDet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        backbone_dims: tuple[int, int, int, int] = (16, 32, 64, 128),
        backbone_depths: tuple[int, int, int, int] = (2, 2, 4, 2),
        head_depth: int = 2,
        use_quality_head: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Expected one of {sorted(VARIANTS)}.")

        cfg = VARIANTS[variant]
        self.num_classes = int(num_classes)
        self.variant = variant
        self.backbone_dims = tuple(int(value) for value in backbone_dims)
        self.backbone_depths = tuple(int(value) for value in backbone_depths)
        self.head_depth = int(head_depth)
        self.use_quality_head = bool(use_quality_head)

        self.backbone = PRISMBackbone(dims=self.backbone_dims, depths=self.backbone_depths)
        self.neck = CAFPNNeck(self.backbone.channels, cfg.head_channels)
        self.strides = self.backbone.reductions
        self.head = DenseHead(
            channels=cfg.head_channels,
            num_classes=self.num_classes,
            levels=len(self.strides),
            depth=self.head_depth,
            use_quality_head=self.use_quality_head,
        )

    def forward(self, images: torch.Tensor) -> dict[str, list[torch.Tensor] | tuple[int, ...]]:
        features = self.backbone(images)
        pyramid = self.neck(features)
        outputs = self.head(pyramid)
        outputs["strides"] = self.strides
        outputs["image_size"] = tuple(images.shape[-2:])
        return outputs

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.05,
        nms_iou: float = 0.6,
        max_det: int = 300,
    ) -> list[dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()
        outputs = self(images)
        predictions = decode_predictions(
            outputs,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            max_det=max_det,
        )
        if was_training:
            self.train()
        return predictions

    def param_count(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def decode_predictions(
    outputs: dict[str, list[torch.Tensor] | tuple[int, ...]],
    conf_threshold: float,
    nms_iou: float,
    max_det: int,
) -> list[dict[str, torch.Tensor]]:
    cls_levels = outputs["cls"]  # type: ignore[index]
    box_levels = outputs["box"]  # type: ignore[index]
    quality_levels = outputs.get("quality")  # type: ignore[assignment]
    strides = outputs["strides"]  # type: ignore[index]
    image_h, image_w = outputs["image_size"]  # type: ignore[index]

    batch_size = cls_levels[0].shape[0]
    results: list[dict[str, torch.Tensor]] = []

    for batch_index in range(batch_size):
        image_boxes = []
        image_scores = []
        image_labels = []

        level_iter = zip(
            cls_levels,
            box_levels,
            strides,
            quality_levels if quality_levels is not None else [None] * len(cls_levels),
        )
        for cls_map, box_map, stride, quality_map in level_iter:
            _, num_classes, feat_h, feat_w = cls_map.shape
            points = build_points(feat_h, feat_w, int(stride), cls_map.device, cls_map.dtype)

            cls_scores = cls_map[batch_index].permute(1, 2, 0).reshape(-1, num_classes).sigmoid()
            box_distances = box_map[batch_index].permute(1, 2, 0).reshape(-1, 4) * int(stride)

            if quality_map is not None:
                quality = quality_map[batch_index].permute(1, 2, 0).reshape(-1).sigmoid()
                raw_scores, labels = cls_scores.max(dim=1)
                scores = (raw_scores * quality).clamp(min=0.0, max=1.0)
            else:
                scores, labels = cls_scores.max(dim=1)

            keep = scores > conf_threshold
            if not keep.any():
                continue

            boxes = distance_to_boxes(points[keep], box_distances[keep])
            boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=image_w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=image_h)

            image_boxes.append(boxes)
            image_scores.append(scores[keep])
            image_labels.append(labels[keep])

        if not image_boxes:
            device = cls_levels[0].device
            results.append(
                {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "confidences": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                }
            )
            continue

        boxes = torch.cat(image_boxes, dim=0)
        scores = torch.cat(image_scores, dim=0)
        labels = torch.cat(image_labels, dim=0)

        keep = batched_nms(boxes, scores, labels, nms_iou)[:max_det]
        boxes = xyxy_abs_to_cxcywh_norm(boxes[keep], image_h=image_h, image_w=image_w)
        scores = scores[keep]
        labels = labels[keep]

        results.append(
            {
                "boxes": boxes,
                "scores": scores,
                "confidences": scores,
                "labels": labels,
            }
        )

    return results


if __name__ == "__main__":
    model = DenseDet(num_classes=6)
    model.eval()
    images = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(images)
        predictions = model.predict(images, conf_threshold=0.25)
    print("levels:", len(outputs["cls"]))
    print("strides:", outputs["strides"])
    print("preds:", len(predictions), predictions[0]["boxes"].shape)
