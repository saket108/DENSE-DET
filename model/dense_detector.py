"""Dense detector baseline for fast mAP benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from model.dense_blocks import (
    ContextAwareFusion,
    ConvBNAct,
    DetailStem,
    DilatedContextBridge,
    Scale,
    SpatialChannelGate,
    WeightedFeatureFusion,
)
from model.vst_backbone import VSTBackbone
from utils.box_ops import distance_to_boxes, xyxy_abs_to_cxcywh_norm
from utils.points import build_points


BACKBONE_ALIASES = {
    "vst": "vst",
    "vst_backbone": "vst",
    "vst_s": "vst",
    "vst_small": "vst",
    "vst_l": "vst_large",
    "vst_large": "vst_large",
    "vst_custom": "vst_custom",
    "custom": "vst_custom",
}

VST_PRESETS = {
    "vst": {
        "dims": (32, 64, 128, 256),
        "depths": (2, 2, 4, 2),
    },
    "vst_large": {
        "dims": (60, 120, 240, 480),
        "depths": (2, 2, 4, 2),
    },
}


@dataclass(frozen=True)
class VariantConfig:
    detail_channels: int
    head_channels: int


VARIANTS = {
    "tiny": VariantConfig(detail_channels=32, head_channels=96),
    "small": VariantConfig(detail_channels=48, head_channels=128),
    "base": VariantConfig(detail_channels=64, head_channels=160),
}


def _extract_backbone_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("backbone", "backbone_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break

    if not isinstance(payload, dict):
        raise ValueError("Unsupported pretrained backbone checkpoint format.")

    filtered: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue

        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        if normalized.startswith("backbone."):
            normalized = normalized[len("backbone.") :]

        filtered[normalized] = value

    if not filtered:
        raise ValueError("No tensor weights found in pretrained backbone checkpoint.")
    return filtered


def _load_backbone_weights(backbone: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_backbone_state_dict(checkpoint)
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)

    print(f"  Loaded pretrained backbone weights from: {checkpoint_path}")
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        print(f"    Missing keys: {preview}{suffix}")
    if unexpected:
        preview = ", ".join(unexpected[:5])
        suffix = " ..." if len(unexpected) > 5 else ""
        print(f"    Unexpected keys: {preview}{suffix}")


def build_backbone(
    model_name: str,
    pretrained: bool = False,
    pretrained_path: str | None = None,
    dims: tuple[int, int, int, int] | None = None,
    depths: tuple[int, int, int, int] | None = None,
) -> nn.Module:
    resolved_name = BACKBONE_ALIASES.get(model_name, model_name)
    if (dims is None) != (depths is None):
        raise ValueError("backbone_dims and backbone_depths must be provided together.")

    if dims is not None and depths is not None:
        dims = tuple(int(value) for value in dims)
        depths = tuple(int(value) for value in depths)
        backbone = VSTBackbone(dims=dims, depths=depths)
    elif resolved_name in VST_PRESETS:
        preset = VST_PRESETS[resolved_name]
        backbone = VSTBackbone(dims=preset["dims"], depths=preset["depths"])
    else:
        supported = ", ".join(sorted(VST_PRESETS))
        raise ValueError(
            f"Unsupported backbone '{model_name}'. DenseDet is configured for VST only. "
            f"Supported preset names: {supported}. "
            "For a custom VST layout, provide backbone_dims and backbone_depths."
        )

    if pretrained_path:
        _load_backbone_weights(backbone, pretrained_path)
    elif pretrained:
        print("  VST backbone uses random initialization only (ignoring pretrained_backbone=True)")
    return backbone


class BiFusionNeck(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: int,
        use_polarized_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_detail_branch = len(in_channels) == 5
        self.lateral = nn.ModuleList(
            [ConvBNAct(ch, out_channels, kernel_size=1) for ch in in_channels]
        )
        self.downsample = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, stride=2) for _ in range(4 if self.use_detail_branch else 3)]
        )

        self.topdown_p4 = WeightedFeatureFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.topdown_p3 = WeightedFeatureFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.topdown_p2 = WeightedFeatureFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p3 = WeightedFeatureFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p4 = WeightedFeatureFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p5 = WeightedFeatureFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )

        if self.use_detail_branch:
            self.topdown_p1 = WeightedFeatureFusion(
                out_channels,
                inputs=2,
                use_polarized_gate=use_polarized_attention,
            )
            self.bottomup_p2 = WeightedFeatureFusion(
                out_channels,
                inputs=2,
                use_polarized_gate=use_polarized_attention,
            )

        self.refine = nn.ModuleList(
            [DilatedContextBridge(out_channels) for _ in range(5 if self.use_detail_branch else 4)]
        )

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        projected = [layer(x) for layer, x in zip(self.lateral, features)]

        if self.use_detail_branch:
            p1, p2, p3, p4, p5 = projected
            p4_td = self.topdown_p4([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")])
            p3_td = self.topdown_p3([p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")])
            p2_td = self.topdown_p2([p2, F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")])
            p1_td = self.topdown_p1([p1, F.interpolate(p2_td, size=p1.shape[-2:], mode="nearest")])

            p2_out = self.bottomup_p2([p2_td, self.downsample[0](p1_td)])
            p3_out = self.bottomup_p3([p3_td, self.downsample[1](p2_out)])
            p4_out = self.bottomup_p4([p4_td, self.downsample[2](p3_out)])
            p5_out = self.bottomup_p5([p5, self.downsample[3](p4_out)])
            outputs = [p1_td, p2_out, p3_out, p4_out, p5_out]
        else:
            p2, p3, p4, p5 = projected
            p4_td = self.topdown_p4([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")])
            p3_td = self.topdown_p3([p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")])
            p2_td = self.topdown_p2([p2, F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")])

            p3_out = self.bottomup_p3([p3_td, self.downsample[0](p2_td)])
            p4_out = self.bottomup_p4([p4_td, self.downsample[1](p3_out)])
            p5_out = self.bottomup_p5([p5, self.downsample[2](p4_out)])
            outputs = [p2_td, p3_out, p4_out, p5_out]

        return tuple(block(feature) for block, feature in zip(self.refine, outputs))


class CAFPNNeck(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: int,
        use_polarized_attention: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("CAFPNNeck expects four backbone feature levels.")

        self.lateral = nn.ModuleList(
            [ConvBNAct(ch, out_channels, kernel_size=1) for ch in in_channels]
        )
        self.downsample = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, stride=2) for _ in range(3)]
        )
        self.context_inject = nn.ModuleList(
            [
                SpatialChannelGate(
                    out_channels,
                    polarized=use_polarized_attention,
                )
                for _ in range(4)
            ]
        )

        self.topdown_p4 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.topdown_p3 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.topdown_p2 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p3 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p4 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p5 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.refine = nn.ModuleList([DilatedContextBridge(out_channels) for _ in range(4)])

    def _apply_context(self, feature: torch.Tensor, context: torch.Tensor, index: int) -> torch.Tensor:
        ctx = F.interpolate(context, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return feature + self.context_inject[index](ctx) * feature

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        p2, p3, p4, p5 = [layer(x) for layer, x in zip(self.lateral, features)]
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


class CAFPNP2Neck(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, ...],
        out_channels: int,
        use_polarized_attention: bool = False,
    ) -> None:
        super().__init__()
        if len(in_channels) != 5:
            raise ValueError("CAFPNP2Neck expects a detail feature plus four backbone levels.")

        self.lateral = nn.ModuleList(
            [ConvBNAct(ch, out_channels, kernel_size=1) for ch in in_channels]
        )
        self.downsample = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, stride=2) for _ in range(4)]
        )
        self.context_inject = nn.ModuleList(
            [
                SpatialChannelGate(
                    out_channels,
                    polarized=use_polarized_attention,
                )
                for _ in range(5)
            ]
        )

        self.topdown_p4 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.topdown_p3 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.topdown_p2 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.topdown_p1 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p2 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p3 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p4 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.bottomup_p5 = ContextAwareFusion(
            out_channels,
            inputs=2,
            use_polarized_gate=use_polarized_attention,
        )
        self.refine = nn.ModuleList([DilatedContextBridge(out_channels) for _ in range(5)])

    def _apply_context(self, feature: torch.Tensor, context: torch.Tensor, index: int) -> torch.Tensor:
        ctx = F.interpolate(context, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return feature + self.context_inject[index](ctx) * feature

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        p1, p2, p3, p4, p5 = [layer(x) for layer, x in zip(self.lateral, features)]
        global_context = p5

        p4_td = self.topdown_p4([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")])
        p4_td = self._apply_context(p4_td, global_context, 3)

        p3_td = self.topdown_p3([p3, F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")])
        p3_td = self._apply_context(p3_td, global_context, 2)

        p2_td = self.topdown_p2([p2, F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")])
        p2_td = self._apply_context(p2_td, global_context, 1)

        p1_td = self.topdown_p1([p1, F.interpolate(p2_td, size=p1.shape[-2:], mode="nearest")])
        p1_td = self._apply_context(p1_td, global_context, 0)

        p2_out = self.bottomup_p2([p2_td, self.downsample[0](p1_td)])
        p3_out = self.bottomup_p3([p3_td, self.downsample[1](p2_out)])
        p4_out = self.bottomup_p4([p4_td, self.downsample[2](p3_out)])
        p5_out = self.bottomup_p5([p5, self.downsample[3](p4_out)])
        p5_out = self._apply_context(p5_out, global_context, 4)

        outputs = [p1_td, p2_out, p3_out, p4_out, p5_out]
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
        self.use_quality_head = use_quality_head
        self.cls_tower = HeadTower(channels, depth=depth)
        self.reg_tower = HeadTower(channels, depth=depth)

        self.cls_pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.box_pred = nn.Conv2d(channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale() for _ in range(levels)])

        if use_quality_head:
            self.task_align = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            self.quality_pred = nn.Conv2d(channels, 1, 3, padding=1)

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


class AuxiliaryDenseHead(nn.Module):
    """Lightweight auxiliary heads used only during training."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        strides: tuple[int, ...],
        target_strides: tuple[int, ...] = (8, 16),
        use_quality_head: bool = True,
    ) -> None:
        super().__init__()
        target_stride_set = {int(value) for value in target_strides}
        self.target_indices = [
            index for index, stride in enumerate(strides) if int(stride) in target_stride_set
        ]
        self.target_strides = tuple(int(strides[index]) for index in self.target_indices)
        self.use_quality_head = use_quality_head

        self.cls_preds = nn.ModuleList(
            [nn.Conv2d(channels, num_classes, 3, padding=1) for _ in self.target_indices]
        )
        self.box_preds = nn.ModuleList(
            [nn.Conv2d(channels, 4, 3, padding=1) for _ in self.target_indices]
        )
        self.scales = nn.ModuleList([Scale() for _ in self.target_indices])
        if self.use_quality_head:
            self.quality_preds = nn.ModuleList(
                [nn.Conv2d(channels, 1, 3, padding=1) for _ in self.target_indices]
            )
        else:
            self.quality_preds = None

        self._init_biases()

    def _init_biases(self, prior_prob: float = 0.01) -> None:
        cls_bias = math.log(prior_prob / (1.0 - prior_prob))
        for layer in self.cls_preds:
            nn.init.constant_(layer.bias, cls_bias)
        for layer in self.box_preds:
            nn.init.constant_(layer.bias, 1.0)
        if self.quality_preds is not None:
            for layer in self.quality_preds:
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        image_size: tuple[int, int],
    ) -> list[dict[str, list[torch.Tensor] | tuple[int, ...]]]:
        outputs = []
        for head_index, feature_index in enumerate(self.target_indices):
            feature = features[feature_index]
            current: dict[str, list[torch.Tensor] | tuple[int, ...]] = {
                "cls": [self.cls_preds[head_index](feature)],
            }
            with torch.autocast(device_type=feature.device.type, enabled=False):
                reg_logits = self.box_preds[head_index](feature.float())
                reg_distances = F.softplus(self.scales[head_index](reg_logits)).clamp(max=1e4)
            current["box"] = [reg_distances]
            if self.quality_preds is not None:
                current["quality"] = [self.quality_preds[head_index](feature)]
            current["strides"] = (self.target_strides[head_index],)
            current["image_size"] = image_size
            outputs.append(current)
        return outputs


class DenseDet(nn.Module):
    """Dense detection baseline with a standard FPN-style pyramid."""

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        backbone_name: str = "vst",
        pretrained_backbone: bool = False,
        pretrained_backbone_path: str | None = None,
        backbone_dims: tuple[int, int, int, int] | None = None,
        backbone_depths: tuple[int, int, int, int] | None = None,
        neck_name: str = "cafpn",
        head_depth: int = 2,
        use_detail_branch: bool = False,
        use_quality_head: bool = True,
        use_auxiliary_heads: bool = False,
        use_polarized_attention: bool = False,
        auxiliary_strides: tuple[int, ...] = (8, 16),
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Expected one of {sorted(VARIANTS)}.")

        cfg = VARIANTS[variant]
        self.num_classes = num_classes
        self.variant = variant
        self.backbone_name = BACKBONE_ALIASES.get(backbone_name, backbone_name)
        self.pretrained_backbone = pretrained_backbone
        self.pretrained_backbone_path = pretrained_backbone_path
        self.backbone_dims = backbone_dims
        self.backbone_depths = backbone_depths
        self.neck_name = neck_name
        self.head_depth = head_depth
        self.use_detail_branch = use_detail_branch
        self.use_quality_head = use_quality_head
        self.use_auxiliary_heads = use_auxiliary_heads
        self.use_polarized_attention = use_polarized_attention
        self.auxiliary_strides = tuple(int(value) for value in auxiliary_strides)

        self.backbone = build_backbone(
            self.backbone_name,
            pretrained=pretrained_backbone,
            pretrained_path=pretrained_backbone_path,
            dims=backbone_dims,
            depths=backbone_depths,
        )
        self.pretrained_backbone = bool(pretrained_backbone_path)
        self.backbone_dims = tuple(int(value) for value in self.backbone.channels)
        self.backbone_depths = tuple(int(value) for value in getattr(self.backbone, "depths", ()))
        self.uses_stride2_path = use_detail_branch or neck_name == "cafpn_p2"
        self.detail_stem = DetailStem(cfg.detail_channels) if self.uses_stride2_path else None

        base_channels = self.backbone.channels
        neck_in_channels = (
            (cfg.detail_channels, *base_channels) if self.uses_stride2_path else base_channels
        )

        if neck_name == "bifusion":
            self.neck = BiFusionNeck(
                neck_in_channels,
                cfg.head_channels,
                use_polarized_attention=use_polarized_attention,
            )
        elif neck_name == "cafpn":
            if use_detail_branch:
                raise ValueError("neck_name='cafpn' does not support use_detail_branch=True.")
            self.neck = CAFPNNeck(
                neck_in_channels,
                cfg.head_channels,
                use_polarized_attention=use_polarized_attention,
            )
        elif neck_name == "cafpn_p2":
            self.neck = CAFPNP2Neck(
                neck_in_channels,
                cfg.head_channels,
                use_polarized_attention=use_polarized_attention,
            )
        else:
            raise ValueError(
                f"Unknown neck '{neck_name}'. Expected one of ['bifusion', 'cafpn', 'cafpn_p2']."
            )

        self.strides = (
            (2, *self.backbone.reductions)
            if self.uses_stride2_path
            else self.backbone.reductions
        )
        self.head = DenseHead(
            cfg.head_channels,
            num_classes,
            levels=len(self.strides),
            depth=head_depth,
            use_quality_head=use_quality_head,
        )
        self.aux_head = None
        if self.use_auxiliary_heads:
            self.aux_head = AuxiliaryDenseHead(
                cfg.head_channels,
                num_classes,
                strides=self.strides,
                target_strides=self.auxiliary_strides,
                use_quality_head=use_quality_head,
            )

    def forward(self, images: torch.Tensor) -> dict[str, list[torch.Tensor] | tuple[int, ...]]:
        features = self.backbone(images)
        if self.uses_stride2_path:
            assert self.detail_stem is not None
            detail = self.detail_stem(images)
            pyramid = self.neck((detail, *features))
        else:
            pyramid = self.neck(features)

        outputs = self.head(pyramid)
        outputs["strides"] = self.strides
        outputs["image_size"] = tuple(images.shape[-2:])
        if self.training and self.aux_head is not None:
            outputs["aux_outputs"] = self.aux_head(pyramid, outputs["image_size"])
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
    results = []

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
                # Use quality to modulate scores for better precision
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
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        boxes = xyxy_abs_to_cxcywh_norm(boxes, image_h=image_h, image_w=image_w)

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
    model = DenseDet(num_classes=6, backbone_name="vst", pretrained_backbone=False)
    model.eval()
    images = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(images)
        preds = model.predict(images, conf_threshold=0.25)
    print("levels:", len(outputs["cls"]))
    print("strides:", outputs["strides"])
    print("preds:", len(preds), preds[0]["boxes"].shape)
