"""Research blocks aligned with the current DenseDet repository."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from model.dense_blocks import ContextBridge, ConvBNAct, Scale
    from model.novel_accuracy_blocks import CCGNConvBlock
except ImportError:
    from dense_blocks import ContextBridge, ConvBNAct, Scale
    from novel_accuracy_blocks import CCGNConvBlock


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FrequencyDecoupledStem(nn.Module):
    """Detail stem that splits the image into low- and high-frequency paths."""

    def __init__(self, out_channels: int, pool_size: int = 5) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 16)
        self.low_conv = ConvBNAct(3, hidden, kernel_size=3, stride=2)
        self.high_conv = ConvBNAct(3, hidden, kernel_size=3, stride=2)
        self.harmonize = ContextBridge(out_channels)

        kernel = torch.ones(3, 1, pool_size, pool_size, dtype=torch.float32)
        kernel /= float(pool_size * pool_size)
        self.register_buffer("blur_kernel", kernel)
        self.pool_pad = pool_size // 2

    def _low_freq(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.blur_kernel, padding=self.pool_pad, groups=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self._low_freq(x)
        high = x - low
        fused = torch.cat([self.low_conv(low), self.high_conv(high)], dim=1)
        return self.harmonize(fused)


class AnisotropicStripEncoder(nn.Module):
    """Drop-in alternative to VSTBlock with strip-aware depthwise branches."""

    def __init__(self, dim: int, strip_k: int = 7, expansion: int = 4) -> None:
        super().__init__()
        hidden = dim * expansion
        gate_hidden = max(dim // 4, 8)

        self.dw_h = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, strip_k),
            padding=(0, strip_k // 2),
            groups=dim,
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            dim,
            dim,
            kernel_size=(strip_k, 1),
            padding=(strip_k // 2, 0),
            groups=dim,
            bias=False,
        )
        self.dw_sq = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        self.branch_w = nn.Parameter(torch.ones(3))
        self.branch_merge = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(dim), dim),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, gate_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.expand = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.project = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(dim), dim)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(torch.full((dim, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        weights = F.softmax(self.branch_w, dim=0)
        merged = (
            weights[0] * self.dw_h(x)
            + weights[1] * self.dw_v(x)
            + weights[2] * self.dw_sq(x)
        )
        merged = self.branch_merge(merged)
        merged = merged * self.channel_gate(merged)
        out = self.act(self.expand(merged))
        out = self.norm(self.project(out))
        return residual + self.layer_scale * out


class ContentAdaptiveDilationBlock(nn.Module):
    """Adaptive alternative to DilatedContextBridge with spatial branch weighting."""

    def __init__(
        self,
        channels: int,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        num_branches = len(self.dilations)
        branch_ch = max(channels // num_branches, 16)
        branch_groups = max(math.gcd(channels, branch_ch), 1)

        self.branches = nn.ModuleList(
            [
                ConvBNAct(
                    channels,
                    branch_ch,
                    kernel_size=3,
                    dilation=dilation,
                    groups=branch_groups,
                )
                for dilation in self.dilations
            ]
        )
        self.fuse = ConvBNAct(branch_ch * num_branches, channels, kernel_size=1)
        gate_hidden = max(channels // 8, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, gate_hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(gate_hidden, num_branches, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        weights = F.softmax(self.gate(x), dim=1)
        weighted = torch.cat(
            [weights[:, index : index + 1] * branch for index, branch in enumerate(branch_outs)],
            dim=1,
        )
        refined = self.fuse(weighted)
        return F.silu(x + refined, inplace=True)


class GradientPreservationNeck(nn.Module):
    """Cross-level residual adapter that can be inserted before the main neck."""

    def __init__(self, channels: tuple[int, ...], aux_dim: int = 64) -> None:
        super().__init__()
        self.channels = tuple(int(value) for value in channels)
        total_dim = aux_dim * len(self.channels)
        hidden = max(total_dim // 2, aux_dim)

        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, aux_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(aux_dim), aux_dim),
                    nn.GELU(),
                )
                for channel in self.channels
            ]
        )
        self.cross_level = nn.Sequential(
            nn.Conv2d(total_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, total_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.correctors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(aux_dim, channel, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(channel), channel),
                )
                for channel in self.channels
            ]
        )
        self.level_scale = nn.ParameterList(
            [nn.Parameter(torch.zeros(channel, 1, 1)) for channel in self.channels]
        )

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(features) != len(self.channels):
            raise ValueError(
                f"Expected {len(self.channels)} feature levels, got {len(features)}."
            )

        projections = [project(feature) for project, feature in zip(self.projectors, features)]
        pooled = [F.adaptive_avg_pool2d(projection, 1) for projection in projections]
        global_ctx = torch.cat(pooled, dim=1)
        gates = self.cross_level(global_ctx).chunk(len(features), dim=1)

        corrected = []
        for feature, projection, gate, corrector, scale in zip(
            features,
            projections,
            gates,
            self.correctors,
            self.level_scale,
        ):
            correction = corrector(projection * gate)
            corrected.append(feature + scale * correction)
        return tuple(corrected)


class NovelHeadTower(nn.Sequential):
    def __init__(self, channels: int, depth: int = 2) -> None:
        layers = []
        for _ in range(depth):
            layers.append(ConvBNAct(channels, channels, groups=channels))
            layers.append(ConvBNAct(channels, channels, kernel_size=1))
        super().__init__(*layers)


class EvidentialQualityHead(nn.Module):
    """Repo-compatible evidential head.

    This head keeps the current DenseDet contract intact:
    - `quality` remains a list of 1-channel logits so the existing loss/decode path
      can still run without modification.
    - `uncertainty` and `quality_evidence` are exposed as extra outputs for future
      research losses or post-processing.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        levels: int,
        depth: int = 2,
        use_class_conditional_gn: bool = False,
    ) -> None:
        super().__init__()
        self.use_class_conditional_gn = use_class_conditional_gn
        self.cls_tower = NovelHeadTower(channels, depth=depth)
        self.reg_tower = NovelHeadTower(channels, depth=depth)
        if self.use_class_conditional_gn:
            self.cls_ccgn = CCGNConvBlock(channels, num_classes)
            self.reg_ccgn = CCGNConvBlock(channels, num_classes)
        else:
            self.cls_ccgn = None
            self.reg_ccgn = None

        self.cls_pred = nn.Conv2d(channels, num_classes, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(channels, 4, kernel_size=3, padding=1)
        self.scales = nn.ModuleList([Scale() for _ in range(levels)])

        self.task_align = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.quality_pred = nn.Conv2d(channels, 2, kernel_size=3, padding=1)

        self._init_biases()

    def _init_biases(self, prior_prob: float = 0.01) -> None:
        cls_bias = math.log(prior_prob / (1.0 - prior_prob))
        nn.init.constant_(self.cls_pred.bias, cls_bias)
        nn.init.constant_(self.box_pred.bias, 1.0)
        nn.init.zeros_(self.quality_pred.weight)
        nn.init.zeros_(self.quality_pred.bias)

    def forward(self, features: tuple[torch.Tensor, ...]) -> dict[str, list[torch.Tensor]]:
        outputs: dict[str, list[torch.Tensor]] = {
            "cls": [],
            "box": [],
            "quality": [],
            "uncertainty": [],
            "quality_evidence": [],
        }

        for feature, scale in zip(features, self.scales):
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)

            cls_logits = self.cls_pred(cls_feat)
            if self.use_class_conditional_gn:
                assert self.cls_ccgn is not None and self.reg_ccgn is not None
                cls_feat = self.cls_ccgn(cls_feat, cls_logits)
                reg_feat = self.reg_ccgn(reg_feat, cls_logits)
                cls_logits = self.cls_pred(cls_feat)
            outputs["cls"].append(cls_logits)
            with torch.autocast(device_type=feature.device.type, enabled=False):
                reg_logits = self.box_pred(reg_feat.float())
                reg_distances = F.softplus(scale(reg_logits)).clamp(max=1e4)
            outputs["box"].append(reg_distances)

            aligned = self.task_align(torch.cat([cls_feat, reg_feat], dim=1))
            with torch.autocast(device_type=feature.device.type, enabled=False):
                evidence_logits = self.quality_pred(aligned.float())
                evidence = F.softplus(evidence_logits) + 1.0
                alpha_pos = evidence[:, 0:1]
                alpha_neg = evidence[:, 1:2]
                quality_logits = torch.log(alpha_pos) - torch.log(alpha_neg)
                uncertainty = 2.0 / (alpha_pos + alpha_neg)

            outputs["quality"].append(quality_logits.to(dtype=feature.dtype))
            outputs["uncertainty"].append(uncertainty.to(dtype=feature.dtype))
            outputs["quality_evidence"].append(evidence.to(dtype=feature.dtype))

        return outputs


__all__ = [
    "AnisotropicStripEncoder",
    "ContentAdaptiveDilationBlock",
    "EvidentialQualityHead",
    "FrequencyDecoupledStem",
    "GradientPreservationNeck",
]


if __name__ == "__main__":
    batch = 2
    channels = 128
    spatial = 40

    stem = FrequencyDecoupledStem(channels)
    images = torch.randn(batch, 3, spatial * 2, spatial * 2)
    print("FDS :", tuple(stem(images).shape))

    encoder = AnisotropicStripEncoder(channels)
    feature = torch.randn(batch, channels, spatial, spatial)
    print("ASE :", tuple(encoder(feature).shape))

    cadb = ContentAdaptiveDilationBlock(channels)
    print("CADB:", tuple(cadb(feature).shape))

    feature_channels = (32, 64, 128, 256)
    gpn = GradientPreservationNeck(feature_channels)
    pyramid = tuple(
        torch.randn(batch, channel, spatial // (2**index), spatial // (2**index))
        for index, channel in enumerate(feature_channels)
    )
    print("GPN :", [tuple(t.shape) for t in gpn(pyramid)])

    head = EvidentialQualityHead(channels, num_classes=6, levels=4)
    head_out = head(
        tuple(torch.randn(batch, channels, spatial // (2**index), spatial // (2**index)) for index in range(4))
    )
    print("EQH quality      :", [tuple(t.shape) for t in head_out["quality"]])
    print("EQH uncertainty :", [tuple(t.shape) for t in head_out["uncertainty"]])
