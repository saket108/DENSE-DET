"""Building blocks for the dense detection baseline."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8) -> int:
    return int((value + divisor - 1) // divisor * divisor)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: bool = True,
    ) -> None:
        padding = kernel_size // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class EdgeResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 4.0,
    ) -> None:
        super().__init__()
        hidden = make_divisible(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = (
            ConvBNAct(in_channels, hidden, kernel_size=1)
            if hidden != in_channels
            else nn.Identity()
        )
        self.depthwise = ConvBNAct(hidden, hidden, stride=stride, groups=hidden)
        self.se = SqueezeExcite(hidden)
        self.project = ConvBNAct(hidden, out_channels, kernel_size=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.depthwise(y)
        y = self.se(y)
        y = self.project(y)
        if self.use_residual:
            y = y + x
        return F.silu(y, inplace=True)


class SpatialChannelGate(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        polarized: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.directional_gate = PolarizedGate(channels) if polarized else None
        self.spatial_conv = None
        if self.directional_gate is None:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_fc(x)
        if self.directional_gate is not None:
            return self.directional_gate(x)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        assert self.spatial_conv is not None
        return x * self.spatial_conv(torch.cat([avg, mx], dim=1))


class PolarizedGate(nn.Module):
    """Directional gate that preserves thin, elongated defect responses."""

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.height_proj = nn.Conv2d(hidden, 1, kernel_size=1)
        self.width_proj = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.reduce(x)
        pooled_h = shared.mean(dim=3, keepdim=True)
        pooled_w = shared.mean(dim=2, keepdim=True)
        mask = torch.sigmoid(self.height_proj(pooled_h) + self.width_proj(pooled_w))
        return x * mask


class DilatedContextBridge(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_ch = max(channels // 4, 16)
        self.d1 = ConvBNAct(channels, branch_ch, kernel_size=3, dilation=1, groups=branch_ch)
        self.d2 = ConvBNAct(channels, branch_ch, kernel_size=3, dilation=2, groups=branch_ch)
        self.d4 = ConvBNAct(channels, branch_ch, kernel_size=3, dilation=4, groups=branch_ch)
        self.d8 = ConvBNAct(channels, branch_ch, kernel_size=3, dilation=8, groups=branch_ch)
        self.fuse = ConvBNAct(branch_ch * 4, channels, kernel_size=1)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = self.fuse(torch.cat([self.d1(x), self.d2(x), self.d4(x), self.d8(x)], dim=1))
        gate = self.global_gate(x)
        return F.silu(x + multi * gate, inplace=True)


class ContextBridge(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 16)
        self.local = ConvBNAct(channels, channels, groups=channels)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.mix = ConvBNAct(channels, channels, kernel_size=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        gated = local * self.global_gate(x)
        return F.silu(x + self.mix(gated), inplace=True)


class DetailStem(nn.Sequential):
    def __init__(self, out_channels: int) -> None:
        hidden = max(out_channels // 2, 16)
        super().__init__(
            ConvBNAct(3, hidden, stride=2),
            ConvBNAct(hidden, out_channels),
            ContextBridge(out_channels),
        )


class WeightedFeatureFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        inputs: int,
        use_polarized_gate: bool = False,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))
        self.project = ConvBNAct(channels, channels, kernel_size=1)
        self.directional_gate = PolarizedGate(channels) if use_polarized_gate else None

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-4)
        fused = sum(weight * feature for weight, feature in zip(weights, features))
        fused = self.project(fused)
        if self.directional_gate is not None:
            fused = self.directional_gate(fused)
        return fused


class ContextAwareFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        inputs: int,
        use_polarized_gate: bool = False,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(inputs))
        hidden = max(channels // 4, 16)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.directional_gate = PolarizedGate(channels) if use_polarized_gate else None
        self.spatial_gate = None
        if self.directional_gate is None:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid(),
            )
        self.local = ConvBNAct(channels, channels, groups=channels)
        self.mix = ConvBNAct(channels, channels, kernel_size=1, act=False)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + 1e-4)
        fused = sum(weight * feature for weight, feature in zip(weights, features))
        ch_gated = fused * self.channel_gate(fused)
        if self.directional_gate is not None:
            gated = self.directional_gate(ch_gated)
        else:
            avg = ch_gated.mean(dim=1, keepdim=True)
            mx = ch_gated.max(dim=1, keepdim=True).values
            assert self.spatial_gate is not None
            gated = ch_gated * self.spatial_gate(torch.cat([avg, mx], dim=1))
        refined = self.local(gated) + fused
        return F.silu(self.mix(refined) + fused, inplace=True)


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
