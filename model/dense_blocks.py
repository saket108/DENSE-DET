"""Core building blocks for the lightweight DenseDet model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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


class SpatialChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_fc(x)
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        return x * self.spatial_conv(torch.cat([avg, mx], dim=1))


class DilatedContextBridge(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = max(channels // 4, 16)
        self.d1 = ConvBNAct(channels, branch_channels, kernel_size=3, dilation=1, groups=branch_channels)
        self.d2 = ConvBNAct(channels, branch_channels, kernel_size=3, dilation=2, groups=branch_channels)
        self.d4 = ConvBNAct(channels, branch_channels, kernel_size=3, dilation=4, groups=branch_channels)
        self.d8 = ConvBNAct(channels, branch_channels, kernel_size=3, dilation=8, groups=branch_channels)
        self.fuse = ConvBNAct(branch_channels * 4, channels, kernel_size=1)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = self.fuse(torch.cat([self.d1(x), self.d2(x), self.d4(x), self.d8(x)], dim=1))
        return F.silu(x + multi * self.global_gate(x), inplace=True)


class ContextAwareFusion(nn.Module):
    def __init__(self, channels: int, inputs: int) -> None:
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
        gated = fused * self.channel_gate(fused)
        avg = gated.mean(dim=1, keepdim=True)
        mx = gated.max(dim=1, keepdim=True).values
        gated = gated * self.spatial_gate(torch.cat([avg, mx], dim=1))
        refined = self.local(gated) + fused
        return F.silu(self.mix(refined) + fused, inplace=True)


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
