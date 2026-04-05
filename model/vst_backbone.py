"""VST backbone for dense aircraft-damage detection experiments."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class VSTBlock(nn.Module):
    """Dual-branch spatial block with channel gating and residual scaling."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = dim * expansion
        gate_hidden = max(dim // 4, 8)

        self.local_dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.context_dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.branch_merge = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
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
        local = self.local_dw(x)
        context = self.context_dw(x)
        x = self.branch_merge(torch.cat([local, context], dim=1))
        x = x * self.channel_gate(x)
        x = self.act(self.expand(x))
        x = self.norm(self.project(x))
        return residual + self.layer_scale * x


class VSTStem(nn.Module):
    """Two-stage entry stem that preserves fine damage cues."""

    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        mid = out_channels // 2
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(mid), mid),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                mid,
                mid,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=mid,
                bias=False,
            ),
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage2(self.stage1(x))


class VSTDownsample(nn.Module):
    """Learnable stride-2 transition between VST stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class VSTBackbone(nn.Module):
    """Visual Spatial Two-branch backbone with P2-P5 outputs."""

    def __init__(
        self,
        dims: tuple[int, int, int, int] = (32, 64, 128, 256),
        depths: tuple[int, int, int, int] = (2, 2, 4, 2),
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("VSTBackbone expects exactly four stage dims and depths.")

        self.model_name = "vst"
        self.channels = tuple(int(value) for value in dims)
        self.depths = tuple(int(value) for value in depths)
        self.reductions = (4, 8, 16, 32)

        self.stem = VSTStem(in_channels=3, out_channels=dims[0])
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        for index in range(4):
            if index == 0:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(VSTDownsample(dims[index - 1], dims[index]))
            self.stages.append(
                nn.Sequential(*[VSTBlock(dims[index]) for _ in range(depths[index])])
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        outputs = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = downsample(x)
            x = stage(x)
            outputs.append(x)
        return tuple(outputs)


if __name__ == "__main__":
    model = VSTBackbone()
    dummy = torch.randn(2, 3, 640, 640)
    outputs = model(dummy)
    print("VST Backbone output shapes:")
    for name, feature in zip(("P2", "P3", "P4", "P5"), outputs):
        print(f"  {name}: {tuple(feature.shape)}")
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Total params: {total:,}")
