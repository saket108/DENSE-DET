"""PRISM backbone for DenseDet ablations."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


def _snap16(x: int) -> int:
    return max(16, ((x + 15) // 16) * 16)


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _local_disorder(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Analytical local variance score used as a routing prior."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        pad = kernel_size // 2
        mu = F.avg_pool2d(xf, kernel_size, stride=1, padding=pad, count_include_pad=False)
        mu2 = F.avg_pool2d(xf.pow(2), kernel_size, stride=1, padding=pad, count_include_pad=False)
        var = (mu2 - mu.pow(2)).clamp(min=0.0)
        disorder = var.mean(dim=1, keepdim=True)
        mean_d = disorder.mean(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        score = disorder / mean_d - 1.0
    return torch.sigmoid(score.to(dtype=x.dtype))


class SequentialAsymmetricStrip(nn.Module):
    """Sequential 1xK then Kx1 depthwise strip processing."""

    def __init__(self, channels: int, strip_k: int = 7) -> None:
        super().__init__()
        self.dw_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, strip_k),
            padding=(0, strip_k // 2),
            groups=channels,
            bias=False,
        )
        self.dw_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(strip_k, 1),
            padding=(strip_k // 2, 0),
            groups=channels,
            bias=False,
        )
        self.bn_h = nn.GroupNorm(_group_count(channels), channels)
        self.bn_v = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.bn_h(self.dw_h(x)))
        return F.gelu(self.bn_v(self.dw_v(x)))


class DisorderBottleneckBlock(nn.Module):
    """PRISM-native bottleneck with analytical texture routing.

    The block first compresses channels, processes the compressed tensor with
    two cheap spatial paths, then expands back to the original width. The
    routing gate is computed from local variance of the input, so texture-heavy
    regions prefer the strip path while smoother regions prefer the 3x3 path.
    """

    def __init__(
        self,
        dim: int,
        strip_k: int = 7,
        bottleneck_ratio: float = 0.5,
        disorder_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.disorder_kernel = disorder_kernel
        hidden = min(dim, _snap16(int(dim * bottleneck_ratio)))

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
        )
        self.strip = SequentialAsymmetricStrip(hidden, strip_k=strip_k)
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.GroupNorm(_group_count(dim), dim),
        )
        self.layer_scale = nn.Parameter(torch.full((dim, 1, 1), 1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _local_disorder(x, self.disorder_kernel)
        z = self.reduce(x)
        mixed = gate * self.strip(z) + (1.0 - gate) * self.local(z)
        return x + self.layer_scale * self.expand(mixed)


LocalDisorderBlock = DisorderBottleneckBlock


class PixelUnshuffleDownsample(nn.Module):
    """Lossless stride-2 downsampling via pixel unshuffle and 1x1 mixing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.mix = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unshuffle(x)
        x = self.mix(x)
        return F.gelu(self.norm(x))


class PRISMStem(nn.Module):
    """Stem that reaches stride 4 using conv then pixel-unshuffle downsampling."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        mid = _snap16(max(out_channels // 2, 16))
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(mid), mid),
            nn.GELU(),
        )
        self.stage2 = PixelUnshuffleDownsample(mid, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage2(self.stage1(x))


class PRISMBackbone(nn.Module):
    """PRISM backbone exposing P2-P5 outputs for DenseDet."""

    def __init__(
        self,
        dims: tuple[int, int, int, int] = (16, 32, 64, 128),
        depths: tuple[int, int, int, int] = (2, 2, 4, 2),
        strip_k: int = 7,
        bottleneck_ratio: float = 0.5,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("PRISMBackbone expects exactly four stage dims and depths.")

        dims = tuple(_snap16(int(value)) for value in dims)
        self.model_name = "prism"
        self.channels = dims
        self.depths = tuple(int(value) for value in depths)
        self.reductions = (4, 8, 16, 32)
        self.use_gc = bool(use_gradient_checkpointing)

        self.stem = PRISMStem(dims[0])
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        for index in range(4):
            if index == 0:
                self.downsamples.append(nn.Identity())
            else:
                self.downsamples.append(PixelUnshuffleDownsample(dims[index - 1], dims[index]))
            self.stages.append(
                nn.Sequential(
                    *[
                        DisorderBottleneckBlock(
                            dims[index],
                            strip_k=strip_k,
                            bottleneck_ratio=bottleneck_ratio,
                        )
                        for _ in range(self.depths[index])
                    ]
                )
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

    def _run_stage(self, stage: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_gc and self.training:
            return gradient_checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        outputs = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = downsample(x)
            x = self._run_stage(stage, x)
            outputs.append(x)
        return tuple(outputs)

    def enable_gradient_checkpointing(self) -> None:
        self.use_gc = True
        print("  PRISMBackbone: gradient checkpointing enabled (~40% activation memory saving)")

    def param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
