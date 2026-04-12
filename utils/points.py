"""Point-grid helpers for dense detectors.

Changes vs original
-------------------
FIX-1  build_points is called once per FPN level per training step.  The result
       depends only on (feat_h, feat_w, stride) — not on the batch content.
       An LRU cache now returns the pre-built tensor on subsequent calls.

       Cache key includes device and dtype as strings so grids are rebuilt
       correctly if the model moves between CPU and GPU, or if dtype changes.

       The cache is bounded to 32 entries (8 levels × 4 dtype/device combos
       covers any realistic training configuration with room to spare).
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=32)
def build_points(
    feat_h: int,
    feat_w: int,
    stride: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return FCOS-style point centers for a feature map.

    Result is cached: identical (feat_h, feat_w, stride, device, dtype)
    arguments return the same tensor without recomputation.
    """
    shifts_x = (torch.arange(feat_w, device=device, dtype=dtype) + 0.5) * stride
    shifts_y = (torch.arange(feat_h, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).clone()
