"""Optional training and inference efficiency helpers for DenseDet ablations."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn


def setup_t4_training(verbose: bool = True) -> dict[str, Any]:
    """Enable safe CUDA performance toggles for fixed-size detection training."""
    enabled: dict[str, Any] = {}
    if not torch.cuda.is_available():
        if verbose:
            print("  [efficiency] No CUDA detected; skipping CUDA-specific optimizations")
        return enabled

    device_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    enabled["tf32"] = True
    enabled["cudnn_benchmark"] = True
    enabled["cudnn_deterministic"] = False

    if verbose:
        print(f"  [efficiency] CUDA settings applied to: {device_name}")
        for key, value in enabled.items():
            print(f"    {key}: {value}")
    return enabled


def enable_channels_last(model: nn.Module, verbose: bool = True) -> nn.Module:
    model = model.to(memory_format=torch.channels_last)
    if verbose:
        print("  [efficiency] channels_last memory format enabled")
    return model


def to_channels_last(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.to(memory_format=torch.channels_last)
    return x


def compile_model(model: nn.Module, mode: str = "reduce-overhead", verbose: bool = True) -> nn.Module:
    if not hasattr(torch, "compile"):
        if verbose:
            print("  [efficiency] torch.compile unavailable; skipping")
        return model
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=False)
        if verbose:
            print(f"  [efficiency] torch.compile enabled (mode={mode})")
        return compiled
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(f"torch.compile failed: {exc}. Falling back to eager mode.")
        return model


def enable_backbone_gradient_checkpointing(model: nn.Module, verbose: bool = True) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        if verbose:
            print("  [efficiency] No backbone found; skipping gradient checkpointing")
        return

    if hasattr(backbone, "enable_gradient_checkpointing"):
        backbone.enable_gradient_checkpointing()
        return

    stages = getattr(backbone, "stages", None)
    if stages is None:
        if verbose:
            print("  [efficiency] backbone.stages not found; skipping gradient checkpointing")
        return

    from torch.utils.checkpoint import checkpoint

    for stage in stages:
        original_forward = stage.forward

        def make_gc_forward(fwd):
            def gc_forward(x):
                if torch.is_grad_enabled():
                    return checkpoint(fwd, x, use_reentrant=False)
                return fwd(x)

            return gc_forward

        stage.forward = make_gc_forward(original_forward)

    if verbose:
        print(
            f"  [efficiency] Gradient checkpointing enabled on {len(stages)} backbone stages"
        )


def _fuse_single_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    with torch.no_grad():
        weight = conv.weight.clone()
        bias = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels)

        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        std = (var + bn.eps).sqrt()
        scale = gamma / std

        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        fused.weight.data = weight * scale.view(-1, 1, 1, 1)
        fused.bias.data = (bias - mean) * scale + beta
        return fused


def _fuse_conv_bn_inplace(module: nn.Module) -> None:
    children = list(module.named_children())
    for index, (name, child) in enumerate(children):
        if (
            index + 1 < len(children)
            and isinstance(child, nn.Conv2d)
            and isinstance(children[index + 1][1], nn.BatchNorm2d)
        ):
            bn_name, bn = children[index + 1]
            setattr(module, name, _fuse_single_conv_bn(child, bn))
            setattr(module, bn_name, nn.Identity())
        else:
            _fuse_conv_bn_inplace(child)


def apply_training_efficiency(
    model: nn.Module,
    device: torch.device,
    use_channels_last: bool = False,
    use_compile: bool = False,
    use_gc: bool = False,
    compile_mode: str = "reduce-overhead",
    verbose: bool = True,
) -> nn.Module:
    if verbose:
        print("\n[efficiency] Applying training optimizations...")
    if device.type == "cuda":
        setup_t4_training(verbose=verbose)
    if use_gc:
        enable_backbone_gradient_checkpointing(model, verbose=verbose)
    if use_channels_last:
        model = enable_channels_last(model, verbose=verbose)
    if use_compile:
        model = compile_model(model, mode=compile_mode, verbose=verbose)
    if verbose:
        print("[efficiency] Done.\n")
    return model


def apply_inference_efficiency(
    model: nn.Module,
    device: torch.device,
    use_channels_last: bool = False,
    use_compile: bool = False,
    fuse_bn: bool = False,
    compile_mode: str = "reduce-overhead",
    verbose: bool = True,
) -> nn.Module:
    if verbose:
        print("\n[efficiency] Applying inference optimizations...")
    model.eval()
    if use_channels_last:
        model = enable_channels_last(model, verbose=verbose)
    if fuse_bn:
        _fuse_conv_bn_inplace(model)
        if verbose:
            print("  [efficiency] Conv+BN fusion applied")
    if use_compile:
        model = compile_model(model, mode=compile_mode, verbose=verbose)
    if verbose:
        print("[efficiency] Done.\n")
    return model
