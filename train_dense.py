"""Train the lightweight PRISM-based DenseDet model."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import amp
import torch.nn as nn
from torch.optim import AdamW

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from data.loader import DetectionAugmenter, build_train_loader, build_val_loader
from evaluate_dense import print_benchmark_comparison, run_dense_evaluation_with_raw
from model.dense_detector import DenseDet
from training.dense_loss import DenseDetectionLoss
from utils.detection_metrics import mean_metric
from utils.reporting import save_batch_preview, save_detection_artifacts, save_history_plot
from utils.runtime import coalesce, load_yaml_config, normalize_class_names, parse_int_tuple, require_existing_paths, resolve_detection_paths


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "dense_det.yaml")


def config_section(config, key):
    value = config.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_epochs: int, warmup_epochs: int = 0, min_lr_ratio: float = 0.05, last_epoch: int = 0):
        self.optimizer = optimizer
        self.total_epochs = max(int(total_epochs), 1)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lrs = [lr * self.min_lr_ratio for lr in self.base_lrs]
        self.last_epoch = int(last_epoch)
        self._set(self._compute(self.last_epoch))

    def _compute(self, epoch_index: int) -> list[float]:
        epoch_index = max(epoch_index, 0)
        out = []
        for base_lr, min_lr in zip(self.base_lrs, self.min_lrs):
            if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
                out.append(base_lr * float(epoch_index + 1) / float(self.warmup_epochs))
                continue
            if self.total_epochs <= self.warmup_epochs:
                out.append(base_lr)
                continue
            progress = (epoch_index - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            out.append(min_lr + (base_lr - min_lr) * cosine)
        return out

    def _set(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def step(self) -> None:
        self.last_epoch += 1
        self._set(self._compute(self.last_epoch))

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"total_epochs": self.total_epochs, "warmup_epochs": self.warmup_epochs, "min_lr_ratio": self.min_lr_ratio, "base_lrs": self.base_lrs, "min_lrs": self.min_lrs, "last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        self.total_epochs = int(state_dict.get("total_epochs", self.total_epochs))
        self.warmup_epochs = int(state_dict.get("warmup_epochs", self.warmup_epochs))
        self.min_lr_ratio = float(state_dict.get("min_lr_ratio", self.min_lr_ratio))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self.min_lrs = list(state_dict.get("min_lrs", self.min_lrs))
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))
        self._set(self._compute(self.last_epoch))


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = float(decay)
        self.ema = deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for key, ema_value in self.ema.state_dict().items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.ema.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.ema.load_state_dict(state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the lightweight DenseDet model")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--train_images", type=str, default=None)
    parser.add_argument("--val_images", type=str, default=None)
    parser.add_argument("--train_labels", type=str, default=None)
    parser.add_argument("--val_labels", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_every_epochs", type=int, default=None)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--eval_conf", type=float, default=None)
    parser.add_argument("--monitor_conf", type=float, default=None)
    parser.add_argument("--eval_iou", type=float, default=None)
    parser.add_argument("--eval_nms_iou", type=float, default=None)
    parser.add_argument("--checkpoint_metric", type=str, default=None, choices=["map50_95", "map50", "val_loss"])
    parser.add_argument("--balanced_sampler", dest="balanced_sampler", action="store_true")
    parser.add_argument("--no_balanced_sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--ema", dest="use_ema", action="store_true")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--mixed_precision", dest="use_mixed_precision", action="store_true")
    parser.add_argument("--no_mixed_precision", dest="use_mixed_precision", action="store_false")
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--backbone_dims", type=str, default=None)
    parser.add_argument("--backbone_depths", type=str, default=None)
    parser.add_argument("--head_depth", type=int, default=None)
    parser.add_argument("--quality_head", dest="use_quality_head", action="store_true")
    parser.add_argument("--no_quality_head", dest="use_quality_head", action="store_false")
    parser.set_defaults(balanced_sampler=None, augment=None, use_ema=None, use_mixed_precision=None, use_quality_head=None)
    return parser.parse_args()


def resolve_args(args):
    config = load_yaml_config(args.config)
    model_cfg = config_section(config, "model")
    data_cfg = config_section(config, "data")
    train_cfg = config_section(config, "train")
    optimizer_cfg = config_section(config, "optimizer")
    scheduler_cfg = config_section(config, "scheduler")
    checkpoint_cfg = config_section(config, "checkpoint")
    eval_cfg = config_section(config, "eval")
    loss_cfg = config_section(config, "loss")
    augmentation_cfg = config_section(config, "augmentation")
    benchmark_cfg = config_section(config, "benchmark")

    train_paths = resolve_detection_paths(args.data_config, args.dataset_root, "train", args.train_images, args.train_labels)
    val_paths = resolve_detection_paths(args.data_config, train_paths["dataset_root"], "val", args.val_images, args.val_labels)
    class_names = coalesce(train_paths["class_names"], normalize_class_names(data_cfg.get("class_names")))
    num_classes = coalesce(train_paths["num_classes"], data_cfg.get("num_classes"), len(class_names) if class_names else None, 6)

    resolved = {
        "config": args.config,
        "data_config": args.data_config,
        "dataset_root": train_paths["dataset_root"],
        "train_images": train_paths["images_dir"],
        "val_images": val_paths["images_dir"],
        "train_labels": train_paths["labels_dir"],
        "val_labels": val_paths["labels_dir"],
        "epochs": coalesce(args.epochs, train_cfg.get("epochs"), 45),
        "patience": coalesce(args.patience, train_cfg.get("patience"), 8),
        "batch": coalesce(args.batch, train_cfg.get("batch_size"), 8),
        "accumulation_steps": coalesce(args.accumulation_steps, train_cfg.get("accumulation_steps"), 2),
        "eval_batch": coalesce(eval_cfg.get("batch_size"), train_cfg.get("batch_size"), 8),
        "lr": coalesce(args.lr, optimizer_cfg.get("lr"), 4e-4),
        "weight_decay": coalesce(args.weight_decay, optimizer_cfg.get("weight_decay"), 0.01),
        "imgsz": coalesce(args.imgsz, train_cfg.get("image_size"), data_cfg.get("image_size"), 640),
        "workers": args.workers if args.workers is not None else coalesce(train_cfg.get("num_workers"), 4),
        "save_dir": coalesce(args.save_dir, checkpoint_cfg.get("save_dir"), "runs/dense_det"),
        "resume": args.resume,
        "eval_every_epochs": coalesce(args.eval_every_epochs, eval_cfg.get("during_train_every_epochs"), 5),
        "eval_max_batches": coalesce(args.eval_max_batches, eval_cfg.get("during_train_max_batches")),
        "eval_conf": coalesce(args.eval_conf, eval_cfg.get("conf_thresh"), 0.05),
        "eval_monitor_conf": coalesce(args.monitor_conf, eval_cfg.get("monitor_conf_thresh")),
        "eval_iou": coalesce(args.eval_iou, eval_cfg.get("iou_thresh"), 0.5),
        "eval_nms_iou": coalesce(args.eval_nms_iou, eval_cfg.get("nms_iou"), 0.6),
        "eval_max_det": coalesce(eval_cfg.get("max_det"), 300),
        "eval_use_ema": coalesce(eval_cfg.get("use_ema"), args.use_ema, train_cfg.get("use_ema"), True),
        "checkpoint_metric": coalesce(args.checkpoint_metric, eval_cfg.get("checkpoint_metric"), "map50_95"),
        "balanced_sampler": coalesce(args.balanced_sampler, train_cfg.get("balanced_sampler"), True),
        "augment": coalesce(args.augment, augmentation_cfg.get("enabled"), True),
        "close_augment_after_epochs": coalesce(augmentation_cfg.get("close_after_epochs"), 35),
        "augment_fliplr": coalesce(augmentation_cfg.get("fliplr"), 0.5),
        "augment_flipud": coalesce(augmentation_cfg.get("flipud"), 0.0),
        "augment_rotate_deg": coalesce(augmentation_cfg.get("rotate_deg"), 0.0),
        "augment_mosaic_prob": coalesce(augmentation_cfg.get("mosaic_prob"), 0.0),
        "augment_hsv_h": coalesce(augmentation_cfg.get("hsv_h"), 0.01),
        "augment_hsv_s": coalesce(augmentation_cfg.get("hsv_s"), 0.3),
        "augment_hsv_v": coalesce(augmentation_cfg.get("hsv_v"), 0.15),
        "augment_blur_prob": coalesce(augmentation_cfg.get("blur_prob"), 0.01),
        "augment_grayscale_prob": coalesce(augmentation_cfg.get("grayscale_prob"), 0.0),
        "augment_equalize_prob": coalesce(augmentation_cfg.get("equalize_prob"), 0.01),
        "augment_erasing_prob": coalesce(augmentation_cfg.get("erasing_prob"), 0.05),
        "warmup_epochs": coalesce(scheduler_cfg.get("warmup_epochs"), 5),
        "min_lr_ratio": coalesce(scheduler_cfg.get("eta_min_ratio"), 0.05),
        "use_ema": coalesce(args.use_ema, train_cfg.get("use_ema"), True),
        "ema_decay": coalesce(train_cfg.get("ema_decay"), 0.9998),
        "use_mixed_precision": coalesce(args.use_mixed_precision, train_cfg.get("mixed_precision"), True),
        "num_classes": num_classes,
        "class_names": class_names,
        "variant": coalesce(args.variant, model_cfg.get("variant"), "small"),
        "backbone_dims": parse_int_tuple(coalesce(args.backbone_dims, model_cfg.get("backbone_dims"), (16, 32, 64, 128)), "backbone_dims", 4),
        "backbone_depths": parse_int_tuple(coalesce(args.backbone_depths, model_cfg.get("backbone_depths"), (2, 2, 4, 2)), "backbone_depths", 4),
        "head_depth": coalesce(args.head_depth, model_cfg.get("head_depth"), 2),
        "use_quality_head": coalesce(args.use_quality_head, model_cfg.get("use_quality_head"), True),
        "assigner": coalesce(loss_cfg.get("assigner"), "atss"),
        "quality_loss_weight": coalesce(loss_cfg.get("quality"), 1.0),
        "benchmark": benchmark_cfg,
    }
    require_existing_paths(train_images=resolved["train_images"], val_images=resolved["val_images"], train_labels=resolved["train_labels"], val_labels=resolved["val_labels"])
    return argparse.Namespace(**resolved)


def build_model_config(args) -> dict:
    return {"num_classes": args.num_classes, "variant": args.variant, "backbone_dims": args.backbone_dims, "backbone_depths": args.backbone_depths, "head_depth": args.head_depth, "use_quality_head": args.use_quality_head}


def apply_checkpoint_model_config(args, checkpoint):
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        return args
    for key in ("num_classes", "class_names", "variant", "backbone_dims", "backbone_depths", "head_depth", "use_quality_head"):
        if key in model_config:
            setattr(args, key, model_config[key])
    return args


def resolve_checkpoint_alias(checkpoint_arg: str | None, save_dir: str) -> str | None:
    if checkpoint_arg is None:
        return None
    aliases = {"best": os.path.join(save_dir, "dense_det_best.pt"), "last": os.path.join(save_dir, "dense_det_last.pt"), "latest": os.path.join(save_dir, "dense_det_last.pt")}
    return aliases.get(checkpoint_arg, checkpoint_arg)


def append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_preview_from_loader(loader, class_names, path: str) -> None:
    try:
        images, targets, _ = next(iter(loader))
    except StopIteration:
        return
    save_batch_preview(images, targets, class_names, path)


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_dir, tag, model_config, best_val, best_epoch, epochs_without_improvement, best_checkpoint_score, train_loss=None, ema=None, ema_decay=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"dense_det_{tag}.pt")
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "model_config": model_config, "train_loss": train_loss, "val_loss": val_loss, "best_val": best_val, "best_epoch": best_epoch, "epochs_without_improvement": epochs_without_improvement, "best_checkpoint_score": best_checkpoint_score, "ema": None if ema is None else ema.state_dict(), "ema_decay": ema_decay}, path)
    return path


def format_metric(value, digits=4):
    return "n/a" if value == "" or value is None else f"{value:.{digits}f}"


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch, total_epochs=None, ema=None, accumulation_steps=1, use_mixed_precision=True):
    model.train()
    total_loss, processed_batches = 0.0, 0
    progress = tqdm(loader, total=len(loader), desc=f"Train {epoch}/{total_epochs or '?'}", dynamic_ncols=True, leave=True, disable=(tqdm is None or not sys.stdout.isatty())) if tqdm else None
    iterator = progress or loader
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, targets, _) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        window_start = (batch_index // accumulation_steps) * accumulation_steps
        window_end = min(window_start + accumulation_steps, len(loader))
        loss_scale = 1.0 / float(max(window_end - window_start, 1))
        with amp.autocast(device_type=device.type, enabled=(use_mixed_precision and device.type == "cuda")):
            outputs = model(images)
            losses = loss_fn(outputs, targets)
            loss = losses.total * loss_scale
        if device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if ((batch_index + 1) % accumulation_steps == 0) or ((batch_index + 1) == len(loader)):
            if device.type == "cuda":
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            step_successful = False
            if device.type == "cuda":
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_successful = scaler.get_scale() >= scale_before
            else:
                optimizer.step()
                step_successful = True
            if step_successful and ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        total_loss += losses.total.item()
        processed_batches += 1
        if progress is not None:
            progress.set_postfix({"loss": f"{total_loss / processed_batches:.4f}", "cls": f"{losses.cls.item():.3f}", "box": f"{losses.box.item():.3f}", "qual": f"{losses.qual.item():.3f}", "pos": losses.positives, "lr": f"{optimizer.param_groups[0]['lr']:.2e}", "acc": f"{((batch_index % accumulation_steps) + 1)}/{accumulation_steps}"})
    if progress is not None:
        progress.close()
    return total_loss / max(processed_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, use_mixed_precision=True):
    model.eval()
    total_loss = 0.0
    progress = tqdm(loader, total=len(loader), desc=f"Val {epoch}", dynamic_ncols=True, leave=True, disable=(tqdm is None or not sys.stdout.isatty())) if tqdm else None
    iterator = progress or loader
    for batch_index, (images, targets, _) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        with amp.autocast(device_type=device.type, enabled=(use_mixed_precision and device.type == "cuda")):
            outputs = model(images)
            losses = loss_fn(outputs, targets)
        total_loss += losses.total.item()
        if progress is not None:
            progress.set_postfix({"loss": f"{total_loss / (batch_index + 1):.4f}"})
    if progress is not None:
        progress.close()
    return total_loss / max(len(loader), 1)


def main():
    args = resolve_args(parse_args())
    args.resume = resolve_checkpoint_alias(args.resume, args.save_dir)
    resume_ckpt = None
    if args.resume:
        require_existing_paths(resume=args.resume)
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        args = apply_checkpoint_model_config(args, resume_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print("Architecture : PRISM + CAFPN + DenseHead")
    print(f"Variant      : {args.variant}")
    print(f"Backbone cfg : dims={args.backbone_dims}, depths={args.backbone_depths}")
    print(f"Quality head : {args.use_quality_head}")
    print(f"Epochs       : {args.epochs}  patience={args.patience}")
    print(f"Batch        : {args.batch}   accum={args.accumulation_steps}   effective={args.batch * args.accumulation_steps}")
    print(f"LR / WD      : {args.lr} / {args.weight_decay}")
    print(f"Image size   : {args.imgsz}")
    print(f"Workers      : {args.workers}")
    print(f"Classes      : {args.num_classes}")
    print(f"Augment      : {args.augment}  close_after={args.close_augment_after_epochs}")
    print(f"EMA          : {args.use_ema}  decay={args.ema_decay}")
    print(f"AMP          : {args.use_mixed_precision}")
    print(f"Assigner     : {args.assigner}")
    print(f"Ckpt metric  : {args.checkpoint_metric}")
    if args.eval_monitor_conf is not None:
        print(f"Monitor eval : conf={args.eval_monitor_conf}")

    print("\nBuilding DenseDet...")
    model_config = build_model_config(args)
    model = DenseDet(**model_config).to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    counts = model.param_count()
    print(f"  Total params     : {counts['total']:,}")
    print(f"  Trainable params : {counts['trainable']:,}")

    print("\nBuilding data loaders...")
    augmenter = DetectionAugmenter(
        enabled=args.augment,
        close_after_epochs=args.close_augment_after_epochs,
        fliplr=args.augment_fliplr,
        flipud=args.augment_flipud,
        rotate_deg=args.augment_rotate_deg,
        mosaic_prob=args.augment_mosaic_prob,
        hsv_h=args.augment_hsv_h,
        hsv_s=args.augment_hsv_s,
        hsv_v=args.augment_hsv_v,
        blur_prob=args.augment_blur_prob,
        grayscale_prob=args.augment_grayscale_prob,
        equalize_prob=args.augment_equalize_prob,
        erasing_prob=args.augment_erasing_prob,
        image_size=args.imgsz,
    )
    train_loader = build_train_loader(args.train_images, args.train_labels, batch_size=args.batch, image_size=args.imgsz, num_workers=args.workers, balanced=args.balanced_sampler, class_names=args.class_names, augmenter=augmenter)
    val_loader = build_val_loader(args.val_images, args.val_labels, batch_size=args.eval_batch, image_size=args.imgsz, num_workers=args.workers, class_names=args.class_names)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    loss_fn = DenseDetectionLoss(num_classes=args.num_classes, strides=model.strides, assigner=args.assigner, quality_loss_weight=args.quality_loss_weight)
    param_groups = build_param_groups(model, args.weight_decay)
    print(f"\nOptimizer param groups: {len(param_groups[0]['params'])} decay, {len(param_groups[1]['params'])} no-decay")
    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, total_epochs=args.epochs, warmup_epochs=args.warmup_epochs, min_lr_ratio=args.min_lr_ratio)
    scaler = amp.GradScaler(device.type, enabled=(args.use_mixed_precision and device.type == "cuda"))

    start_epoch, best_val, best_epoch, best_checkpoint_score, epochs_without_improvement = 1, float("inf"), 0, float("-inf"), 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        if ema is not None:
            ema_state = resume_ckpt.get("ema")
            ema.load_state_dict(ema_state if ema_state is not None else model.state_dict())
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best_val = float(resume_ckpt.get("best_val", resume_ckpt.get("val_loss", float("inf"))))
        best_epoch = int(resume_ckpt.get("best_epoch", 0))
        best_checkpoint_score = float(resume_ckpt.get("best_checkpoint_score", float("-inf")))
        epochs_without_improvement = int(resume_ckpt.get("epochs_without_improvement", 0))
        print(f"\nResumed from epoch {resume_ckpt['epoch']}")

    print(f"\nStarting training for {args.epochs} epochs...")
    os.makedirs(args.save_dir, exist_ok=True)
    history_path = os.path.join(args.save_dir, "train_history.csv")
    results_csv_path = os.path.join(args.save_dir, "results.csv")
    fields = ["epoch", "train_loss", "val_loss", "lr", "elapsed_sec", "best_val", "best_epoch", "epochs_without_improvement", "map50", "map5095", "macro_precision", "macro_recall", "micro_precision", "micro_recall"]
    print(f"{'Epoch':<10} {'Train':>10} {'Val':>10} {'Prec':>8} {'Recall':>8} {'mAP50':>10} {'mAP50-95':>10} {'Sec':>8}")

    save_preview_from_loader(train_loader, args.class_names, os.path.join(args.save_dir, "train_batch.jpg"))
    save_preview_from_loader(val_loader, args.class_names, os.path.join(args.save_dir, "val_batch_labels.jpg"))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if hasattr(train_loader.dataset, "augmenter") and train_loader.dataset.augmenter is not None:
            train_loader.dataset.augmenter.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, epoch, total_epochs=args.epochs, ema=ema, accumulation_steps=args.accumulation_steps, use_mixed_precision=args.use_mixed_precision)
        eval_model = (
            ema.ema
            if args.eval_use_ema and ema is not None and epoch > args.warmup_epochs
            else model
        )
        val_loss = validate(eval_model, val_loader, loss_fn, device, epoch, use_mixed_precision=args.use_mixed_precision)
        map50 = map5095 = ""
        summary = None
        all_preds = all_targets = None
        if args.eval_every_epochs and epoch % args.eval_every_epochs == 0:
            ap50, ap5095, pr_metrics, summary, all_preds, all_targets = run_dense_evaluation_with_raw(
                eval_model,
                val_loader,
                device,
                num_classes=args.num_classes,
                conf_thresh=args.eval_conf,
                match_iou=args.eval_iou,
                nms_iou=args.eval_nms_iou,
                max_batches=args.eval_max_batches,
                max_det=args.eval_max_det,
                verbose=False,
                progress_label=f"Eval {epoch}/{args.epochs}",
            )
            map50 = mean_metric(list(ap50.values()))
            map5095 = mean_metric(list(ap5095.values()))
            print_benchmark_comparison(ap50, ap5095, pr_metrics, summary, args.benchmark, args.class_names)
            if (
                args.eval_monitor_conf is not None
                and float(args.eval_monitor_conf) > float(args.eval_conf)
            ):
                mon_ap50, mon_ap5095, _, mon_summary, _, _ = run_dense_evaluation_with_raw(
                    eval_model,
                    val_loader,
                    device,
                    num_classes=args.num_classes,
                    conf_thresh=float(args.eval_monitor_conf),
                    match_iou=args.eval_iou,
                    nms_iou=args.eval_nms_iou,
                    max_batches=args.eval_max_batches,
                    max_det=args.eval_max_det,
                    verbose=False,
                    progress_label=f"Monitor {epoch}/{args.epochs}",
                )
                print(
                    "  Monitor @ "
                    f"conf={float(args.eval_monitor_conf):.2f} | "
                    f"Prec={mon_summary['macro_precision']:.3f} "
                    f"Recall={mon_summary['macro_recall']:.3f} "
                    f"mAP50={mean_metric(list(mon_ap50.values())):.3f} "
                    f"mAP50-95={mean_metric(list(mon_ap5095.values())):.3f}"
                )
            if all_preds is not None and all_targets is not None:
                save_detection_artifacts(all_preds, all_targets, args.class_names, args.save_dir, iou_threshold=args.eval_iou)
        scheduler.step()
        elapsed = time.time() - t0
        if val_loss < best_val:
            best_val = val_loss
        checkpoint_score = -float(val_loss) if args.checkpoint_metric == "val_loss" else None
        if summary is not None:
            if args.checkpoint_metric == "map50_95" and map5095 != "":
                checkpoint_score = float(map5095)
            elif args.checkpoint_metric == "map50" and map50 != "":
                checkpoint_score = float(map50)
        improved = checkpoint_score is not None and checkpoint_score > best_checkpoint_score
        if improved:
            best_checkpoint_score, best_epoch, epochs_without_improvement = float(checkpoint_score), epoch, 0
        elif epoch > args.warmup_epochs and summary is not None:
            epochs_without_improvement += 1
        macro_precision = None if summary is None else summary["macro_precision"]
        macro_recall = None if summary is None else summary["macro_recall"]
        print(f"{f'{epoch}/{args.epochs}':<10} {train_loss:>10.4f} {val_loss:>10.4f} {format_metric(macro_precision, 3):>8} {format_metric(macro_recall, 3):>8} {format_metric(map50, 3):>10} {format_metric(map5095, 3):>10} {elapsed:>8.0f}")
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0], "elapsed_sec": elapsed, "best_val": best_val, "best_epoch": best_epoch, "epochs_without_improvement": epochs_without_improvement, "map50": map50, "map5095": map5095, "macro_precision": "" if summary is None else summary["macro_precision"], "macro_recall": "" if summary is None else summary["macro_recall"], "micro_precision": "" if summary is None else summary["micro_precision"], "micro_recall": "" if summary is None else summary["micro_recall"]}
        append_csv_row(history_path, fields, row)
        append_csv_row(results_csv_path, fields, row)
        save_history_plot(results_csv_path, os.path.join(args.save_dir, "results.png"))
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args.save_dir, "last", model_config, best_val, best_epoch, epochs_without_improvement, best_checkpoint_score, train_loss=train_loss, ema=ema, ema_decay=args.ema_decay)
        if improved:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args.save_dir, "best", model_config, best_val, best_epoch, epochs_without_improvement, best_checkpoint_score, train_loss=train_loss, ema=ema, ema_decay=args.ema_decay)
            metric_label = "mAP50-95" if args.checkpoint_metric == "map50_95" else args.checkpoint_metric
            print(f"  New best checkpoint by {metric_label}: {best_checkpoint_score:.4f}")
        if args.patience and epoch > args.warmup_epochs and epochs_without_improvement >= args.patience:
            print(f"  Early stopping at epoch {epoch}: no improvement for {epochs_without_improvement} epochs (best epoch: {best_epoch}, best score: {best_checkpoint_score:.4f})")
            break

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"History:     {results_csv_path}")


if __name__ == "__main__":
    main()
