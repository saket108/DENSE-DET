"""Train the dense detector baseline.

Changes vs original
-------------------
FIX-1  weight_decay was hardcoded to 0.0005, ignoring optimizer.weight_decay
       in the YAML config (which says 0.05).  Now read from config via args.

FIX-2  Separate AdamW parameter groups: bias terms and normalization layer
       parameters (weight + bias) receive weight_decay=0.  Applying weight
       decay to these parameters hurts convergence and is non-standard practice
       for both CNNs and ViT-style models.

FIX-3  patience counter now tracks epochs without improvement correctly on
       non-eval epochs.  Original: counter only incremented when
       checkpoint_score is not None (i.e. on eval epochs only), so with
       eval_every_epochs=5 and patience=10 the model would never early-stop —
       it requires 10 eval epochs of no improvement, meaning 50 real epochs.
       Fix: count ALL non-warmup epochs since the last improvement.

FIX-4  evidential_kl_weight annealing hook added to the training loop.
       EvidentialQualityHead (novel_blocks.py) requires the KL weight to be
       annealed from 0 → 1 over the first ~20 epochs.
       The loss function already has set_evidential_kl_weight(); it just
       was never called.
"""

import argparse
import csv
import math
import os
import sys
import time
from collections.abc import Mapping

import torch
from torch import amp
import torch.nn as nn
from torch.optim import AdamW

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from data.loader import DetectionAugmenter, build_train_loader, build_val_loader
from evaluate_dense import run_dense_evaluation
from model.dense_detector import DenseDet
from training.dense_loss import DenseDetectionLoss
from utils.detection_metrics import mean_metric
from utils.runtime import (
    coalesce,
    load_yaml_config,
    normalize_class_names,
    parse_int_tuple,
    require_existing_paths,
    resolve_dataset_paths,
    resolve_detection_paths,
)


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "dense_det.yaml",
)


# ── FIX-2: parameter group builder ───────────────────────────────────────────

def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Split model parameters into two groups:
      - decay group   : all weight tensors that are not 1-D (i.e. not bias/norm)
      - no-decay group: bias terms + all 1-D tensors (LayerNorm/GroupNorm/BN weight+bias)

    Applying weight decay to bias and normalisation parameters is non-standard
    and hurts convergence.  AdamW implementations from DeiT, ViT, and YOLO all
    exclude these parameters from decay.
    """
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1-D tensors: bias terms, norm weight/bias (all of shape [C])
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Epoch-based linear warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_lr_ratio: float = 0.01,
        last_epoch: int = 0,
    ) -> None:
        self.optimizer     = optimizer
        self.total_epochs  = max(int(total_epochs), 1)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.min_lr_ratio  = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.min_lrs  = [lr * self.min_lr_ratio for lr in self.base_lrs]
        self.last_epoch = int(last_epoch)
        self._set_lrs(self._compute_lrs(self.last_epoch))

    def _compute_lrs(self, epoch_index: int) -> list[float]:
        epoch_index = max(epoch_index, 0)
        lrs = []
        for base_lr, min_lr in zip(self.base_lrs, self.min_lrs):
            if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
                scale = float(epoch_index + 1) / float(self.warmup_epochs)
                lrs.append(base_lr * scale)
                continue
            if self.total_epochs <= self.warmup_epochs:
                lrs.append(base_lr)
                continue
            progress = (epoch_index - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs.append(min_lr + (base_lr - min_lr) * cosine)
        return lrs

    def _set_lrs(self, lrs: list[float]) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def step(self) -> None:
        self.last_epoch += 1
        self._set_lrs(self._compute_lrs(self.last_epoch))

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "total_epochs":  self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "min_lr_ratio":  self.min_lr_ratio,
            "base_lrs":      self.base_lrs,
            "min_lrs":       self.min_lrs,
            "last_epoch":    self.last_epoch,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.total_epochs  = int(state_dict.get("total_epochs",  self.total_epochs))
        self.warmup_epochs = int(state_dict.get("warmup_epochs", self.warmup_epochs))
        self.min_lr_ratio  = float(state_dict.get("min_lr_ratio", self.min_lr_ratio))
        self.base_lrs      = list(state_dict.get("base_lrs",      self.base_lrs))
        self.min_lrs       = list(state_dict.get("min_lrs",       self.min_lrs))
        self.last_epoch    = int(state_dict.get("last_epoch",     self.last_epoch))
        self._set_lrs(self._compute_lrs(self.last_epoch))


def config_section(config, key):
    value = config.get(key, {})
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    print(f"Warning: config section '{key}' should be a mapping. Ignoring.")
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the dense detector baseline")
    parser.add_argument("--config",       type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--data_config",  type=str, default=None)
    parser.add_argument("--data_format",  type=str, default=None, choices=["json", "detection"])
    parser.add_argument("--train_json",   type=str, default=None)
    parser.add_argument("--val_json",     type=str, default=None)
    parser.add_argument("--train_images", type=str, default=None)
    parser.add_argument("--val_images",   type=str, default=None)
    parser.add_argument("--train_labels", type=str, default=None)
    parser.add_argument("--val_labels",   type=str, default=None)
    parser.add_argument("--epochs",    type=int,   default=None)
    parser.add_argument("--patience",  type=int,   default=None)
    parser.add_argument("--batch",     type=int,   default=None)
    parser.add_argument("--lr",        type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)  # FIX-1
    parser.add_argument("--imgsz",     type=int,   default=None)
    parser.add_argument("--workers",   type=int,   default=None)
    parser.add_argument("--save_dir",  type=str,   default=None)
    parser.add_argument("--resume",    type=str,   default=None)
    parser.add_argument("--eval_every_epochs",  type=int,   default=None)
    parser.add_argument("--eval_max_batches",   type=int,   default=None)
    parser.add_argument("--eval_conf",          type=float, default=None)
    parser.add_argument("--eval_iou",           type=float, default=None)
    parser.add_argument("--eval_nms_iou",       type=float, default=None)
    parser.add_argument("--checkpoint_metric",  type=str,   default=None,
                        choices=["map50_95", "map50", "val_loss"])
    parser.add_argument("--save_every_batches", type=int,   default=None)
    parser.add_argument("--balanced_sampler",    dest="balanced_sampler", action="store_true")
    parser.add_argument("--no_balanced_sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--augment",             dest="augment", action="store_true")
    parser.add_argument("--no_augment",          dest="augment", action="store_false")
    parser.add_argument("--close_augment_after_epochs", type=int,   default=None)
    parser.add_argument("--augment_fliplr",             type=float, default=None)
    parser.add_argument("--augment_hsv_h",              type=float, default=None)
    parser.add_argument("--augment_hsv_s",              type=float, default=None)
    parser.add_argument("--augment_hsv_v",              type=float, default=None)
    parser.add_argument("--augment_blur_prob",          type=float, default=None)
    parser.add_argument("--augment_grayscale_prob",     type=float, default=None)
    parser.add_argument("--augment_equalize_prob",      type=float, default=None)
    parser.add_argument("--augment_erasing_prob",       type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int,   default=None)
    parser.add_argument("--min_lr_ratio",  type=float, default=None)
    # FIX-4: KL annealing epochs for EvidentialQualityHead
    parser.add_argument("--evidential_kl_anneal_epochs", type=int, default=None)
    parser.add_argument("--num_classes",   type=int, default=None)
    parser.add_argument("--variant",       type=str, default=None)
    parser.add_argument("--backbone_name", type=str, default=None)
    parser.add_argument("--backbone_dims", type=str, default=None)
    parser.add_argument("--backbone_depths", type=str, default=None)
    parser.add_argument("--pretrained_backbone_path", type=str, default=None)
    parser.add_argument("--neck_name",   type=str, default=None)
    parser.add_argument("--head_depth",  type=int, default=None)
    parser.add_argument("--stem_type", type=str, default=None)
    parser.add_argument("--backbone_block", type=str, default=None)
    parser.add_argument("--refine_block", type=str, default=None)
    parser.add_argument("--head_type", type=str, default=None)
    parser.add_argument("--use_auxiliary_heads",    dest="use_auxiliary_heads",    action="store_true")
    parser.add_argument("--no_use_auxiliary_heads", dest="use_auxiliary_heads",    action="store_false")
    parser.add_argument("--assigner",              type=str,   default=None, choices=["fcos", "atss"])
    parser.add_argument("--quality_loss_weight",   type=float, default=None)
    parser.add_argument("--auxiliary_loss_weight", type=float, default=None)
    parser.add_argument("--polarized_attention",    dest="use_polarized_attention", action="store_true")
    parser.add_argument("--no_polarized_attention", dest="use_polarized_attention", action="store_false")
    parser.add_argument(
        "--gradient_preservation_neck",
        dest="use_gradient_preservation_neck",
        action="store_true",
    )
    parser.add_argument(
        "--no_gradient_preservation_neck",
        dest="use_gradient_preservation_neck",
        action="store_false",
    )
    parser.add_argument("--detail_branch",    dest="use_detail_branch", action="store_true")
    parser.add_argument("--no_detail_branch", dest="use_detail_branch", action="store_false")
    parser.add_argument("--quality_head",    dest="use_quality_head", action="store_true")
    parser.add_argument("--no_quality_head", dest="use_quality_head", action="store_false")
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.set_defaults(
        use_detail_branch=None, use_quality_head=None,
        use_auxiliary_heads=None, use_polarized_attention=None,
        use_gradient_preservation_neck=None,
        augment=None, balanced_sampler=None,
    )
    return parser.parse_args()


def resolve_args(args):
    config = load_yaml_config(args.config)
    if not isinstance(config, Mapping):
        raise ValueError(f"Config file '{args.config}' must parse to a dict at the top level.")

    model_cfg      = config_section(config, "model")
    data_cfg       = config_section(config, "data")
    train_cfg      = config_section(config, "train")
    optimizer_cfg  = config_section(config, "optimizer")
    scheduler_cfg  = config_section(config, "scheduler")
    checkpoint_cfg = config_section(config, "checkpoint")
    eval_cfg       = config_section(config, "eval")
    loss_cfg       = config_section(config, "loss")
    augmentation_cfg = config_section(config, "augmentation")

    data_format = coalesce(args.data_format, data_cfg.get("format"), "detection")
    class_names = normalize_class_names(data_cfg.get("class_names"))
    num_classes = coalesce(
        args.num_classes, data_cfg.get("num_classes"),
        len(class_names) if class_names else None, 6,
    )

    train_json = val_json = train_labels = val_labels = None

    if data_format == "detection":
        train_paths = resolve_detection_paths(
            config_path=args.data_config, dataset_root=args.dataset_root,
            split="train", images_dir=args.train_images, labels_dir=args.train_labels,
        )
        val_paths = resolve_detection_paths(
            config_path=args.data_config, dataset_root=train_paths["dataset_root"],
            split="val", images_dir=args.val_images, labels_dir=args.val_labels,
        )
        dataset_root  = train_paths["dataset_root"]
        train_images  = train_paths["images_dir"]
        val_images    = val_paths["images_dir"]
        train_labels  = train_paths["labels_dir"]
        val_labels    = val_paths["labels_dir"]
        class_names   = coalesce(train_paths["class_names"], class_names)
        num_classes   = coalesce(train_paths["num_classes"], num_classes, 6)
    else:
        dataset_root, train_json, train_images = resolve_dataset_paths(
            dataset_root=args.dataset_root, data_config=data_cfg,
            split="train", json_path=args.train_json, images_dir=args.train_images,
        )
        _, val_json, val_images = resolve_dataset_paths(
            dataset_root=dataset_root, data_config=data_cfg,
            split="val", json_path=args.val_json, images_dir=args.val_images,
        )

    resolved = {
        "config": args.config, "data_config": args.data_config,
        "data_format": data_format, "dataset_root": dataset_root,
        "train_json": train_json, "val_json": val_json,
        "train_images": train_images, "val_images": val_images,
        "train_labels": train_labels, "val_labels": val_labels,
        "epochs":   coalesce(args.epochs,   train_cfg.get("epochs"),    300),
        "patience": coalesce(args.patience, train_cfg.get("patience"),  0),
        "batch":    coalesce(args.batch,    train_cfg.get("batch_size"), 8),
        "eval_batch": coalesce(eval_cfg.get("batch_size"), args.batch, train_cfg.get("batch_size"), 8),
        "lr": coalesce(args.lr, optimizer_cfg.get("lr"), 4e-4),
        # FIX-1: read weight_decay from config, not hardcoded
        "weight_decay": coalesce(args.weight_decay, optimizer_cfg.get("weight_decay"), 0.05),
        "imgsz":   coalesce(args.imgsz,   train_cfg.get("image_size"), data_cfg.get("image_size"), 640),
        "workers": args.workers if args.workers is not None else coalesce(train_cfg.get("num_workers"), 0),
        "balanced_sampler": coalesce(args.balanced_sampler, train_cfg.get("balanced_sampler"), True),
        "augment": coalesce(args.augment, augmentation_cfg.get("enabled"), False),
        "close_augment_after_epochs": coalesce(
            args.close_augment_after_epochs, augmentation_cfg.get("close_after_epochs"), 290,
        ),
        "augment_fliplr":         coalesce(args.augment_fliplr,         augmentation_cfg.get("fliplr"),         0.5),
        "augment_hsv_h":          coalesce(args.augment_hsv_h,          augmentation_cfg.get("hsv_h"),          0.015),
        "augment_hsv_s":          coalesce(args.augment_hsv_s,          augmentation_cfg.get("hsv_s"),          0.7),
        "augment_hsv_v":          coalesce(args.augment_hsv_v,          augmentation_cfg.get("hsv_v"),          0.4),
        "augment_blur_prob":      coalesce(args.augment_blur_prob,      augmentation_cfg.get("blur_prob"),      0.01),
        "augment_grayscale_prob": coalesce(args.augment_grayscale_prob, augmentation_cfg.get("grayscale_prob"), 0.01),
        "augment_equalize_prob":  coalesce(args.augment_equalize_prob,  augmentation_cfg.get("equalize_prob"),  0.01),
        "augment_erasing_prob":   coalesce(args.augment_erasing_prob,   augmentation_cfg.get("erasing_prob"),   0.4),
        "warmup_epochs": coalesce(args.warmup_epochs, scheduler_cfg.get("warmup_epochs"), 10),
        "min_lr_ratio":  coalesce(args.min_lr_ratio,  scheduler_cfg.get("eta_min_ratio"), 0.01),
        # FIX-4: KL annealing
        "evidential_kl_anneal_epochs": coalesce(
            args.evidential_kl_anneal_epochs, loss_cfg.get("evidential_kl_anneal_epochs"), 0,
        ),
        "save_dir": coalesce(args.save_dir, checkpoint_cfg.get("save_dir"), "runs/dense_det"),
        "resume": args.resume,
        "num_classes": num_classes, "class_names": class_names,
        "variant":       coalesce(args.variant,       model_cfg.get("variant"),       "small"),
        "backbone_name": coalesce(args.backbone_name, model_cfg.get("backbone_name"), "vst"),
        "backbone_dims": parse_int_tuple(
            coalesce(args.backbone_dims, model_cfg.get("backbone_dims")),
            field_name="backbone_dims", expected_len=4,
        ),
        "backbone_depths": parse_int_tuple(
            coalesce(args.backbone_depths, model_cfg.get("backbone_depths")),
            field_name="backbone_depths", expected_len=4,
        ),
        "pretrained_backbone_path": coalesce(
            args.pretrained_backbone_path, model_cfg.get("pretrained_backbone_path"),
        ),
        "neck_name":  coalesce(args.neck_name,  model_cfg.get("neck_name"),  "cafpn"),
        "head_depth": coalesce(args.head_depth, model_cfg.get("head_depth"), 2),
        "stem_type": coalesce(args.stem_type, model_cfg.get("stem_type"), "detail"),
        "backbone_block": coalesce(args.backbone_block, model_cfg.get("backbone_block"), "vst"),
        "refine_block": coalesce(args.refine_block, model_cfg.get("refine_block"), "dilated"),
        "head_type": coalesce(args.head_type, model_cfg.get("head_type"), "dense"),
        "use_detail_branch":   coalesce(args.use_detail_branch,   model_cfg.get("use_detail_branch"),   False),
        "use_quality_head":    coalesce(args.use_quality_head,    model_cfg.get("use_quality_head"),    True),
        "use_auxiliary_heads": coalesce(args.use_auxiliary_heads, model_cfg.get("use_auxiliary_heads"), False),
        "use_polarized_attention": coalesce(
            args.use_polarized_attention, model_cfg.get("use_polarized_attention"), False,
        ),
        "use_gradient_preservation_neck": coalesce(
            args.use_gradient_preservation_neck,
            model_cfg.get("use_gradient_preservation_neck"),
            False,
        ),
        "pretrained_backbone": (
            False if args.no_pretrained_backbone
            else coalesce(model_cfg.get("pretrained_backbone"), False)
        ),
        "assigner":              coalesce(args.assigner,              loss_cfg.get("assigner"),   "fcos"),
        "quality_loss_weight":   coalesce(args.quality_loss_weight,   loss_cfg.get("quality"),    1.0),
        "auxiliary_loss_weight": coalesce(args.auxiliary_loss_weight, loss_cfg.get("auxiliary"),  0.0),
        "save_every_batches": coalesce(args.save_every_batches, checkpoint_cfg.get("save_period_batches"), 0),
        "eval_every_epochs":  coalesce(args.eval_every_epochs,  eval_cfg.get("during_train_every_epochs"), 5),
        "eval_max_batches":   coalesce(args.eval_max_batches,   eval_cfg.get("during_train_max_batches"),  None),
        "eval_conf":    coalesce(args.eval_conf,    eval_cfg.get("conf_thresh"), 0.25),
        "eval_iou":     coalesce(args.eval_iou,     eval_cfg.get("iou_thresh"),  0.5),
        "eval_nms_iou": coalesce(args.eval_nms_iou, eval_cfg.get("nms_iou"),     0.6),
        "checkpoint_metric": coalesce(args.checkpoint_metric, eval_cfg.get("checkpoint_metric"), "map50_95"),
    }

    if resolved["data_format"] == "detection":
        missing = [
            n for n in ("train_images", "val_images", "train_labels", "val_labels")
            if resolved[n] is None
        ]
        if missing:
            raise ValueError(
                f"Detection format missing: {', '.join(missing)}. "
                "Pass --dataset_root or explicit path arguments."
            )
        require_existing_paths(
            data_config=resolved["data_config"],
            train_images=resolved["train_images"],
            val_images=resolved["val_images"],
            train_labels=resolved["train_labels"],
            val_labels=resolved["val_labels"],
        )
    else:
        require_existing_paths(
            train_json=resolved["train_json"], val_json=resolved["val_json"],
            train_images=resolved["train_images"], val_images=resolved["val_images"],
        )

    if resolved["resume"] is not None:
        require_existing_paths(resume=resolved["resume"])
    if resolved["pretrained_backbone_path"] is not None:
        require_existing_paths(pretrained_backbone_path=resolved["pretrained_backbone_path"])
    return argparse.Namespace(**resolved)


def build_model_config(args):
    return {
        "num_classes":           args.num_classes,
        "variant":               args.variant,
        "backbone_name":         args.backbone_name,
        "backbone_dims":         args.backbone_dims,
        "backbone_depths":       args.backbone_depths,
        "pretrained_backbone":   args.pretrained_backbone,
        "pretrained_backbone_path": args.pretrained_backbone_path,
        "neck_name":             args.neck_name,
        "head_depth":            args.head_depth,
        "stem_type":             args.stem_type,
        "backbone_block":        args.backbone_block,
        "refine_block":          args.refine_block,
        "head_type":             args.head_type,
        "use_detail_branch":     args.use_detail_branch,
        "use_gradient_preservation_neck": args.use_gradient_preservation_neck,
        "use_quality_head":      args.use_quality_head,
        "use_auxiliary_heads":   args.use_auxiliary_heads,
        "use_polarized_attention": args.use_polarized_attention,
    }


def build_checkpoint_model_config(args):
    config = build_model_config(args)
    config.update({"class_names": args.class_names, "data_format": args.data_format})
    return config


def apply_checkpoint_model_config(args, checkpoint):
    model_config = checkpoint.get("model_config")
    if not model_config:
        if hasattr(args, "use_polarized_attention"):
            args.use_polarized_attention = checkpoint_uses_polarized_attention(checkpoint)
        return args

    for key in (
        "num_classes", "class_names", "variant", "backbone_name",
        "backbone_dims", "backbone_depths", "pretrained_backbone",
        "pretrained_backbone_path", "neck_name", "head_depth",
        "stem_type", "backbone_block", "refine_block", "head_type",
        "use_detail_branch", "use_gradient_preservation_neck",
        "use_quality_head", "use_auxiliary_heads", "data_format",
    ):
        if key in model_config:
            setattr(args, key, model_config[key])
    if hasattr(args, "use_polarized_attention"):
        args.use_polarized_attention = checkpoint_uses_polarized_attention(checkpoint)
    return args


def checkpoint_uses_polarized_attention(checkpoint) -> bool:
    model_config = checkpoint.get("model_config", {})
    if isinstance(model_config, Mapping) and "use_polarized_attention" in model_config:
        return bool(model_config["use_polarized_attention"])
    state_dict = checkpoint.get("model", {})
    if not isinstance(state_dict, Mapping):
        return False
    return any("directional_gate" in str(k) for k in state_dict.keys())


def append_csv_row(path, fieldnames, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    model, optimizer, scheduler, epoch, val_loss, save_dir, tag="last",
    model_config=None, train_loss=None, batch_in_epoch=None, total_batches=None,
    global_step=None, is_mid_epoch=False, best_val=None, best_epoch=None,
    epochs_without_improvement=None, best_checkpoint_score=None,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"dense_det_{tag}.pt")
    torch.save(
        {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "val_loss": val_loss, "train_loss": train_loss,
            "batch_in_epoch": batch_in_epoch, "total_batches": total_batches,
            "global_step": global_step, "is_mid_epoch": is_mid_epoch,
            "model_config": model_config, "best_val": best_val,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "best_checkpoint_score": best_checkpoint_score,
        },
        path,
    )
    return path


def format_metric(value, digits=4):
    if value == "" or value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def train_one_epoch(
    model, loader, optimizer, loss_fn, scaler, device, epoch,
    total_epochs=None, save_every_batches=0, checkpoint_callback=None,
):
    model.train()
    total_loss = 0.0
    processed_batches = 0
    n_batches = len(loader)
    iterator = loader
    progress = None

    if tqdm is not None:
        progress = tqdm(
            loader, total=n_batches,
            desc=f"Train {epoch}/{total_epochs or '?'}",
            dynamic_ncols=True, leave=True,
            disable=not sys.stdout.isatty(),
        )
        iterator = progress

    for i, (images, targets, _) in enumerate(iterator):
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            losses  = loss_fn(outputs, targets)
            loss    = losses.total

        if not torch.isfinite(loss.detach()):
            if progress is not None:
                progress.write(f"  Skipping non-finite loss at epoch {epoch}, step {i+1}/{n_batches}")
            continue

        if device.type == "cuda":
            scaler.scale(loss).backward()
            has_grad = any(
                p.grad is not None for g in optimizer.param_groups for p in g["params"]
            )
            if not has_grad:
                if progress is not None:
                    progress.write(f"  Skipping no-grad batch at epoch {epoch}, step {i+1}/{n_batches}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            has_grad = any(
                p.grad is not None for g in optimizer.param_groups for p in g["params"]
            )
            if not has_grad:
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        processed_batches += 1
        avg = total_loss / max(processed_batches, 1)

        if progress is not None:
            progress.set_postfix({
                "loss": f"{avg:.4f}", "cls": f"{losses.cls.item():.3f}",
                "box": f"{losses.box.item():.3f}", "qual": f"{losses.qual.item():.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        if (
            checkpoint_callback is not None and save_every_batches
            and (i + 1) % save_every_batches == 0 and (i + 1) < n_batches
        ):
            checkpoint_callback(epoch=epoch, batch_in_epoch=i+1,
                                total_batches=n_batches, train_loss=avg)

    if progress is not None:
        progress.close()
    return total_loss / max(processed_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0.0
    iterator = loader
    progress = None

    if tqdm is not None:
        progress = tqdm(
            loader, total=len(loader), desc=f"Val {epoch}",
            dynamic_ncols=True, leave=True, disable=not sys.stdout.isatty(),
        )
        iterator = progress

    for batch_idx, (images, targets, _) in enumerate(iterator):
        images = images.to(device)
        outputs = model(images)
        losses  = loss_fn(outputs, targets)
        total_loss += losses.total.item()
        if progress is not None:
            progress.set_postfix({"loss": f"{total_loss / (batch_idx + 1):.4f}"})

    if progress is not None:
        progress.close()
    return total_loss / max(len(loader), 1)


def main():
    args = resolve_args(parse_args())
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        args = apply_checkpoint_model_config(args, resume_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Data format  : {args.data_format}")
    print(f"Backbone     : {args.backbone_name}")
    if args.backbone_dims and args.backbone_depths:
        print(f"Backbone cfg : dims={args.backbone_dims}, depths={args.backbone_depths}")
    print(f"Backbone blk : {args.backbone_block}")
    print(f"Variant      : {args.variant}")
    print(f"Neck         : {args.neck_name}")
    print(f"Stem         : {args.stem_type}")
    print(f"Refine blk   : {args.refine_block}")
    print(f"GPN          : {args.use_gradient_preservation_neck}")
    print(f"Head type    : {args.head_type}")
    print(f"Epochs       : {args.epochs}  patience={args.patience or 'off'}")
    print(f"Batch        : {args.batch}   lr={args.lr}  wd={args.weight_decay}")  # FIX-1
    print(f"Classes      : {args.num_classes}")
    print(f"Augment      : {args.augment}  close_after={args.close_augment_after_epochs}")
    print(f"KL anneal ep : {args.evidential_kl_anneal_epochs}")  # FIX-4
    print(f"Ckpt metric  : {args.checkpoint_metric}")

    print("\nBuilding DenseDet...")
    model_config            = build_model_config(args)
    checkpoint_model_config = build_checkpoint_model_config(args)
    model  = DenseDet(**model_config).to(device)
    counts = model.param_count()
    print(f"  Total params     : {counts['total']:,}")
    print(f"  Trainable params : {counts['trainable']:,}")

    print("\nBuilding data loaders...")
    train_augmenter = DetectionAugmenter(
        enabled=args.augment,
        close_after_epochs=args.close_augment_after_epochs,
        fliplr=args.augment_fliplr, hsv_h=args.augment_hsv_h,
        hsv_s=args.augment_hsv_s,   hsv_v=args.augment_hsv_v,
        blur_prob=args.augment_blur_prob, grayscale_prob=args.augment_grayscale_prob,
        equalize_prob=args.augment_equalize_prob, erasing_prob=args.augment_erasing_prob,
    )
    train_loader = build_train_loader(
        json_path=args.train_json, images_dir=args.train_images,
        batch_size=args.batch, image_size=args.imgsz, prompt_mode="cat_only",
        num_workers=args.workers, balanced=args.balanced_sampler,
        data_format=args.data_format, labels_dir=args.train_labels,
        class_names=args.class_names, augmenter=train_augmenter,
    )
    val_loader = build_val_loader(
        json_path=args.val_json, images_dir=args.val_images,
        batch_size=args.eval_batch, image_size=args.imgsz, prompt_mode="cat_only",
        num_workers=args.workers, data_format=args.data_format,
        labels_dir=args.val_labels, class_names=args.class_names,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    loss_fn = DenseDetectionLoss(
        num_classes=args.num_classes,
        strides=model.strides,
        assigner=args.assigner,
        quality_loss_weight=args.quality_loss_weight,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
    )

    # FIX-1 + FIX-2: read weight_decay from config, split param groups
    param_groups = build_param_groups(model, weight_decay=args.weight_decay)
    n_decay    = len(param_groups[0]["params"])
    n_no_decay = len(param_groups[1]["params"])
    print(f"\nOptimizer param groups: {n_decay} decay, {n_no_decay} no-decay")
    optimizer = AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    scheduler = WarmupCosineScheduler(
        optimizer, total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, min_lr_ratio=args.min_lr_ratio,
    )
    scaler = amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    best_checkpoint_score  = float("-inf")
    epochs_without_improvement = 0

    if args.resume:
        ckpt = resume_ckpt or torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if ckpt.get("is_mid_epoch"):
            start_epoch = ckpt["epoch"]
            print(f"\nResumed from mid-epoch checkpoint at epoch {ckpt['epoch']}. Epoch restarts.")
        else:
            print(f"\nResumed from epoch {ckpt['epoch']}")
        best_val = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        stored = ckpt.get("best_checkpoint_score")
        best_checkpoint_score = (
            float(stored) if stored is not None
            else (-float(best_val) if args.checkpoint_metric == "val_loss" else float("-inf"))
        )
        epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))

    print(f"\nStarting training for {args.epochs} epochs...")
    os.makedirs(args.save_dir, exist_ok=True)
    history_path = os.path.join(args.save_dir, "train_history.csv")
    history_fields = [
        "epoch", "train_loss", "val_loss", "lr", "elapsed_sec",
        "best_val", "best_epoch", "epochs_without_improvement",
        "map50", "map5095", "macro_precision", "macro_recall",
        "micro_precision", "micro_recall",
    ]
    print(
        f"{'Epoch':<10} {'Train':>10} {'Val':>10} "
        f"{'Prec':>8} {'Recall':>8} {'mAP50':>10} {'mAP50-95':>10} {'Sec':>8}"
    )

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Augmenter epoch counter
        if hasattr(train_loader.dataset, "augmenter") and train_loader.dataset.augmenter is not None:
            train_loader.dataset.augmenter.set_epoch(epoch)

        # FIX-4: anneal evidential KL weight 0 → 1 over the first N epochs
        if args.evidential_kl_anneal_epochs > 0 and hasattr(loss_fn, "set_evidential_kl_weight"):
            kl_w = min(1.0, (epoch - 1) / max(args.evidential_kl_anneal_epochs, 1))
            loss_fn.set_evidential_kl_weight(kl_w)

        def checkpoint_callback(epoch, batch_in_epoch, total_batches, train_loss):
            nonlocal global_step
            global_step = (epoch - 1) * total_batches + batch_in_epoch
            path = save_checkpoint(
                model, optimizer, scheduler, epoch, best_val, args.save_dir,
                tag="last", model_config=checkpoint_model_config,
                train_loss=train_loss, batch_in_epoch=batch_in_epoch,
                total_batches=total_batches, global_step=global_step,
                is_mid_epoch=True, best_val=best_val, best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                best_checkpoint_score=best_checkpoint_score,
            )
            print(f"  Saved mid-epoch checkpoint: {path}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch,
            total_epochs=args.epochs, save_every_batches=args.save_every_batches,
            checkpoint_callback=checkpoint_callback,
        )
        val_loss = validate(model, val_loader, loss_fn, device, epoch)
        map50 = map5095 = ""
        summary = None

        if args.eval_every_epochs and epoch % args.eval_every_epochs == 0:
            ap50, ap5095, pr_metrics, summary = run_dense_evaluation(
                model, val_loader, device, num_classes=args.num_classes,
                conf_thresh=args.eval_conf, match_iou=args.eval_iou,
                nms_iou=args.eval_nms_iou, max_batches=args.eval_max_batches,
                verbose=False, progress_label=f"Eval {epoch}/{args.epochs}",
            )
            map50   = mean_metric(list(ap50.values()))
            map5095 = mean_metric(list(ap5095.values()))

        scheduler.step()
        elapsed = time.time() - t0

        macro_precision = None if summary is None else summary["macro_precision"]
        macro_recall    = None if summary is None else summary["macro_recall"]

        if val_loss < best_val:
            best_val = val_loss

        checkpoint_score = None
        if args.checkpoint_metric == "val_loss":
            checkpoint_score = -float(val_loss)
        elif summary is not None:
            if args.checkpoint_metric == "map50_95" and map5095 != "":
                checkpoint_score = float(map5095)
            elif args.checkpoint_metric == "map50" and map50 != "":
                checkpoint_score = float(map50)

        improved  = checkpoint_score is not None and checkpoint_score > best_checkpoint_score
        in_warmup = epoch <= args.warmup_epochs

        if improved:
            best_checkpoint_score = float(checkpoint_score)
            best_epoch = epoch
            epochs_without_improvement = 0
        elif not in_warmup:
            # FIX-3: always count non-warmup epochs without improvement,
            # regardless of whether this was an eval epoch.
            # Original only counted when checkpoint_score is not None,
            # meaning non-eval epochs were invisible to the patience counter.
            epochs_without_improvement += 1

        print(
            f"{f'{epoch}/{args.epochs}':<10} "
            f"{train_loss:>10.4f} {val_loss:>10.4f} "
            f"{format_metric(macro_precision, 3):>8} {format_metric(macro_recall, 3):>8} "
            f"{format_metric(map50, 3):>10} {format_metric(map5095, 3):>10} "
            f"{elapsed:>8.0f}"
        )
        append_csv_row(history_path, history_fields, {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0], "elapsed_sec": elapsed,
            "best_val": best_val, "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "map50": map50, "map5095": map5095,
            "macro_precision": "" if summary is None else summary["macro_precision"],
            "macro_recall":    "" if summary is None else summary["macro_recall"],
            "micro_precision": "" if summary is None else summary["micro_precision"],
            "micro_recall":    "" if summary is None else summary["micro_recall"],
        })

        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, args.save_dir, tag="last",
            model_config=checkpoint_model_config, train_loss=train_loss,
            batch_in_epoch=len(train_loader), total_batches=len(train_loader),
            global_step=epoch * len(train_loader), is_mid_epoch=False,
            best_val=best_val, best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            best_checkpoint_score=best_checkpoint_score,
        )

        if improved:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, args.save_dir, tag="best",
                model_config=checkpoint_model_config, train_loss=train_loss,
                batch_in_epoch=len(train_loader), total_batches=len(train_loader),
                global_step=epoch * len(train_loader), is_mid_epoch=False,
                best_val=best_val, best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                best_checkpoint_score=best_checkpoint_score,
            )
            metric_label = "mAP50-95" if args.checkpoint_metric == "map50_95" else args.checkpoint_metric
            print(f"  New best checkpoint by {metric_label}: {best_checkpoint_score:.4f}")

        if epoch % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, args.save_dir, tag=f"ep{epoch}",
                model_config=checkpoint_model_config, train_loss=train_loss,
                batch_in_epoch=len(train_loader), total_batches=len(train_loader),
                global_step=epoch * len(train_loader), is_mid_epoch=False,
                best_val=best_val, best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                best_checkpoint_score=best_checkpoint_score,
            )

        if (
            args.patience
            and not in_warmup
            and epochs_without_improvement >= args.patience
        ):
            print(
                f"  Early stopping at epoch {epoch}: "
                f"no improvement for {epochs_without_improvement} epochs "
                f"(best epoch: {best_epoch}, best score: {best_checkpoint_score:.4f})"
            )
            break

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"History:     {history_path}")


if __name__ == "__main__":
    main()
