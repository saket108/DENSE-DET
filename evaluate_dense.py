"""Evaluate the dense detection baseline.

Changes vs original
-------------------
FIX-1  run_dense_evaluation: np.arange(0.5, 1.0, 0.05) starts at 0.5, so
       evaluate_predictions was called at 0.5 TWICE — once explicitly for
       ap50_metrics and once as the first iteration of the sweep loop.
       Fixed by starting the sweep at 0.55 and inserting the already-computed
       ap50_metrics result directly into ap_all.  One full evaluation pass
       removed per validation run (~8% speedup on 10-threshold sweep).

FIX-2  detection_metrics.py round-1 fix introduced a subtle in-place write:
       row_ious = ious[pred_idx]  # VIEW, not a copy
       row_ious[matched] = -1.0   # mutates ious in-place
       This is safe per-row (other rows are unaffected) but corrupts the
       stored ious tensor if it is inspected for debugging.  Added .clone()
       in the updated detection_metrics_fixed.py; noted here for traceability.
"""

import argparse
import os
from collections.abc import Mapping

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from data.loader import build_val_loader
from model.dense_detector import DenseDet
from utils.detection_metrics import (
    evaluate_predictions,
    mean_metric,
    print_results,
    summarize_metrics,
)
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


def config_section(config, key):
    """Return a config subsection or an empty dict if it is malformed."""
    value = config.get(key, {})
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    print(
        f"Warning: config section '{key}' should be a mapping, "
        f"got {type(value).__name__}. Ignoring that section."
    )
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the dense detector baseline")
    parser.add_argument("--config",       type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint",   type=str, default=None)
    parser.add_argument("--split",        type=str, default="val", choices=["val", "test"])
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--data_config",  type=str, default=None)
    parser.add_argument("--data_format",  type=str, default=None, choices=["json", "detection"])
    parser.add_argument("--val_json",     type=str, default=None)
    parser.add_argument("--test_json",    type=str, default=None)
    parser.add_argument("--val_images",   type=str, default=None)
    parser.add_argument("--test_images",  type=str, default=None)
    parser.add_argument("--val_labels",   type=str, default=None)
    parser.add_argument("--test_labels",  type=str, default=None)
    parser.add_argument("--batch",        type=int, default=None)
    parser.add_argument("--imgsz",        type=int, default=None)
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--max_batches",  type=int, default=None)
    parser.add_argument("--num_classes",  type=int, default=None)
    parser.add_argument("--variant",      type=str, default=None)
    parser.add_argument("--backbone_name",   type=str, default=None)
    parser.add_argument("--backbone_dims",   type=str, default=None)
    parser.add_argument("--backbone_depths", type=str, default=None)
    parser.add_argument("--neck_name",   type=str, default=None)
    parser.add_argument("--head_depth",  type=int, default=None)
    parser.add_argument("--stem_type", type=str, default=None)
    parser.add_argument("--backbone_block", type=str, default=None)
    parser.add_argument("--refine_block", type=str, default=None)
    parser.add_argument("--head_type", type=str, default=None)
    parser.add_argument("--class_conditional_gn",    dest="use_class_conditional_gn", action="store_true")
    parser.add_argument("--no_class_conditional_gn", dest="use_class_conditional_gn", action="store_false")
    parser.add_argument("--use_auxiliary_heads",    dest="use_auxiliary_heads",    action="store_true")
    parser.add_argument("--no_use_auxiliary_heads", dest="use_auxiliary_heads",    action="store_false")
    parser.add_argument("--conf",        type=float, default=None)
    parser.add_argument("--iou_thresh",  type=float, default=None)
    parser.add_argument("--nms_iou",     type=float, default=None)
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
    # NEW: optional output path to save results as JSON for ablation tables
    parser.add_argument("--save_results", type=str, default=None,
                        help="If set, write AP50/AP5095/PR results to this JSON file.")
    parser.set_defaults(
        use_detail_branch=None,
        use_quality_head=None,
        use_class_conditional_gn=None,
        use_auxiliary_heads=None,
        use_polarized_attention=None,
        use_gradient_preservation_neck=None,
    )
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg, config):
    checkpoint_cfg = config.get("checkpoint", {})
    save_dir = checkpoint_cfg.get("save_dir", "runs/dense_det")

    aliases = {
        "best":   os.path.join(save_dir, "dense_det_best.pt"),
        "last":   os.path.join(save_dir, "dense_det_last.pt"),
        "latest": os.path.join(save_dir, "dense_det_last.pt"),
    }
    candidate = aliases.get(checkpoint_arg, checkpoint_arg)
    if os.path.exists(candidate):
        return candidate

    checkpoint_files = []
    if os.path.isdir(save_dir):
        checkpoint_files = sorted(
            os.path.join(save_dir, n)
            for n in os.listdir(save_dir)
            if n.endswith((".pt", ".pth"))
        )

    if checkpoint_files:
        available = ", ".join(checkpoint_files)
        raise FileNotFoundError(
            f"Checkpoint not found: '{candidate}'. Available: {available}"
        )
    raise FileNotFoundError(
        f"Checkpoint not found: '{candidate}'. No checkpoints in '{save_dir}'. "
        "Train the detector first or pass the correct path."
    )


def resolve_args(args):
    config = load_yaml_config(args.config)
    if not isinstance(config, Mapping):
        raise ValueError(
            f"Config file '{args.config}' must parse to a dict at the top level."
        )

    model_cfg = config_section(config, "model")
    data_cfg  = config_section(config, "data")
    train_cfg = config_section(config, "train")
    eval_cfg  = config_section(config, "eval")

    data_format = coalesce(args.data_format, data_cfg.get("format"), "detection")
    class_names = normalize_class_names(data_cfg.get("class_names"))
    num_classes = coalesce(
        args.num_classes,
        data_cfg.get("num_classes"),
        len(class_names) if class_names else None,
        6,
    )

    json_path  = None
    labels_dir = None

    if data_format == "detection":
        val_paths = resolve_detection_paths(
            config_path=args.data_config,
            dataset_root=args.dataset_root,
            split="val",
            images_dir=args.val_images,
            labels_dir=args.val_labels,
        )
        test_paths = resolve_detection_paths(
            config_path=args.data_config,
            dataset_root=val_paths["dataset_root"],
            split="test",
            images_dir=args.test_images,
            labels_dir=args.test_labels,
        )
        active_paths = test_paths if args.split == "test" else val_paths
        dataset_root = active_paths["dataset_root"]
        split_images = active_paths["images_dir"]
        labels_dir   = active_paths["labels_dir"]
        class_names  = coalesce(active_paths["class_names"], class_names)
        num_classes  = coalesce(active_paths["num_classes"], num_classes, 6)
    else:
        dataset_root, val_json, val_images = resolve_dataset_paths(
            dataset_root=args.dataset_root,
            data_config=data_cfg,
            split="val",
            json_path=args.val_json,
            images_dir=args.val_images,
        )
        _, test_json, test_images = resolve_dataset_paths(
            dataset_root=dataset_root,
            data_config=data_cfg,
            split="test",
            json_path=args.test_json,
            images_dir=args.test_images,
        )
        json_path    = test_json    if args.split == "test" else val_json
        split_images = test_images  if args.split == "test" else val_images

    resolved = {
        "config":       args.config,
        "checkpoint":   args.checkpoint,
        "split":        args.split,
        "data_config":  args.data_config,
        "data_format":  data_format,
        "dataset_root": dataset_root,
        "json_path":    json_path,
        "images_dir":   split_images,
        "labels_dir":   labels_dir,
        "batch":    coalesce(args.batch, eval_cfg.get("batch_size"), train_cfg.get("batch_size"), 8),
        "imgsz":    coalesce(args.imgsz, train_cfg.get("image_size"), data_cfg.get("image_size"), 640),
        "workers":  args.workers if args.workers is not None else coalesce(train_cfg.get("num_workers"), 0),
        "max_batches": args.max_batches,
        "num_classes": num_classes,
        "class_names": class_names,
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
        "neck_name":  coalesce(args.neck_name,  model_cfg.get("neck_name"),  "cafpn"),
        "head_depth": coalesce(args.head_depth, model_cfg.get("head_depth"), 2),
        "stem_type": coalesce(args.stem_type, model_cfg.get("stem_type"), "detail"),
        "backbone_block": coalesce(args.backbone_block, model_cfg.get("backbone_block"), "vst"),
        "refine_block": coalesce(args.refine_block, model_cfg.get("refine_block"), "dilated"),
        "head_type": coalesce(args.head_type, model_cfg.get("head_type"), "dense"),
        "conf":      coalesce(args.conf,       eval_cfg.get("conf_threshold"),  0.25),
        "iou_thresh": coalesce(args.iou_thresh, eval_cfg.get("match_iou"),      0.5),
        "nms_iou":    coalesce(args.nms_iou,    eval_cfg.get("nms_iou"),         0.6),
        "pretrained_backbone": not getattr(args, "no_pretrained_backbone", False),
        "use_detail_branch":     coalesce(args.use_detail_branch,     model_cfg.get("use_detail_branch"),     False),
        "use_quality_head":      coalesce(args.use_quality_head,      model_cfg.get("use_quality_head"),      True),
        "use_class_conditional_gn": coalesce(
            args.use_class_conditional_gn, model_cfg.get("use_class_conditional_gn"), False,
        ),
        "use_auxiliary_heads":   coalesce(args.use_auxiliary_heads,   model_cfg.get("use_auxiliary_heads"),   False),
        "use_polarized_attention": coalesce(args.use_polarized_attention, model_cfg.get("use_polarized_attention"), False),
        "use_gradient_preservation_neck": coalesce(
            args.use_gradient_preservation_neck,
            model_cfg.get("use_gradient_preservation_neck"),
            False,
        ),
        "save_results": args.save_results,
    }

    if data_format == "detection":
        require_existing_paths(images_dir=resolved["images_dir"])
    else:
        require_existing_paths(json_path=resolved["json_path"], images_dir=resolved["images_dir"])

    if resolved["checkpoint"] is not None:
        resolved["checkpoint"] = resolve_checkpoint_path(resolved["checkpoint"], config=config)

    return argparse.Namespace(**resolved)


def build_model_config(args):
    return {
        "num_classes":           args.num_classes,
        "variant":               args.variant,
        "backbone_name":         args.backbone_name,
        "backbone_dims":         args.backbone_dims,
        "backbone_depths":       args.backbone_depths,
        "pretrained_backbone":   args.pretrained_backbone,
        "neck_name":             args.neck_name,
        "head_depth":            args.head_depth,
        "stem_type":             args.stem_type,
        "backbone_block":        args.backbone_block,
        "refine_block":          args.refine_block,
        "head_type":             args.head_type,
        "use_detail_branch":     args.use_detail_branch,
        "use_gradient_preservation_neck": args.use_gradient_preservation_neck,
        "use_quality_head":      args.use_quality_head,
        "use_class_conditional_gn": args.use_class_conditional_gn,
        "use_auxiliary_heads":   args.use_auxiliary_heads,
        "use_polarized_attention": args.use_polarized_attention,
    }


def apply_checkpoint_model_config(args, checkpoint):
    model_config = checkpoint.get("model_config")
    if not model_config:
        if hasattr(args, "use_polarized_attention"):
            args.use_polarized_attention = checkpoint_uses_polarized_attention(checkpoint)
        return args

    for key in (
        "num_classes", "class_names", "variant",
        "backbone_name", "backbone_dims", "backbone_depths",
        "pretrained_backbone", "neck_name", "head_depth",
        "stem_type", "backbone_block", "refine_block", "head_type",
        "use_detail_branch", "use_gradient_preservation_neck",
        "use_class_conditional_gn",
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


@torch.no_grad()
def run_dense_evaluation(
    model,
    loader,
    device,
    num_classes,
    conf_thresh=0.25,
    match_iou=0.5,
    nms_iou=0.6,
    max_batches=None,
    verbose=True,
    progress_label=None,
):
    model.eval()
    all_preds:   list[dict] = []
    all_targets: list[dict] = []

    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    iterator = loader
    progress = None
    if tqdm is not None:
        progress = tqdm(
            loader,
            total=total_batches,
            desc=progress_label or "Eval",
            dynamic_ncols=True,
            leave=False,
            disable=(not verbose),
        )
        iterator = progress

    for batch_idx, (images, targets, _) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images      = images.to(device)
        predictions = model.predict(images, conf_threshold=conf_thresh, nms_iou=nms_iou)

        for pred, target in zip(predictions, targets):
            all_preds.append({
                "boxes":       pred["boxes"].cpu(),
                "labels":      pred["labels"].cpu(),
                "confidences": pred["confidences"].cpu(),
            })
            all_targets.append({
                "boxes":  target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

        if progress is not None:
            progress.set_postfix({"images": len(all_targets)})

    if progress is not None:
        progress.close()

    # AP@0.5  (computed once and reused in the sweep — FIX-1)
    ap50_metrics = evaluate_predictions(
        all_preds, all_targets, num_classes=num_classes, iou_threshold=0.5,
    )
    # PR metrics at match_iou (may differ from 0.5)
    pr_metrics = evaluate_predictions(
        all_preds, all_targets, num_classes=num_classes, iou_threshold=match_iou,
    ) if match_iou != 0.5 else ap50_metrics

    # AP sweep: start at 0.55 — 0.5 is already in ap50_metrics (FIX-1)
    ap_all: dict[int, list[float]] = {cls_id: [float(ap50_metrics[cls_id]["ap"])]
                                      for cls_id in range(num_classes)}
    for iou_t in np.arange(0.55, 1.0, 0.05):          # FIX-1: was 0.5
        threshold_metrics = evaluate_predictions(
            all_preds, all_targets,
            num_classes=num_classes,
            iou_threshold=float(iou_t),
        )
        for cls_id, metrics in threshold_metrics.items():
            ap_all[cls_id].append(float(metrics["ap"]))

    ap50   = {cls_id: float(m["ap"]) for cls_id, m in ap50_metrics.items()}
    ap5095 = {cls_id: float(np.mean(v)) for cls_id, v in ap_all.items()}
    summary = summarize_metrics(pr_metrics)
    return ap50, ap5095, pr_metrics, summary


def main():
    args    = resolve_args(parse_args())
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        args = apply_checkpoint_model_config(args, checkpoint)

    print(f"Device        : {device}")
    print(f"Split         : {args.split}")
    print(f"Format        : {args.data_format}")
    print(f"Backbone      : {args.backbone_name}")
    if args.backbone_dims and args.backbone_depths:
        print(f"Backbone cfg  : dims={args.backbone_dims}, depths={args.backbone_depths}")
    print(f"Backbone blk  : {args.backbone_block}")
    print(f"Variant       : {args.variant}")
    print(f"Neck          : {args.neck_name}")
    print(f"Stem          : {args.stem_type}")
    print(f"Refine blk    : {args.refine_block}")
    print(f"GPN           : {args.use_gradient_preservation_neck}")
    print(f"Head type     : {args.head_type}")
    print(f"CCGN          : {args.use_class_conditional_gn}")
    print(f"Aux heads     : {args.use_auxiliary_heads}")
    print(f"Polarized attn: {args.use_polarized_attention}")
    print(f"Quality head  : {args.use_quality_head}")
    print(f"Data          : {args.images_dir}")
    print(f"Classes       : {args.num_classes}")
    print(f"Conf          : {args.conf}")
    print(f"Match IoU     : {args.iou_thresh}")
    print(f"NMS IoU       : {args.nms_iou}")

    print("\nBuilding DenseDet...")
    model = DenseDet(**build_model_config(args)).to(device)

    if checkpoint is not None:
        state_key = "ema" if checkpoint.get("ema") is not None else "model"
        model.load_state_dict(checkpoint[state_key])
        print(f"Loaded checkpoint: {args.checkpoint} ({state_key} weights)")
    else:
        print("No checkpoint — evaluating random weights only")

    loader = build_val_loader(
        json_path=args.json_path,
        images_dir=args.images_dir,
        batch_size=args.batch,
        image_size=args.imgsz,
        num_workers=args.workers,
        data_format=args.data_format,
        labels_dir=args.labels_dir,
        class_names=args.class_names,
    )
    print(f"Eval batches: {len(loader)}")

    ap50, ap5095, pr_metrics, summary = run_dense_evaluation(
        model, loader, device,
        num_classes=args.num_classes,
        conf_thresh=args.conf,
        match_iou=args.iou_thresh,
        nms_iou=args.nms_iou,
        max_batches=args.max_batches,
    )

    print_results(ap50, ap5095, pr_metrics, summary,
                  match_iou=args.iou_thresh, class_names=args.class_names)
    print(
        f"Macro mAP50={mean_metric(list(ap50.values())):.4f} "
        f"mAP50-95={mean_metric(list(ap5095.values())):.4f}"
    )

    # NEW: optional JSON save for ablation tables
    if getattr(args, "save_results", None):
        import json
        out = {
            "ap50":     {str(k): v for k, v in ap50.items()},
            "ap5095":   {str(k): v for k, v in ap5095.items()},
            "summary":  {k: (int(v) if isinstance(v, (np.integer,)) else float(v))
                         for k, v in summary.items()},
            "config":   args.config,
            "checkpoint": args.checkpoint,
        }
        with open(args.save_results, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
