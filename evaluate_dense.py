"""Evaluate the lightweight PRISM-based DenseDet model."""

from __future__ import annotations

import argparse
import json
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
from utils.detection_metrics import evaluate_predictions, mean_metric, print_results, summarize_metrics
from utils.reporting import save_batch_preview, save_detection_artifacts
from utils.runtime import coalesce, load_yaml_config, normalize_class_names, parse_int_tuple, require_existing_paths, resolve_detection_paths


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "dense_det.yaml")


def config_section(config, key):
    value = config.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def print_benchmark_comparison(ap50, ap5095, pr_metrics, summary, benchmark, class_names=None):
    if not isinstance(benchmark, Mapping) or not benchmark:
        return

    metrics = benchmark.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}

    name = benchmark.get("name", "benchmark")
    current = {
        "precision": float(summary["macro_precision"]),
        "recall": float(summary["macro_recall"]),
        "map50": mean_metric(list(ap50.values())),
        "map50_95": mean_metric(list(ap5095.values())),
    }
    print("\n" + "=" * 82)
    print(f"BENCHMARK COMPARISON: {name}")
    print("-" * 82)
    print(f"{'Metric':<14} {'DenseDet':>10} {'Benchmark':>10} {'Delta':>10} {'Status':>8}")
    for key in ("precision", "recall", "map50", "map50_95"):
        if key not in metrics:
            continue
        base = float(metrics[key])
        delta = current[key] - base
        status = "PASS" if delta >= 0.0 else "MISS"
        print(f"{key:<14} {current[key]:>10.3f} {base:>10.3f} {delta:>+10.3f} {status:>8}")

    class_bench = benchmark.get("classes", {})
    if isinstance(class_bench, Mapping) and class_names:
        print("-" * 82)
        print(f"{'Class':<18} {'AP50 d':>9} {'AP50-95 d':>11} {'Recall d':>10}")
        for class_id, class_name in enumerate(class_names):
            target = class_bench.get(class_name)
            if not isinstance(target, Mapping):
                continue
            ap50_delta = float(ap50.get(class_id, 0.0)) - float(target.get("map50", 0.0))
            ap95_delta = float(ap5095.get(class_id, 0.0)) - float(target.get("map50_95", 0.0))
            recall_delta = float(pr_metrics[class_id]["recall"]) - float(target.get("recall", 0.0))
            print(f"{class_name:<18} {ap50_delta:>+9.3f} {ap95_delta:>+11.3f} {recall_delta:>+10.3f}")
    print("=" * 82)


def print_prediction_diagnostics(all_preds, max_det):
    counts = [int(pred["confidences"].numel()) for pred in all_preds]
    if not counts:
        return

    scores = [
        pred["confidences"].detach().float().cpu()
        for pred in all_preds
        if pred["confidences"].numel() > 0
    ]
    avg_count = float(np.mean(counts))
    capped = sum(1 for count in counts if count >= int(max_det))

    print("\nPrediction diagnostics:")
    print(f"  Detections/image : avg={avg_count:.1f}  maxed={capped}/{len(counts)} at max_det={max_det}")
    if scores:
        scores_t = torch.cat(scores)
        q = torch.quantile(scores_t, torch.tensor([0.5, 0.9, 0.99], dtype=scores_t.dtype))
        print(
            "  Confidence       : "
            f"min={scores_t.min().item():.4f}  "
            f"p50={q[0].item():.4f}  "
            f"p90={q[1].item():.4f}  "
            f"p99={q[2].item():.4f}  "
            f"max={scores_t.max().item():.4f}"
        )
    if capped > 0:
        print("  Note : many images are hitting max_det; precision will be very noisy.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the lightweight DenseDet model")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--val_images", type=str, default=None)
    parser.add_argument("--test_images", type=str, default=None)
    parser.add_argument("--val_labels", type=str, default=None)
    parser.add_argument("--test_labels", type=str, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou_thresh", type=float, default=None)
    parser.add_argument("--nms_iou", type=float, default=None)
    parser.add_argument("--max_det", type=int, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--backbone_dims", type=str, default=None)
    parser.add_argument("--backbone_depths", type=str, default=None)
    parser.add_argument("--head_depth", type=int, default=None)
    parser.add_argument("--quality_head", dest="use_quality_head", action="store_true")
    parser.add_argument("--no_quality_head", dest="use_quality_head", action="store_false")
    parser.add_argument("--weights", type=str, default="auto", choices=["auto", "ema", "model"])
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--artifact_dir", type=str, default=None)
    parser.add_argument("--strict", dest="strict_load", action="store_true")
    parser.add_argument("--no_strict", dest="strict_load", action="store_false")
    parser.set_defaults(use_quality_head=None)
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg, save_dir):
    aliases = {"best": os.path.join(save_dir, "dense_det_best.pt"), "last": os.path.join(save_dir, "dense_det_last.pt"), "latest": os.path.join(save_dir, "dense_det_last.pt")}
    candidate = aliases.get(checkpoint_arg, checkpoint_arg)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: '{candidate}'")


def resolve_args(args):
    config = load_yaml_config(args.config)
    model_cfg = config_section(config, "model")
    data_cfg = config_section(config, "data")
    train_cfg = config_section(config, "train")
    eval_cfg = config_section(config, "eval")
    checkpoint_cfg = config_section(config, "checkpoint")
    benchmark_cfg = config_section(config, "benchmark")

    val_paths = resolve_detection_paths(args.data_config, args.dataset_root, "val", args.val_images, args.val_labels)
    test_paths = resolve_detection_paths(args.data_config, val_paths["dataset_root"], "test", args.test_images, args.test_labels)
    active = test_paths if args.split == "test" else val_paths

    class_names = coalesce(active["class_names"], normalize_class_names(data_cfg.get("class_names")))
    num_classes = coalesce(active["num_classes"], data_cfg.get("num_classes"), len(class_names) if class_names else None, 6)
    resolved = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "data_config": args.data_config,
        "dataset_root": active["dataset_root"],
        "images_dir": active["images_dir"],
        "labels_dir": active["labels_dir"],
        "batch": coalesce(args.batch, eval_cfg.get("batch_size"), train_cfg.get("batch_size"), 8),
        "imgsz": coalesce(args.imgsz, train_cfg.get("image_size"), data_cfg.get("image_size"), 640),
        "workers": args.workers if args.workers is not None else coalesce(train_cfg.get("num_workers"), 4),
        "max_batches": args.max_batches,
        "conf": coalesce(args.conf, eval_cfg.get("conf_thresh"), 0.05),
        "iou_thresh": coalesce(args.iou_thresh, eval_cfg.get("iou_thresh"), 0.5),
        "nms_iou": coalesce(args.nms_iou, eval_cfg.get("nms_iou"), 0.6),
        "max_det": coalesce(args.max_det, eval_cfg.get("max_det"), 300),
        "num_classes": num_classes,
        "class_names": class_names,
        "variant": coalesce(args.variant, model_cfg.get("variant"), "small"),
        "backbone_dims": parse_int_tuple(coalesce(args.backbone_dims, model_cfg.get("backbone_dims"), (16, 32, 64, 128)), "backbone_dims", 4),
        "backbone_depths": parse_int_tuple(coalesce(args.backbone_depths, model_cfg.get("backbone_depths"), (2, 2, 4, 2)), "backbone_depths", 4),
        "head_depth": coalesce(args.head_depth, model_cfg.get("head_depth"), 2),
        "use_quality_head": coalesce(args.use_quality_head, model_cfg.get("use_quality_head"), True),
        "weights": args.weights,
        "save_results": args.save_results,
        "artifact_dir": args.artifact_dir,
        "strict_load": args.strict_load if args.strict_load is not None else True,
        "save_dir": coalesce(checkpoint_cfg.get("save_dir"), "runs/dense_det"),
        "benchmark": benchmark_cfg,
    }
    require_existing_paths(images_dir=resolved["images_dir"], labels_dir=resolved["labels_dir"])
    if resolved["checkpoint"] is not None:
        resolved["checkpoint"] = resolve_checkpoint_path(resolved["checkpoint"], resolved["save_dir"])
    return argparse.Namespace(**resolved)


def build_model_config(args):
    return {"num_classes": args.num_classes, "variant": args.variant, "backbone_dims": args.backbone_dims, "backbone_depths": args.backbone_depths, "head_depth": args.head_depth, "use_quality_head": args.use_quality_head}


def apply_checkpoint_model_config(args, checkpoint):
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        return args
    for key in ("num_classes", "class_names", "variant", "backbone_dims", "backbone_depths", "head_depth", "use_quality_head"):
        if key in model_config:
            setattr(args, key, model_config[key])
    return args


@torch.no_grad()
def run_dense_evaluation(model, loader, device, num_classes, conf_thresh=0.05, match_iou=0.5, nms_iou=0.6, max_batches=None, max_det=300, verbose=True, progress_label=None):
    model.eval()
    all_preds, all_targets = [], []
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(loader, total=total_batches, desc=progress_label or "Eval", dynamic_ncols=True, leave=False, disable=(tqdm is None or not verbose)) if tqdm else None
    iterator = progress or loader
    for batch_index, (images, targets, _) in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        predictions = model.predict(images.to(device), conf_threshold=conf_thresh, nms_iou=nms_iou, max_det=max_det)
        for pred, target in zip(predictions, targets):
            all_preds.append({"boxes": pred["boxes"].cpu(), "labels": pred["labels"].cpu(), "confidences": pred["confidences"].cpu()})
            all_targets.append({"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})
    if progress is not None:
        progress.close()
    if verbose:
        print_prediction_diagnostics(all_preds, max_det)

    ap50_metrics = evaluate_predictions(all_preds, all_targets, num_classes=num_classes, iou_threshold=0.5)
    pr_metrics = ap50_metrics if match_iou == 0.5 else evaluate_predictions(all_preds, all_targets, num_classes=num_classes, iou_threshold=match_iou)
    ap_all = {class_id: [float(ap50_metrics[class_id]["ap"])] for class_id in range(num_classes)}
    for iou_threshold in np.arange(0.55, 1.0, 0.05):
        metrics = evaluate_predictions(all_preds, all_targets, num_classes=num_classes, iou_threshold=float(iou_threshold))
        for class_id, class_metrics in metrics.items():
            ap_all[class_id].append(float(class_metrics["ap"]))
    ap50 = {class_id: float(metrics["ap"]) for class_id, metrics in ap50_metrics.items()}
    ap5095 = {class_id: float(np.mean(values)) for class_id, values in ap_all.items()}
    summary = summarize_metrics(pr_metrics)
    return ap50, ap5095, pr_metrics, summary


@torch.no_grad()
def run_dense_evaluation_with_raw(model, loader, device, num_classes, conf_thresh=0.05, match_iou=0.5, nms_iou=0.6, max_batches=None, max_det=300, verbose=True, progress_label=None):
    model.eval()
    all_preds, all_targets = [], []
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    progress = tqdm(loader, total=total_batches, desc=progress_label or "Eval", dynamic_ncols=True, leave=False, disable=(tqdm is None or not verbose)) if tqdm else None
    iterator = progress or loader
    for batch_index, (images, targets, _) in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break
        predictions = model.predict(images.to(device), conf_threshold=conf_thresh, nms_iou=nms_iou, max_det=max_det)
        for pred, target in zip(predictions, targets):
            all_preds.append({"boxes": pred["boxes"].cpu(), "labels": pred["labels"].cpu(), "confidences": pred["confidences"].cpu()})
            all_targets.append({"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})
    if progress is not None:
        progress.close()
    if verbose:
        print_prediction_diagnostics(all_preds, max_det)

    ap50_metrics = evaluate_predictions(all_preds, all_targets, num_classes=num_classes, iou_threshold=0.5)
    pr_metrics = ap50_metrics if match_iou == 0.5 else evaluate_predictions(all_preds, all_targets, num_classes=num_classes, iou_threshold=match_iou)
    ap_all = {class_id: [float(ap50_metrics[class_id]["ap"])] for class_id in range(num_classes)}
    for iou_threshold in np.arange(0.55, 1.0, 0.05):
        metrics = evaluate_predictions(all_preds, all_targets, num_classes=num_classes, iou_threshold=float(iou_threshold))
        for class_id, class_metrics in metrics.items():
            ap_all[class_id].append(float(class_metrics["ap"]))
    ap50 = {class_id: float(metrics["ap"]) for class_id, metrics in ap50_metrics.items()}
    ap5095 = {class_id: float(np.mean(values)) for class_id, values in ap_all.items()}
    summary = summarize_metrics(pr_metrics)
    return ap50, ap5095, pr_metrics, summary, all_preds, all_targets


def main():
    args = resolve_args(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        args = apply_checkpoint_model_config(args, checkpoint)

    print(f"Device       : {device}")
    print("Architecture : PRISM + CAFPN + DenseHead")
    print(f"Variant      : {args.variant}")
    print(f"Backbone cfg : dims={args.backbone_dims}, depths={args.backbone_depths}")
    print(f"Quality head : {args.use_quality_head}")
    print(f"Data         : {args.images_dir}")
    print(f"Classes      : {args.num_classes}")
    print(f"Conf         : {args.conf}")
    print(f"Match IoU    : {args.iou_thresh}")
    print(f"NMS IoU      : {args.nms_iou}")
    print(f"Max det      : {args.max_det}")

    print("\nBuilding DenseDet...")
    model = DenseDet(**build_model_config(args)).to(device)
    if checkpoint is not None:
        if args.weights == "ema" and checkpoint.get("ema") is None:
            raise ValueError("Checkpoint does not contain EMA weights.")
        state_key = (
            "ema"
            if args.weights == "ema" or (args.weights == "auto" and checkpoint.get("ema") is not None)
            else "model"
        )
        if args.strict_load:
            model.load_state_dict(checkpoint[state_key])
        else:
            missing, unexpected = model.load_state_dict(checkpoint[state_key], strict=False)
            if missing:
                print(f"Warning: {len(missing)} missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
        print(f"Loaded checkpoint: {args.checkpoint} ({state_key} weights)")
    else:
        print("No checkpoint, evaluating random weights only")

    loader = build_val_loader(args.images_dir, args.labels_dir, batch_size=args.batch, image_size=args.imgsz, num_workers=args.workers, class_names=args.class_names)
    print(f"Eval batches: {len(loader)}")

    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        artifact_dir = os.path.dirname(args.checkpoint) if args.checkpoint else args.save_dir

    preview_batch = next(iter(loader))
    save_batch_preview(preview_batch[0], preview_batch[1], args.class_names, os.path.join(artifact_dir, "val_batch_labels.jpg"))

    ap50, ap5095, pr_metrics, summary, all_preds, all_targets = run_dense_evaluation_with_raw(
        model,
        loader,
        device,
        num_classes=args.num_classes,
        conf_thresh=args.conf,
        match_iou=args.iou_thresh,
        nms_iou=args.nms_iou,
        max_batches=args.max_batches,
        max_det=args.max_det,
    )
    print_results(ap50, ap5095, pr_metrics, summary, match_iou=args.iou_thresh, class_names=args.class_names)
    print(f"Macro mAP50={mean_metric(list(ap50.values())):.4f} mAP50-95={mean_metric(list(ap5095.values())):.4f}")
    print_benchmark_comparison(ap50, ap5095, pr_metrics, summary, args.benchmark, args.class_names)
    save_detection_artifacts(all_preds, all_targets, args.class_names, artifact_dir, iou_threshold=args.iou_thresh)

    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as handle:
            json.dump({"ap50": {str(k): v for k, v in ap50.items()}, "ap5095": {str(k): v for k, v in ap5095.items()}, "summary": summary, "benchmark": args.benchmark, "config": args.config, "checkpoint": args.checkpoint}, handle, indent=2)
        print(f"Results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
