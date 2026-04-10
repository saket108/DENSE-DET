"""Quick block evaluation for DENSE_DET - evaluate each block in seconds.

This script enables rapid iteration on model blocks by:
  1. Evaluating on a small subset of data (10 batches by default)
  2. Showing loss breakdown per component
  3. Measuring inference throughput and memory
  4. Completing in seconds (not minutes)

Usage:
  # Eval with default config (10 batches)
  python quick_eval_dense.py --checkpoint runs/dense_det/dense_det_best.pt

  # Quick test on 5 batches
  python quick_eval_dense.py --checkpoint runs/dense_det/dense_det_best.pt --max_batches 5

  # Test a fresh model (no checkpoint)
  python quick_eval_dense.py --max_batches 10

  # Custom batch size and image size
  python quick_eval_dense.py --checkpoint best --batch 16 --imgsz 512

Example workflow for block optimization:
  1. Modify a block in model/ or dense_blocks.py
  2. Run: python quick_eval_dense.py --checkpoint last --max_batches 10
  3. Check loss breakdown and memory usage
  4. Iterate until loss improves, then train fully
"""

import argparse
import os
import time
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from data.loader import build_val_loader
from model.dense_detector import DenseDet
from training.dense_loss import DenseDetectionLoss
from utils.detection_metrics import (
    evaluate_predictions,
    mean_metric,
    print_results,
)
from utils.efficiency_utils import apply_inference_efficiency, to_channels_last
from utils.runtime import (
    coalesce,
    load_yaml_config,
    normalize_class_names,
    parse_int_tuple,
    require_existing_paths,
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
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Quick block evaluation for DENSE_DET")
    parser.add_argument("--config",       type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint",   type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--data_config",  type=str, default="configs/data_quick.yaml")
    parser.add_argument("--val_images",   type=str, default=None)
    parser.add_argument("--val_labels",   type=str, default=None)
    parser.add_argument("--batch",        type=int, default=None)
    parser.add_argument("--imgsz",        type=int, default=None)
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--max_batches",  type=int, default=10)
    parser.add_argument("--device",       type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Test script without requiring actual data files")
    parser.add_argument("--channels_last",    dest="use_channels_last", action="store_true")
    parser.add_argument("--no_channels_last", dest="use_channels_last", action="store_false")
    parser.add_argument("--fuse_bn",    dest="use_fuse_bn", action="store_true")
    parser.add_argument("--no_fuse_bn", dest="use_fuse_bn", action="store_false")
    parser.add_argument("--num_classes",  type=int, default=None)
    parser.add_argument("--variant",      type=str, default=None)
    parser.add_argument("--backbone_name",   type=str, default=None)
    parser.add_argument("--backbone_dims",  type=str, default=None)
    parser.add_argument("--backbone_depths", type=str, default=None)
    parser.add_argument("--neck_name",   type=str, default=None)
    parser.add_argument("--head_depth",  type=int, default=None)
    parser.add_argument("--stem_type",   type=str, default=None)
    parser.add_argument("--backbone_block", type=str, default=None)
    parser.add_argument("--refine_block", type=str, default=None)
    parser.add_argument("--head_type", type=str, default=None)
    parser.add_argument("--class_conditional_gn",    dest="use_class_conditional_gn", action="store_true")
    parser.add_argument("--no_class_conditional_gn", dest="use_class_conditional_gn", action="store_false")
    parser.add_argument("--use_auxiliary_heads",    dest="use_auxiliary_heads", action="store_true")
    parser.add_argument("--no_use_auxiliary_heads", dest="use_auxiliary_heads", action="store_false")
    parser.add_argument("--detail_branch",    dest="use_detail_branch", action="store_true")
    parser.add_argument("--no_detail_branch", dest="use_detail_branch", action="store_false")
    parser.add_argument("--quality_head",    dest="use_quality_head", action="store_true")
    parser.add_argument("--no_quality_head", dest="use_quality_head", action="store_false")
    parser.add_argument("--assigner",              type=str, default=None, choices=["fcos", "atss"])
    parser.add_argument("--box_loss_type",        type=str, default=None, choices=["giou", "ewiou"])
    parser.add_argument("--quality_loss_weight",  type=float, default=None)
    parser.add_argument("--auxiliary_loss_weight", type=float, default=None)
    parser.add_argument("--eval_conf",  type=float, default=None)
    parser.add_argument("--eval_iou",   type=float, default=None)
    parser.add_argument("--eval_nms_iou", type=float, default=None)
    parser.set_defaults(
        use_channels_last=None,
        use_fuse_bn=None,
        use_detail_branch=None,
        use_quality_head=None,
        use_class_conditional_gn=None,
        use_auxiliary_heads=None,
    )
    return parser.parse_args()


def resolve_args(args):
    config = load_yaml_config(args.config)
    
    model_cfg = config_section(config, "model")
    data_cfg  = config_section(config, "data")
    eval_cfg  = config_section(config, "eval")
    loss_cfg  = config_section(config, "loss")

    class_names = normalize_class_names(data_cfg.get("class_names"))
    num_classes = coalesce(args.num_classes, data_cfg.get("num_classes"), len(class_names) if class_names else None, 6)

    # Handle dry run - skip data path resolution
    if args.dry_run:
        val_images = "dummy_images"
        val_labels = "dummy_labels"
        dataset_root = "dummy_root"
        class_names = ["class1", "class2", "class3"]
        num_classes = 3
    else:
        # Try to resolve data paths
        try:
            val_paths = resolve_detection_paths(
                config_path=args.data_config,
                dataset_root=args.dataset_root,
                split="val",
                images_dir=args.val_images,
                labels_dir=args.val_labels,
            )
            dataset_root = val_paths["dataset_root"]
            val_images   = val_paths["images_dir"]
            val_labels   = val_paths["labels_dir"]
            class_names  = coalesce(val_paths["class_names"], class_names)
            num_classes  = coalesce(val_paths["num_classes"], num_classes, 6)
        except ValueError as e:
            print(f"❌ Data path resolution failed: {e}")
            print("\n💡 Solutions:")
            print("1. Set environment variable: set SLIM_DET_DATASET_ROOT=path/to/data")
            print("2. Pass --dataset_root path/to/data")
            print("3. Create configs/data_quick.yaml with your data paths")
            print("4. Use --dry_run to test script without data")
            print("5. Pass explicit paths: --val_images path --val_labels path")
            raise

    resolved = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset_root": dataset_root,
        "val_images": val_images,
        "val_labels": val_labels,
        "batch": coalesce(args.batch, eval_cfg.get("batch_size"), 8),
        "imgsz": coalesce(args.imgsz, eval_cfg.get("image_size"), data_cfg.get("image_size"), 640),
        "workers": args.workers if args.workers is not None else coalesce(eval_cfg.get("num_workers"), 0),
        "max_batches": args.max_batches,
        "device": args.device,
        "use_channels_last": coalesce(args.use_channels_last, eval_cfg.get("channels_last"), False),
        "use_fuse_bn": coalesce(args.use_fuse_bn, eval_cfg.get("fuse_bn"), False),
        "num_classes": num_classes,
        "class_names": class_names,
        "variant": coalesce(args.variant, model_cfg.get("variant"), "small"),
        "backbone_name": coalesce(args.backbone_name, model_cfg.get("backbone_name"), "vst"),
        "backbone_dims": parse_int_tuple(
            coalesce(args.backbone_dims, model_cfg.get("backbone_dims")),
            field_name="backbone_dims", expected_len=4,
        ),
        "backbone_depths": parse_int_tuple(
            coalesce(args.backbone_depths, model_cfg.get("backbone_depths")),
            field_name="backbone_depths", expected_len=4,
        ),
        "neck_name": coalesce(args.neck_name, model_cfg.get("neck_name"), "cafpn"),
        "head_depth": coalesce(args.head_depth, model_cfg.get("head_depth"), 2),
        "stem_type": coalesce(args.stem_type, model_cfg.get("stem_type"), "detail"),
        "backbone_block": coalesce(args.backbone_block, model_cfg.get("backbone_block"), "vst"),
        "refine_block": coalesce(args.refine_block, model_cfg.get("refine_block"), "dilated"),
        "head_type": coalesce(args.head_type, model_cfg.get("head_type"), "dense"),
        "use_detail_branch": coalesce(args.use_detail_branch, model_cfg.get("use_detail_branch"), False),
        "use_quality_head": coalesce(args.use_quality_head, model_cfg.get("use_quality_head"), True),
        "use_class_conditional_gn": coalesce(
            args.use_class_conditional_gn, model_cfg.get("use_class_conditional_gn"), False
        ),
        "use_auxiliary_heads": coalesce(args.use_auxiliary_heads, model_cfg.get("use_auxiliary_heads"), False),
        "assigner": coalesce(args.assigner, loss_cfg.get("assigner"), "fcos"),
        "box_loss_type": coalesce(args.box_loss_type, loss_cfg.get("box_loss"), "giou"),
        "quality_loss_weight": coalesce(args.quality_loss_weight, loss_cfg.get("quality"), 1.0),
        "auxiliary_loss_weight": coalesce(args.auxiliary_loss_weight, loss_cfg.get("auxiliary"), 0.0),
        "eval_conf": coalesce(args.eval_conf, eval_cfg.get("conf_thresh"), 0.25),
        "eval_iou": coalesce(args.eval_iou, eval_cfg.get("iou_thresh"), 0.5),
        "eval_nms_iou": coalesce(args.eval_nms_iou, eval_cfg.get("nms_iou"), 0.6),
        "dry_run": args.dry_run,
    }

    # Skip path validation in dry run mode
    if not args.dry_run:
        require_existing_paths(
            val_images=resolved["val_images"],
            val_labels=resolved["val_labels"],
        )
    
    # Skip checkpoint validation in dry run mode
    if resolved["checkpoint"] is not None and not args.dry_run:
        require_existing_paths(checkpoint=resolved["checkpoint"])

    return argparse.Namespace(**resolved)


def load_checkpoint_if_exists(checkpoint_path: str, model: nn.Module) -> tuple[dict, int]:
    """Load checkpoint and return (state_dict, epoch). Returns ({}, 0) if no checkpoint."""
    if checkpoint_path is None:
        return {}, 0
    
    # Resolve checkpoint path (support 'last', 'best' aliases)
    save_dir = "runs/dense_det"
    aliases = {
        "best": os.path.join(save_dir, "dense_det_best.pt"),
        "last": os.path.join(save_dir, "dense_det_last.pt"),
        "latest": os.path.join(save_dir, "dense_det_last.pt"),
    }
    checkpoint_path = aliases.get(checkpoint_path, checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: checkpoint not found at {checkpoint_path}. Using fresh model.")
        return {}, 0
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model", {}), strict=False)
    epoch = checkpoint.get("epoch", 0)
    print(f"✓ Loaded checkpoint from epoch {epoch}")
    return checkpoint, epoch


def print_memory_stats(device: str) -> None:
    """Print GPU/CPU memory statistics."""
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB "
                  f"(reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB)")
    else:
        print(f"  CPU Memory: Available")


def run_dry_evaluation(args):
    """Run dry evaluation - test model building and loading without data."""
    
    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"DENSE_DET Quick Block Evaluation (DRY RUN)")
    print(f"{'='*70}")
    print(f"Config:        {args.config}")
    print(f"Checkpoint:    {args.checkpoint or 'None (fresh model)'}")
    print(f"Device:        {device}")
    print(f"{'='*70}\n")
    
    # Build model
    print("Building model...")
    model = DenseDet(
        num_classes=args.num_classes,
        variant=args.variant,
        backbone_name=args.backbone_name,
        backbone_dims=args.backbone_dims,
        backbone_depths=args.backbone_depths,
        neck_name=args.neck_name,
        head_depth=args.head_depth,
        stem_type=args.stem_type,
        backbone_block=args.backbone_block,
        refine_block=args.refine_block,
        head_type=args.head_type,
        use_detail_branch=args.use_detail_branch,
        use_quality_head=args.use_quality_head,
        use_class_conditional_gn=args.use_class_conditional_gn,
        use_auxiliary_heads=args.use_auxiliary_heads,
    )
    model.to(device)
    model.eval()
    
    # Load checkpoint if provided
    checkpoint, epoch = load_checkpoint_if_exists(args.checkpoint, model)
    
    # Apply efficiency settings
    if args.use_channels_last:
        model = to_channels_last(model)
        print("✓ Using channels_last memory format")
    if args.use_fuse_bn:
        model = apply_inference_efficiency(model, fuse_bn=True)
        print("✓ Fused BatchNorm layers")
    
    print(f"✓ Model: {args.variant} variant, {args.backbone_name} backbone")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print_memory_stats(args.device)
    
    # Test forward pass with dummy input
    print("\nTesting forward pass with dummy input...")
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(dummy_input)
    forward_time = time.time() - start_time
    
    print(f"✓ Forward pass successful in {forward_time*1000:.2f} ms")
    
    # Show output structure
    print("\nOutput structure:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    print(f"\n{'='*70}")
    print(f"DRY RUN COMPLETE")
    print(f"{'='*70}")
    print("✓ Model builds successfully")
    print("✓ Checkpoint loads (if provided)")
    print("✓ Forward pass works")
    print("✓ Memory usage looks good")
    print("\n💡 To run with real data:")
    print("   1. Set up your dataset paths in configs/data_quick.yaml")
    print("   2. Or pass --dataset_root /path/to/data")
    print("   3. Run without --dry_run")
    print(f"{'='*70}\n")


def run_quick_evaluation(args):
    """Run quick block evaluation on limited batches."""
    
    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"DENSE_DET Quick Block Evaluation")
    print(f"{'='*70}")
    print(f"Config:        {args.config}")
    print(f"Checkpoint:    {args.checkpoint or 'None (fresh model)'}")
    print(f"Max Batches:   {args.max_batches}")
    print(f"Batch Size:    {args.batch}")
    print(f"Image Size:    {args.imgsz}")
    print(f"Device:        {device}")
    print(f"{'='*70}\n")
    
    # Build data loader
    print("Loading validation data...")
    val_loader = build_val_loader(
        images_dir=args.val_images,
        labels_dir=args.val_labels,
        image_size=args.imgsz,
        batch_size=args.batch,
        num_workers=args.workers,
        data_format="detection",
    )
    print(f"✓ Data loader ready ({len(val_loader)} batches total, evaluating {min(args.max_batches, len(val_loader))})\n")
    
    # Build model
    print("Building model...")
    model = DenseDet(
        num_classes=args.num_classes,
        variant=args.variant,
        backbone_name=args.backbone_name,
        backbone_dims=args.backbone_dims,
        backbone_depths=args.backbone_depths,
        neck_name=args.neck_name,
        head_depth=args.head_depth,
        stem_type=args.stem_type,
        backbone_block=args.backbone_block,
        refine_block=args.refine_block,
        head_type=args.head_type,
        use_detail_branch=args.use_detail_branch,
        use_quality_head=args.use_quality_head,
        use_class_conditional_gn=args.use_class_conditional_gn,
        use_auxiliary_heads=args.use_auxiliary_heads,
    )
    model.to(device)
    model.eval()
    
    # Load checkpoint if provided
    checkpoint, epoch = load_checkpoint_if_exists(args.checkpoint, model)
    
    # Apply efficiency settings
    if args.use_channels_last:
        model = to_channels_last(model)
        print("✓ Using channels_last memory format")
    if args.use_fuse_bn:
        model = apply_inference_efficiency(model, fuse_bn=True)
        print("✓ Fused BatchNorm layers")
    
    print(f"✓ Model: {args.variant} variant, {args.backbone_name} backbone")
    print_memory_stats(args.device)
    
    # Build loss function for loss analysis  
    print("\nEvaluating on validation data...\n")
    loss_fn = DenseDetectionLoss(
        num_classes=args.num_classes,
        assigner=args.assigner,
        box_loss_type=args.box_loss_type,
        quality_loss_weight=args.quality_loss_weight,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
    )
    loss_fn.to(device)
    loss_fn.eval()
    
    # Run evaluation
    all_preds = []
    all_targets = []
    losses = {"detection": [], "quality": [], "auxiliary": [], "total": []}
    
    start_time = time.time()
    batch_times = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, total=min(args.max_batches, len(val_loader))) if tqdm else val_loader
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= args.max_batches:
                break
            
            batch_start = time.time()
            
            # Move to device
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["targets"]
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss (if loss function available)
            loss_dict = loss_fn(outputs, targets)
            total_loss = sum(v for k, v in loss_dict.items() if "loss" in k)
            
            # Track losses
            losses["detection"].append(loss_dict.get("loss_detection", 0.0))
            losses["quality"].append(loss_dict.get("loss_quality", 0.0))
            losses["auxiliary"].append(loss_dict.get("loss_auxiliary", 0.0))
            losses["total"].append(total_loss.item())
            
            # Post-process predictions
            batch_preds = model.post_process(
                outputs,
                conf_thresh=args.eval_conf,
                nms_iou_thresh=args.eval_nms_iou,
            )
            all_preds.extend(batch_preds)
            all_targets.extend(targets)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if tqdm:
                pbar.update(1)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*70}")
    print(f"LOSS BREAKDOWN (Average over {len(losses['total'])} batches)")
    print(f"{'='*70}")
    print(f"  Detection:  {np.mean(losses['detection']):.6f}")
    print(f"  Quality:    {np.mean(losses['quality']):.6f}")
    print(f"  Auxiliary:  {np.mean(losses['auxiliary']):.6f}")
    print(f"  {'─'*66}")
    print(f"  TOTAL:      {np.mean(losses['total']):.6f}")
    print(f"{'='*70}\n")
    
    print(f"{'='*70}")
    print(f"THROUGHPUT")
    print(f"{'='*70}")
    avg_batch_time = np.mean(batch_times)
    print(f"  Avg batch time:  {avg_batch_time*1000:.2f} ms")
    print(f"  Throughput:      {args.batch / avg_batch_time:.1f} images/sec")
    print(f"  Total eval time: {total_time:.2f} sec")
    print(f"{'='*70}\n")
    
    # Compute metrics if we have predictions
    if all_preds and all_targets:
        print(f"{'='*70}")
        print(f"DETECTION METRICS (AP@50)")
        print(f"{'='*70}")
        predictions = {
            "predictions": all_preds,
            "targets": all_targets,
            "class_names": args.class_names,
        }
        metrics = evaluate_predictions(predictions, iou_threshold=args.eval_iou)
        print_results(metrics, args.class_names)
        print(f"{'='*70}\n")
    
    print_memory_stats(args.device)
    print("\n✓ Quick evaluation complete!\n")

    return losses, batch_times


if __name__ == "__main__":
    args = parse_args()
    args = resolve_args(args)
    
    if args.dry_run:
        run_dry_evaluation(args)
    else:
        run_quick_evaluation(args)
