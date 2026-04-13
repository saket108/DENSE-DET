from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.detection_metrics import box_iou_cxcywh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_uint8_image(image: torch.Tensor) -> Image.Image:
    tensor = image.detach().cpu().float()
    tensor = tensor * IMAGENET_STD + IMAGENET_MEAN
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1 = (boxes[:, 0] - boxes[:, 2] * 0.5) * width
    y1 = (boxes[:, 1] - boxes[:, 3] * 0.5) * height
    x2 = (boxes[:, 0] + boxes[:, 2] * 0.5) * width
    y2 = (boxes[:, 1] + boxes[:, 3] * 0.5) * height
    out = torch.stack([x1, y1, x2, y2], dim=1)
    out[:, 0::2] = out[:, 0::2].clamp(0, width)
    out[:, 1::2] = out[:, 1::2].clamp(0, height)
    return out


def _color_for_label(label: int) -> tuple[int, int, int]:
    palette = [
        (220, 20, 60),
        (30, 144, 255),
        (255, 140, 0),
        (50, 205, 50),
        (186, 85, 211),
        (255, 215, 0),
        (0, 206, 209),
        (255, 99, 71),
    ]
    return palette[label % len(palette)]


def draw_labeled_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str] | None,
    scores: torch.Tensor | None = None,
) -> Image.Image:
    canvas = _to_uint8_image(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    width, height = canvas.size
    xyxy = _cxcywh_to_xyxy(boxes.detach().cpu(), width, height)

    for index, box in enumerate(xyxy.tolist()):
        label_id = int(labels[index].item())
        name = class_names[label_id] if class_names and 0 <= label_id < len(class_names) else str(label_id)
        color = _color_for_label(label_id)
        caption = name
        if scores is not None:
            caption = f"{caption} {float(scores[index].item()):.2f}"
        draw.rectangle(box, outline=color, width=2)
        text_box = draw.textbbox((box[0], box[1]), caption, font=font)
        text_bg = [text_box[0], text_box[1], text_box[2] + 4, text_box[3] + 2]
        draw.rectangle(text_bg, fill=color)
        draw.text((box[0] + 2, box[1]), caption, fill=(0, 0, 0), font=font)
    return canvas


def save_batch_preview(
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    class_names: Sequence[str] | None,
    path: str,
    max_images: int = 4,
) -> None:
    ensure_dir(os.path.dirname(path))
    limit = min(max_images, images.shape[0])
    tiles = []
    for index in range(limit):
        target = targets[index]
        tile = draw_labeled_boxes(images[index], target["boxes"], target["labels"], class_names)
        tiles.append(tile)

    if not tiles:
        return

    tile_w, tile_h = tiles[0].size
    cols = min(2, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (tile_w * cols, tile_h * rows), color=(24, 24, 24))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        grid.paste(tile, (x, y))
    grid.save(path, quality=95)


def save_history_plot(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return

    epochs = np.array([int(row["epoch"]) for row in rows], dtype=np.int32)

    def _series(name: str) -> np.ndarray:
        values = []
        for row in rows:
            raw = row.get(name, "")
            values.append(float(raw) if raw not in {"", None, "n/a"} else np.nan)
        return np.array(values, dtype=np.float32)

    train_loss = _series("train_loss")
    val_loss = _series("val_loss")
    map50 = _series("map50")
    map5095 = _series("map5095")
    precision = _series("macro_precision")
    recall = _series("macro_recall")

    ensure_dir(os.path.dirname(out_path))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, train_loss, label="train_loss", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label="val_loss", linewidth=2)
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, map50, label="mAP50", linewidth=2)
    axes[0, 1].plot(epochs, map5095, label="mAP50-95", linewidth=2)
    axes[0, 1].set_title("mAP")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, precision, label="precision", linewidth=2)
    axes[1, 0].plot(epochs, recall, label="recall", linewidth=2)
    axes[1, 0].set_title("Precision / Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, val_loss, label="val_loss", linewidth=2)
    axes[1, 1].plot(epochs, map5095, label="mAP50-95", linewidth=2)
    axes[1, 1].set_title("Validation vs mAP50-95")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    for axis in axes.flat:
        axis.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _collect_class_curve_data(
    all_preds: list[dict[str, torch.Tensor]],
    all_targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[dict[int, dict[str, np.ndarray | float]], np.ndarray]:
    class_curves: dict[int, dict[str, np.ndarray | float]] = {}
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    bg = num_classes

    for preds, targets in zip(all_preds, all_targets):
        pred_boxes = preds["boxes"]
        pred_labels = preds["labels"]
        pred_scores = preds["confidences"]
        gt_boxes = targets["boxes"]
        gt_labels = targets["labels"]

        if pred_scores.numel() > 0:
            order = pred_scores.argsort(descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]
            pred_scores = pred_scores[order]

        if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
            ious = box_iou_cxcywh(pred_boxes, gt_boxes)
            matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
            for pred_idx in range(pred_boxes.shape[0]):
                row = ious[pred_idx].clone()
                row[matched_gt] = -1.0
                best_iou, gt_idx = row.max(0)
                pred_cls = int(pred_labels[pred_idx].item())
                if best_iou >= iou_threshold:
                    matched_gt[int(gt_idx.item())] = True
                    true_cls = int(gt_labels[int(gt_idx.item())].item())
                    confusion[true_cls, pred_cls] += 1
                else:
                    confusion[bg, pred_cls] += 1
            for gt_idx, was_matched in enumerate(matched_gt.tolist()):
                if not was_matched:
                    confusion[int(gt_labels[gt_idx].item()), bg] += 1
        elif pred_boxes.numel() > 0:
            for pred_cls in pred_labels.tolist():
                confusion[bg, int(pred_cls)] += 1
        elif gt_boxes.numel() > 0:
            for gt_cls in gt_labels.tolist():
                confusion[int(gt_cls), bg] += 1

    threshold_grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)

    for cls_id in range(num_classes):
        detections: list[tuple[float, int]] = []
        n_gt = 0

        for preds, targets in zip(all_preds, all_targets):
            gt_mask = targets["labels"] == cls_id
            gt_boxes_cls = targets["boxes"][gt_mask]
            n_gt += int(gt_boxes_cls.shape[0])

            pred_mask = preds["labels"] == cls_id
            pred_boxes = preds["boxes"][pred_mask]
            pred_scores = preds["confidences"][pred_mask]
            if pred_scores.numel() == 0:
                continue

            order = pred_scores.argsort(descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

            if gt_boxes_cls.numel() == 0:
                detections.extend((float(score), 0) for score in pred_scores.tolist())
                continue

            ious = box_iou_cxcywh(pred_boxes, gt_boxes_cls)
            matched = torch.zeros(gt_boxes_cls.shape[0], dtype=torch.bool, device=pred_boxes.device)
            for pred_idx in range(pred_boxes.shape[0]):
                row = ious[pred_idx].clone()
                row[matched] = -1.0
                best_iou, best_idx = row.max(0)
                is_tp = 0
                if best_iou >= iou_threshold:
                    matched[int(best_idx.item())] = True
                    is_tp = 1
                detections.append((float(pred_scores[pred_idx].item()), is_tp))

        if not detections:
            class_curves[cls_id] = {
                "recall_curve": np.array([0.0], dtype=np.float32),
                "precision_curve": np.array([0.0], dtype=np.float32),
                "thresholds": threshold_grid,
                "precision_by_threshold": np.zeros_like(threshold_grid),
                "recall_by_threshold": np.zeros_like(threshold_grid),
                "f1_by_threshold": np.zeros_like(threshold_grid),
                "ap": 0.0,
            }
            continue

        score_arr = np.array([score for score, _ in detections], dtype=np.float32)
        tp_arr = np.array([flag for _, flag in detections], dtype=np.float32)
        order = score_arr.argsort()[::-1]
        score_arr = score_arr[order]
        tp_arr = tp_arr[order]

        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1.0 - tp_arr)
        recall_curve = cum_tp / max(n_gt, 1)
        precision_curve = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
        precision_by_threshold = np.zeros_like(threshold_grid)
        recall_by_threshold = np.zeros_like(threshold_grid)
        f1_by_threshold = np.zeros_like(threshold_grid)

        for idx, threshold in enumerate(threshold_grid):
            keep = score_arr >= threshold
            tp = float(tp_arr[keep].sum())
            fp = float(keep.sum() - tp)
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(float(n_gt), 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            precision_by_threshold[idx] = precision
            recall_by_threshold[idx] = recall
            f1_by_threshold[idx] = f1

        class_curves[cls_id] = {
            "recall_curve": recall_curve,
            "precision_curve": precision_curve,
            "thresholds": threshold_grid,
            "precision_by_threshold": precision_by_threshold,
            "recall_by_threshold": recall_by_threshold,
            "f1_by_threshold": f1_by_threshold,
            "ap": float(np.trapz(precision_curve[::-1], recall_curve[::-1])) if recall_curve.size > 1 else 0.0,
        }

    return class_curves, confusion


def _plot_confusion_matrix(matrix: np.ndarray, labels: Sequence[str], path: str, normalize: bool = False) -> None:
    ensure_dir(os.path.dirname(path))
    display = matrix.astype(np.float32)
    if normalize:
        row_sum = display.sum(axis=1, keepdims=True)
        display = np.divide(display, np.maximum(row_sum, 1.0), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(display, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pr_curve(class_curves: dict[int, dict[str, np.ndarray | float]], class_names: Sequence[str], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_id, name in enumerate(class_names):
        data = class_curves[class_id]
        ax.plot(data["recall_curve"], data["precision_curve"], linewidth=2, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_threshold_curve(
    class_curves: dict[int, dict[str, np.ndarray | float]],
    class_names: Sequence[str],
    path: str,
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    ensure_dir(os.path.dirname(path))
    fig, ax = plt.subplots(figsize=(8, 6))
    for class_id, name in enumerate(class_names):
        data = class_curves[class_id]
        ax.plot(data["thresholds"], data[metric_key], linewidth=2, label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_detection_artifacts(
    all_preds: list[dict[str, torch.Tensor]],
    all_targets: list[dict[str, torch.Tensor]],
    class_names: Sequence[str],
    save_dir: str,
    iou_threshold: float = 0.5,
) -> None:
    ensure_dir(save_dir)
    class_curves, confusion = _collect_class_curve_data(
        all_preds,
        all_targets,
        num_classes=len(class_names),
        iou_threshold=iou_threshold,
    )
    labels = list(class_names) + ["background"]
    _plot_confusion_matrix(confusion, labels, os.path.join(save_dir, "confusion_matrix.png"), normalize=False)
    _plot_confusion_matrix(confusion, labels, os.path.join(save_dir, "confusion_matrix_normalized.png"), normalize=True)
    _plot_pr_curve(class_curves, class_names, os.path.join(save_dir, "BoxPR_curve.png"))
    _plot_threshold_curve(
        class_curves,
        class_names,
        os.path.join(save_dir, "BoxP_curve.png"),
        metric_key="precision_by_threshold",
        title="Precision-Confidence Curve",
        ylabel="Precision",
    )
    _plot_threshold_curve(
        class_curves,
        class_names,
        os.path.join(save_dir, "BoxR_curve.png"),
        metric_key="recall_by_threshold",
        title="Recall-Confidence Curve",
        ylabel="Recall",
    )
    _plot_threshold_curve(
        class_curves,
        class_names,
        os.path.join(save_dir, "BoxF1_curve.png"),
        metric_key="f1_by_threshold",
        title="F1-Confidence Curve",
        ylabel="F1",
    )
