"""Detection-only data loading for DenseDet."""

from __future__ import annotations

import os
import random
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_transforms(image_size: int = 640) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class DetectionAugmenter:
    """Lightweight image augmentation for detection training."""

    def __init__(
        self,
        enabled: bool = True,
        close_after_epochs: int = 35,
        fliplr: float = 0.5,
        hsv_h: float = 0.01,
        hsv_s: float = 0.3,
        hsv_v: float = 0.15,
        blur_prob: float = 0.01,
        grayscale_prob: float = 0.0,
        equalize_prob: float = 0.01,
        erasing_prob: float = 0.05,
        image_size: int = 640,
    ) -> None:
        self.enabled = bool(enabled)
        self.close_after_epochs = max(int(close_after_epochs), 0)
        self.fliplr = float(fliplr)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.blur_prob = float(blur_prob)
        self.grayscale_prob = float(grayscale_prob)
        self.equalize_prob = float(equalize_prob)
        self.erasing_prob = float(erasing_prob)
        self.image_size = int(image_size)
        self.current_epoch = 1

        self._to_tensor = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._full_no_aug = make_transforms(image_size)
        self._eraser = (
            T.RandomErasing(
                p=self.erasing_prob,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0.0,
            )
            if self.erasing_prob > 0.0
            else None
        )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 1)

    @property
    def active(self) -> bool:
        if not self.enabled:
            return False
        if self.close_after_epochs <= 0:
            return True
        return self.current_epoch <= self.close_after_epochs

    def _apply_pil_ops(self, image: Image.Image) -> Image.Image:
        if self.hsv_h > 0.0:
            image = TF.adjust_hue(image, random.uniform(-self.hsv_h, self.hsv_h))
        if self.hsv_s > 0.0:
            scale = 1.0 + random.uniform(-self.hsv_s, self.hsv_s)
            image = TF.adjust_saturation(image, max(scale, 0.0))
        if self.hsv_v > 0.0:
            scale = 1.0 + random.uniform(-self.hsv_v, self.hsv_v)
            image = TF.adjust_brightness(image, max(scale, 0.0))
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < self.grayscale_prob:
            image = ImageOps.grayscale(image).convert("RGB")
        if random.random() < self.equalize_prob:
            image = ImageOps.equalize(image)
        return image

    def __call__(
        self,
        image: Image.Image,
        boxes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        boxes = None if boxes is None else boxes.clone()

        if not self.active:
            return self._full_no_aug(image), boxes

        image = self._apply_pil_ops(image)

        if random.random() < self.fliplr:
            image = TF.hflip(image)
            if boxes is not None and boxes.numel() > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        tensor = self._to_tensor(image)
        if self._eraser is not None:
            tensor = self._eraser(tensor)
        tensor = self._normalize(tensor)
        return tensor, boxes


def _list_image_files(images_dir: str) -> list[str]:
    image_paths: list[str] = []
    for root, _, files in os.walk(images_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, name))
    image_paths.sort()
    return image_paths


def _label_path_for_image(image_path: str, images_dir: str, labels_dir: str) -> str:
    relative_path = os.path.relpath(image_path, images_dir)
    stem, _ = os.path.splitext(relative_path)
    return os.path.join(labels_dir, stem + ".txt")


def _parse_detection_label_file(label_path: str) -> list[dict[str, float | int | list[float]]]:
    annotations: list[dict[str, float | int | list[float]]] = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(
                    f"Warning: {label_path}:{line_no}: expected 5 values, "
                    f"got {len(parts)}. Skipping line."
                )
                continue

            try:
                class_id = int(float(parts[0]))
                cx, cy, width, height = [float(value) for value in parts[1:]]
            except ValueError:
                print(f"Warning: {label_path}:{line_no}: could not parse line. Skipping.")
                continue

            annotations.append(
                {
                    "class_id": class_id,
                    "box": [
                        max(0.0, min(1.0, cx)),
                        max(0.0, min(1.0, cy)),
                        max(0.0, min(1.0, width)),
                        max(0.0, min(1.0, height)),
                    ],
                }
            )
    return annotations


def _empty_target(image_id: str) -> dict[str, torch.Tensor | str]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
        "image_id": image_id,
    }


def _target_from_annotations(
    annotations: list[dict[str, float | int | list[float]]],
    image_id: str,
) -> dict[str, torch.Tensor | str]:
    if not annotations:
        return _empty_target(image_id)

    boxes = torch.tensor([ann["box"] for ann in annotations], dtype=torch.float32)
    labels = torch.tensor([int(ann["class_id"]) for ann in annotations], dtype=torch.long)
    return {"boxes": boxes, "labels": labels, "image_id": image_id}


class StandardDetectionDataset(Dataset):
    """YOLO-format detection dataset with image folders and txt labels."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: int = 640,
        class_names: list[str] | None = None,
        is_train: bool = True,
        augmenter: DetectionAugmenter | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = int(image_size)
        self.class_names = class_names or []
        self.is_train = bool(is_train)
        self.augmenter = augmenter
        self.transform = make_transforms(image_size)

        self.records: list[dict[str, object]] = []
        self.sample_class_ids: list[list[int]] = []

        for image_path in _list_image_files(images_dir):
            image_id = os.path.relpath(image_path, images_dir)
            label_path = _label_path_for_image(image_path, images_dir, labels_dir)
            annotations = _parse_detection_label_file(label_path)
            self.records.append(
                {
                    "image_path": image_path,
                    "image_id": image_id,
                    "annotations": annotations,
                }
            )
            self.sample_class_ids.append([int(ann["class_id"]) for ann in annotations])

        print(f"Loaded {len(self.records)} images from {images_dir}")
        self._print_class_stats()

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def _print_class_stats(self) -> None:
        counts: Counter[str] = Counter()
        for class_ids in self.sample_class_ids:
            for class_id in class_ids:
                counts[self._class_name(class_id)] += 1
        print("  Class distribution:")
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {name:<16}: {count}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        target = _target_from_annotations(record["annotations"], image_id=record["image_id"])

        if self.is_train and self.augmenter is not None:
            image_tensor, boxes = self.augmenter(image, target["boxes"])
            target["boxes"] = boxes if boxes is not None else target["boxes"]
        else:
            image_tensor = self.transform(image)

        return image_tensor, target, record["image_id"]


def collate_fn(batch):
    images, targets, image_ids = [], [], []
    for image, target, image_id in batch:
        images.append(image)
        targets.append(target)
        image_ids.append(image_id)
    return torch.stack(images, dim=0), targets, image_ids


class ClassBalancedSampler(torch.utils.data.Sampler):
    """Oversample images containing rare classes."""

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int | None = None,
        rarity_power: float = 0.5,
        background_weight: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)

        sample_class_ids = getattr(dataset, "sample_class_ids", None)
        if sample_class_ids is None:
            raise ValueError("Dataset does not expose sample_class_ids for balanced sampling.")

        image_class_counts: Counter[int] = Counter()
        for class_ids in sample_class_ids:
            image_class_counts.update(set(class_ids))

        if not image_class_counts:
            self.class_weights = {}
            self.background_weight = 1.0
            self.weights = torch.ones(len(sample_class_ids), dtype=torch.float32)
            return

        raw_weights = {
            class_id: 1.0 / float(max(count, 1)) ** rarity_power
            for class_id, count in image_class_counts.items()
        }
        max_weight = max(raw_weights.values())
        self.class_weights = {
            class_id: weight / max_weight for class_id, weight in raw_weights.items()
        }

        min_weight = min(self.class_weights.values())
        self.background_weight = (
            background_weight
            if background_weight is not None
            else max(min_weight * 0.5, 0.05)
        )

        sample_weights = []
        for class_ids in sample_class_ids:
            unique_ids = sorted(set(class_ids))
            if not unique_ids:
                sample_weights.append(self.background_weight)
                continue
            rarest = max(self.class_weights.get(class_id, min_weight) for class_id in unique_ids)
            diversity_boost = 1.0 + 0.05 * max(len(unique_ids) - 1, 0)
            sample_weights.append(rarest * diversity_boost)

        self.weights = torch.tensor(sample_weights, dtype=torch.float32)
        self._print_sampler_stats(image_class_counts)

    def _class_name(self, class_id: int) -> str:
        if hasattr(self.dataset, "_class_name"):
            return self.dataset._class_name(class_id)
        return f"class_{class_id}"

    def _print_sampler_stats(self, image_class_counts: Counter[int]) -> None:
        print("  Balanced sampler:")
        for class_id, count in sorted(image_class_counts.items(), key=lambda item: (item[1], item[0])):
            print(
                f"    {self._class_name(class_id):<16}: "
                f"img_freq={count:<5d} weight={self.class_weights[class_id]:.3f}"
            )

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_train_loader(
    images_dir: str,
    labels_dir: str,
    batch_size: int = 8,
    image_size: int = 640,
    num_workers: int = 4,
    balanced: bool = True,
    class_names: list[str] | None = None,
    augmenter: DetectionAugmenter | None = None,
) -> DataLoader:
    dataset = StandardDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=image_size,
        class_names=class_names,
        is_train=True,
        augmenter=augmenter,
    )
    if len(dataset) == 0:
        raise ValueError(f"No training images found in '{images_dir}'.")

    sampler = ClassBalancedSampler(dataset) if balanced else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) > batch_size,
    )


def build_val_loader(
    images_dir: str,
    labels_dir: str,
    batch_size: int = 8,
    image_size: int = 640,
    num_workers: int = 4,
    class_names: list[str] | None = None,
) -> DataLoader:
    dataset = StandardDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=image_size,
        class_names=class_names,
        is_train=False,
    )
    if len(dataset) == 0:
        raise ValueError(f"No validation images found in '{images_dir}'.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
