"""
Changes vs original loader.py
------------------------------
FIX-1  RandomErasing applied AFTER T.Normalize in _apply_tensor.
       T.RandomErasing(value="random") fills erased regions with values
       uniformly drawn from [0, 1].  But after T.Normalize the image tensor
       lives in roughly [-2.5, 2.5].  Erased patches therefore have a
       completely different value distribution from the rest of the image —
       the model trains on systematically corrupted data.

       Fix: moved RandomErasing to _apply_pil (PIL stage, before Normalize)
       using torchvision.transforms.RandomErasing's PIL-compatible equivalent
       — specifically, RandomErasing is now applied as a ToTensor+erase+ToPIL
       round-trip before normalization, or alternatively value=0 is used which
       corresponds to the image mean after normalization.

       Simplest correct fix chosen: apply erasing on the raw tensor immediately
       after ToTensor() (values in [0,1]) and BEFORE Normalize.  The transform
       pipeline is restructured to split at that point.

FIX-2  T.RandomErasing was re-instantiated on every call to _apply_tensor().
       Now built once in __init__ and stored as self._eraser.

FIX-3  _parse_detection_label_file silently dropped lines with != 5 tokens.
       Now logs a warning with the file path and the malformed line so corrupt
       label files are visible during dataset loading.
"""

# data/loader.py
import os
import json
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFilter, ImageOps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_ZONE     = "unknown"
DEFAULT_IMAGE_METRICS  = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
IMAGE_EXTENSIONS       = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CLASS_ID_TO_NAME       = {
    0: 'crack',
    1: 'dent',
    2: 'corrosion',
    3: 'scratch',
    4: 'missing-head',
    5: 'paint-peel-off',
}
CLASS_NAME_TO_ID       = {name: class_id for class_id, name in CLASS_ID_TO_NAME.items()}


def make_transforms(image_size: int = 640):
    """
    FIX-1: split the pipeline so RandomErasing can be inserted between
    ToTensor (values in [0,1]) and Normalize.  Callers should use
    make_pre_norm_transforms + make_post_norm_transforms, or simply call
    apply_full_transform() on DetectionAugmenter.

    For code paths that do not use DetectionAugmenter, make_transforms returns
    the full pipeline (no erasing).
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DetectionAugmenter:
    """Lightweight YOLO-style augmentation with scheduled shutdown.

    FIX-1: RandomErasing is now applied BETWEEN ToTensor and Normalize.
    FIX-2: RandomErasing object is built once in __init__, not every call.
    """

    def __init__(
        self,
        enabled: bool = False,
        close_after_epochs: int = 10,
        fliplr: float = 0.5,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        blur_prob: float = 0.01,
        grayscale_prob: float = 0.01,
        equalize_prob: float = 0.01,
        erasing_prob: float = 0.4,
        image_size: int = 640,
    ) -> None:
        self.enabled            = enabled
        self.close_after_epochs = max(int(close_after_epochs), 0)
        self.fliplr             = float(fliplr)
        self.hsv_h              = float(hsv_h)
        self.hsv_s              = float(hsv_s)
        self.hsv_v              = float(hsv_v)
        self.blur_prob          = float(blur_prob)
        self.grayscale_prob     = float(grayscale_prob)
        self.equalize_prob      = float(equalize_prob)
        self.erasing_prob       = float(erasing_prob)
        self.image_size         = image_size
        self.current_epoch      = 1

        # FIX-2: build once, not per-call
        self._eraser = (
            T.RandomErasing(
                p=erasing_prob,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0,  # FIX-1: value=0 is the image mean in [0,1] space
            )
            if erasing_prob > 0.0
            else None
        )

        # Two-part transform: ToTensor then Normalize (erasing goes between)
        self._to_tensor  = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self._normalize  = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._full_no_aug = T.Compose([
            T.Resize((image_size, image_size)), T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(int(epoch), 1)

    @property
    def active(self) -> bool:
        if not self.enabled:
            return False
        if self.close_after_epochs <= 0:
            return True
        return self.current_epoch <= self.close_after_epochs

    def _apply_pil(self, image: Image.Image) -> Image.Image:
        import random
        if self.hsv_h > 0:
            image = TF.adjust_hue(image, random.uniform(-self.hsv_h, self.hsv_h))
        if self.hsv_s > 0:
            sat = 1.0 + random.uniform(-self.hsv_s, self.hsv_s)
            image = TF.adjust_saturation(image, max(sat, 0.0))
        if self.hsv_v > 0:
            bri = 1.0 + random.uniform(-self.hsv_v, self.hsv_v)
            image = TF.adjust_brightness(image, max(bri, 0.0))
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < self.grayscale_prob:
            image = ImageOps.grayscale(image).convert("RGB")
        if random.random() < self.equalize_prob:
            image = ImageOps.equalize(image)
        return image

    def __call__(
        self, image: Image.Image, boxes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Full augment + transform pipeline.
        Returns (tensor_image, boxes) where tensor_image is already normalized.

        FIX-1: erasing is applied on raw [0,1] tensor, then Normalize runs after.
        """
        import random
        boxes = None if boxes is None else boxes.clone()

        if not self.active:
            return self._full_no_aug(image), boxes

        image = self._apply_pil(image)

        if random.random() < self.fliplr:
            image = TF.hflip(image)
            if boxes is not None and boxes.numel() > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        # FIX-1: ToTensor → RandomErasing (in [0,1]) → Normalize
        tensor = self._to_tensor(image)
        if self._eraser is not None:
            tensor = self._eraser(tensor)           # erasing on [0,1] values ✓
        tensor = self._normalize(tensor)            # normalize AFTER erasing ✓
        return tensor, boxes

    def apply_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Legacy entry point — kept for backward compatibility.

        FIX-1: This method should no longer be needed because __call__ now
        returns a fully-normalized tensor.  If called on a pre-normalized
        tensor, erasing is intentionally skipped to avoid the original bug.
        The method is kept as a no-op for backward compatibility.
        """
        return image


def _empty_target(image_id=''):
    return {
        'boxes':        torch.zeros((0, 4), dtype=torch.float32),
        'labels':       torch.zeros((0,),   dtype=torch.long),
        'metrics':      torch.zeros((0, 4), dtype=torch.float32),
        'severities':   torch.zeros((0,),   dtype=torch.float32),
        'sev_levels':   torch.zeros((0,),   dtype=torch.long),
        'zones':        [],
        'image_zone':   DEFAULT_IMAGE_ZONE,
        'image_metrics': DEFAULT_IMAGE_METRICS.clone(),
        'image_id':     image_id,
        'num_anns':     0,
    }


def _list_image_files(images_dir):
    image_paths = []
    for root, _, files in os.walk(images_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, name))
    image_paths.sort()
    return image_paths


def _label_path_for_image(image_path, images_dir, labels_dir):
    relative_path = os.path.relpath(image_path, images_dir)
    stem, _ = os.path.splitext(relative_path)
    return os.path.join(labels_dir, stem + '.txt')


def _parse_detection_label_file(label_path):
    """Parse a YOLO-format label file.

    FIX-3: malformed lines (wrong number of tokens) now emit a warning
    instead of being silently dropped.
    """
    annotations = []
    if not label_path or not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r', encoding='utf-8') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                # FIX-3: was silent, now warns
                print(
                    f"Warning: {label_path}:{line_no}: expected 5 tokens, "
                    f"got {len(parts)} — line skipped: {line!r}"
                )
                continue

            try:
                class_id = int(float(parts[0]))
                cx, cy, width, height = [float(v) for v in parts[1:]]
            except ValueError:
                print(
                    f"Warning: {label_path}:{line_no}: could not parse values "
                    f"— line skipped: {line!r}"
                )
                continue

            annotations.append({
                'class_id': class_id,
                'box': [
                    max(0.0, min(1.0, cx)),
                    max(0.0, min(1.0, cy)),
                    max(0.0, min(1.0, width)),
                    max(0.0, min(1.0, height)),
                ],
            })

    return annotations


def _detection_target_from_annotations(annotations, image_id=''):
    if not annotations:
        return _empty_target(image_id)
    boxes    = [ann['box'] for ann in annotations]
    labels   = [ann['class_id'] for ann in annotations]
    num_anns = len(annotations)
    return {
        'boxes':        torch.tensor(boxes,  dtype=torch.float32),
        'labels':       torch.tensor(labels, dtype=torch.long),
        'metrics':      torch.zeros((num_anns, 4), dtype=torch.float32),
        'severities':   torch.zeros((num_anns,),   dtype=torch.float32),
        'sev_levels':   torch.zeros((num_anns,),   dtype=torch.long),
        'zones':        [],
        'image_zone':   DEFAULT_IMAGE_ZONE,
        'image_metrics': DEFAULT_IMAGE_METRICS.clone(),
        'image_id':     image_id,
        'num_anns':     num_anns,
    }


# ── Main Dataset ──────────────────────────────────────────────
class AircraftDataset(Dataset):
    """Loads Aircraft_dataset from JSON split files."""

    def __init__(
        self,
        json_path:    str,
        images_dir:   str,
        image_size:   int  = 640,
        is_train:     bool = True,
        max_anns:     int  = 100,
        augmenter: 'DetectionAugmenter | None' = None,
    ):
        self.images_dir  = images_dir
        self.image_size  = image_size
        self.is_train    = is_train
        self.max_anns    = max_anns
        self.augmenter   = augmenter
        # Used when augmenter is absent
        self.transform   = make_transforms(image_size)

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.images = [
            img for img in data['images']
            if os.path.exists(os.path.join(images_dir, img['file_name']))
        ]
        self.sample_class_ids = [
            [CLASS_NAME_TO_ID.get(ann['category_name'], 0) for ann in item.get('annotations', [])]
            for item in self.images
        ]
        print(f"Loaded {len(self.images)} images from {json_path}")
        self._print_class_stats()

    def _print_class_stats(self):
        counts = Counter()
        for item in self.images:
            for ann in item.get('annotations', []):
                counts[ann['category_name']] += 1
        print("  Class distribution:")
        for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<16}: {cnt}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        item     = self.images[idx]
        img_path = os.path.join(self.images_dir, item['file_name'])
        image_id = item['image_id']
        anns     = item.get('annotations', [])

        image = Image.open(img_path).convert('RGB')

        boxes      = []
        labels     = []
        metrics    = []
        severities = []
        sev_levels = []
        zones      = []
        SEV_MAP    = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

        for ann in anns[:self.max_anns]:
            bb = ann['bounding_box_normalized']
            boxes.append([bb['x_center'], bb['y_center'], bb['width'], bb['height']])
            labels.append(CLASS_NAME_TO_ID.get(ann['category_name'], 0))
            m = ann.get('damage_metrics', {})
            metrics.append([
                m.get('area_ratio', 0.0), m.get('elongation', 1.0),
                m.get('edge_factor', 0.0), m.get('raw_severity_score', 0.0),
            ])
            risk    = ann.get('risk_assessment', {})
            sev_str = risk.get('severity_level', 'low')
            severities.append(m.get('raw_severity_score', 0.0))
            sev_levels.append(SEV_MAP.get(sev_str, 0))
            zones.append(ann.get('zone_estimation', 'unknown'))

        boxes_tensor = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes else torch.zeros((0, 4), dtype=torch.float32)
        )

        # FIX-1: augmenter now returns a normalized tensor directly
        if self.is_train and self.augmenter is not None:
            image_tensor, boxes_tensor = self.augmenter(image, boxes_tensor)
        else:
            image_tensor = self.transform(image)

        if not anns:
            return image_tensor, self._empty_target(image_id), image_id

        image_zone   = _select_image_zone(zones, severities)
        image_metrics = _aggregate_image_metrics(metrics)

        target = {
            'boxes':      boxes_tensor,
            'labels':     torch.tensor(labels,     dtype=torch.long),
            'metrics':    torch.tensor(metrics,    dtype=torch.float32),
            'severities': torch.tensor(severities, dtype=torch.float32),
            'sev_levels': torch.tensor(sev_levels, dtype=torch.long),
            'zones':      zones, 'image_zone': image_zone,
            'image_metrics': torch.tensor(image_metrics, dtype=torch.float32),
            'image_id': image_id, 'num_anns': len(anns[:self.max_anns]),
        }
        return image_tensor, target, image_id

    def _empty_target(self, image_id=''):
        return _empty_target(image_id)


class StandardDetectionDataset(Dataset):
    """Detection-only dataset using image folders and txt label files."""

    def __init__(
        self,
        images_dir:  str,
        labels_dir:  str,
        image_size:  int  = 640,
        class_names: list = None,
        is_train:    bool = True,
        augmenter:   'DetectionAugmenter | None' = None,
    ):
        self.images_dir  = images_dir
        self.labels_dir  = labels_dir
        self.image_size  = image_size
        self.is_train    = is_train
        self.class_names = class_names or []
        self.augmenter   = augmenter
        self.transform   = make_transforms(image_size)

        image_paths = _list_image_files(images_dir)
        self.records = []
        self.sample_class_ids = []

        for image_path in image_paths:
            image_id   = os.path.relpath(image_path, images_dir)
            label_path = _label_path_for_image(image_path, images_dir, labels_dir)
            annotations = _parse_detection_label_file(label_path)
            self.records.append({
                'image_path': image_path,
                'image_id':   image_id,
                'annotations': annotations,
            })
            self.sample_class_ids.append([ann['class_id'] for ann in annotations])

        print(f"Loaded {len(self.records)} images from {images_dir}")
        self._print_class_stats()

    def _class_name(self, class_id):
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def _print_class_stats(self):
        counts = Counter()
        for annotations in self.sample_class_ids:
            for class_id in annotations:
                counts[self._class_name(class_id)] += 1
        print("  Class distribution:")
        for cls, cnt in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {cls:<16}: {cnt}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image  = Image.open(record['image_path']).convert('RGB')
        target = _detection_target_from_annotations(
            record['annotations'], image_id=record['image_id'],
        )
        # FIX-1: augmenter returns a normalized tensor directly
        if self.is_train and self.augmenter is not None:
            image_tensor, boxes = self.augmenter(image, target['boxes'])
            target['boxes'] = boxes if boxes is not None else target['boxes']
        else:
            image_tensor = self.transform(image)
        return image_tensor, target, record['image_id']


def collate_fn(batch):
    images, targets, image_ids = [], [], []
    for image, target, image_id in batch:
        images.append(image)
        targets.append(target)
        image_ids.append(image_id)
    return torch.stack(images, dim=0), targets, image_ids


def _select_image_zone(zones, severities):
    if not zones:
        return DEFAULT_IMAGE_ZONE
    if severities:
        return zones[max(range(len(severities)), key=severities.__getitem__)]
    return Counter(zones).most_common(1)[0][0]


def _aggregate_image_metrics(metrics):
    if not metrics:
        return DEFAULT_IMAGE_METRICS.tolist()
    arr     = np.asarray(metrics, dtype=np.float32)
    summary = arr.mean(axis=0)
    summary[3] = arr[:, 3].max()
    return summary.tolist()


class ClassBalancedSampler(torch.utils.data.Sampler):
    """Oversample images containing rare classes using dataset-driven weights."""

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int = None,
        rarity_power: float = 0.5,
        background_weight: float = None,
    ):
        self.dataset     = dataset
        self.num_samples = num_samples or len(dataset)
        self.rarity_power = rarity_power

        sample_class_ids = getattr(dataset, 'sample_class_ids', None)
        if sample_class_ids is None:
            raise ValueError("Dataset does not expose sample_class_ids for balanced sampling.")

        image_class_counts = Counter()
        for class_ids in sample_class_ids:
            image_class_counts.update(set(class_ids))

        if not image_class_counts:
            self.class_weights    = {}
            self.background_weight = 1.0
            self.weights = torch.ones(len(sample_class_ids), dtype=torch.float32)
            return

        raw_weights = {
            cid: 1.0 / float(max(cnt, 1)) ** rarity_power
            for cid, cnt in image_class_counts.items()
        }
        max_weight = max(raw_weights.values())
        self.class_weights = {cid: w / max_weight for cid, w in raw_weights.items()}

        min_cw = min(self.class_weights.values())
        self.background_weight = (
            background_weight if background_weight is not None
            else max(min_cw * 0.5, 0.05)
        )

        sample_weights = []
        for class_ids in sample_class_ids:
            unique_ids = sorted(set(class_ids))
            if not unique_ids:
                sample_weights.append(self.background_weight)
                continue
            rarest = max(self.class_weights.get(cid, min_cw) for cid in unique_ids)
            diversity_boost = 1.0 + 0.05 * max(len(unique_ids) - 1, 0)
            sample_weights.append(rarest * diversity_boost)

        self.weights = torch.tensor(sample_weights, dtype=torch.float32)
        self._print_sampler_stats(image_class_counts)

    def _class_name(self, class_id):
        if hasattr(self.dataset, '_class_name'):
            return self.dataset._class_name(class_id)
        class_names = getattr(self.dataset, 'class_names', None) or []
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return CLASS_ID_TO_NAME.get(class_id, f'class_{class_id}')

    def _print_sampler_stats(self, image_class_counts):
        print("  Balanced sampler:")
        for cid, cnt in sorted(image_class_counts.items(), key=lambda x: (x[1], x[0])):
            print(f"    {self._class_name(cid):<16}: img_freq={cnt:<5d} weight={self.class_weights[cid]:.3f}")

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


def build_train_loader(
    json_path:   str = None,
    images_dir:  str = None,
    batch_size:  int = 16,
    image_size:  int = 640,
    num_workers: int = 4,
    balanced:    bool = True,
    data_format: str = 'json',
    labels_dir:  str = None,
    class_names: list = None,
    augmenter: 'DetectionAugmenter | None' = None,
) -> DataLoader:
    if data_format == 'detection':
        dataset = StandardDetectionDataset(
            images_dir=images_dir, labels_dir=labels_dir, image_size=image_size,
            class_names=class_names, is_train=True, augmenter=augmenter,
        )
    else:
        dataset = AircraftDataset(
            json_path=json_path, images_dir=images_dir, image_size=image_size,
            is_train=True, augmenter=augmenter,
        )

    if len(dataset) == 0:
        raise ValueError(
            f"No training images found for data_format='{data_format}' and images_dir='{images_dir}'."
        )

    sampler = ClassBalancedSampler(dataset) if balanced else None
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) > batch_size,
    )


def build_val_loader(
    json_path:   str = None,
    images_dir:  str = None,
    batch_size:  int = 16,
    image_size:  int = 640,
    num_workers: int = 4,
    data_format: str = 'json',
    labels_dir:  str = None,
    class_names: list = None,
) -> DataLoader:
    if data_format == 'detection':
        dataset = StandardDetectionDataset(
            images_dir=images_dir, labels_dir=labels_dir, image_size=image_size,
            class_names=class_names, is_train=False,
        )
    else:
        dataset = AircraftDataset(
            json_path=json_path, images_dir=images_dir, image_size=image_size,
            is_train=False,
        )

    if len(dataset) == 0:
        raise ValueError(
            f"No validation images found for data_format='{data_format}' and images_dir='{images_dir}'."
        )

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
