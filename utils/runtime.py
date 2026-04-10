"""Runtime helpers for configuration and dataset path resolution."""

from __future__ import annotations

import os


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def parse_int_tuple(path_value, field_name: str = "value", expected_len: int | None = None):
    if path_value is None:
        return None

    if isinstance(path_value, str):
        items = [item.strip() for item in path_value.split(",") if item.strip()]
    elif isinstance(path_value, (list, tuple)):
        items = list(path_value)
    else:
        raise TypeError(
            f"{field_name} must be a comma-separated string, list, or tuple, "
            f"got {type(path_value).__name__}."
        )

    values = tuple(int(item) for item in items)
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(
            f"{field_name} must contain exactly {expected_len} integers, got {len(values)}."
        )
    return values


def load_yaml_config(path: str | None) -> dict:
    if not path:
        return {}

    if not os.path.exists(path):
        print(f"Warning: config file not found: '{path}'. Falling back to defaults.")
        return {}

    try:
        import yaml
    except ImportError:
        print(f"PyYAML not installed, ignoring config file: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_class_names(raw_names):
    if raw_names is None:
        return None
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        ordered = sorted(
            raw_names.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
        )
        return [str(name) for _, name in ordered]
    return None


def resolve_path(path_value: str | None, base_dir: str | None = None) -> str | None:
    if path_value is None:
        return None
    if os.path.isabs(path_value) or base_dir is None:
        return path_value
    return os.path.join(base_dir, path_value)


def infer_label_dir(images_dir: str | None) -> str | None:
    if images_dir is None:
        return None

    norm_path = os.path.normpath(images_dir)
    drive, tail = os.path.splitdrive(norm_path)
    parts = [part for part in tail.split(os.sep) if part]
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        index = -1

    if index >= 0:
        parts[index] = "labels"
        return drive + os.sep + os.path.join(*parts)

    parent = os.path.dirname(norm_path)
    split_name = os.path.basename(norm_path)
    return os.path.join(parent, "labels", split_name)


def resolve_detection_paths(
    config_path: str | None = None,
    dataset_root: str | None = None,
    split: str = "train",
    images_dir: str | None = None,
    labels_dir: str | None = None,
) -> dict:
    config = load_yaml_config(config_path) if config_path else {}
    config_root = os.path.dirname(os.path.abspath(config_path)) if config_path else None

    resolved_root = coalesce(dataset_root, config.get("path"), config.get("dataset_root"))
    if resolved_root is not None and not os.path.isabs(resolved_root):
        candidates: list[str] = []
        if config_root is not None:
            candidates.append(resolve_path(resolved_root, config_root))
        candidates.append(os.path.abspath(resolved_root))
        resolved_root = next((path for path in candidates if os.path.exists(path)), candidates[0])

    split_images = coalesce(images_dir, config.get(split), config.get(f"{split}_images"))
    if split_images is None and resolved_root is not None:
        split_images = os.path.join("images", split)
    resolved_images = resolve_path(split_images, resolved_root or config_root)
    if resolved_images is None:
        raise ValueError(
            f"Could not resolve images directory for split='{split}'. "
            "Pass --dataset_root or --data_config."
        )

    split_labels = coalesce(labels_dir, config.get(f"{split}_labels"))
    if split_labels is not None:
        resolved_labels = resolve_path(split_labels, resolved_root or config_root)
    elif resolved_root is not None:
        resolved_labels = os.path.join(resolved_root, "labels", split)
    else:
        resolved_labels = infer_label_dir(resolved_images)

    class_names = normalize_class_names(config.get("names"))
    num_classes = config.get("nc")
    if num_classes is None and class_names is not None:
        num_classes = len(class_names)

    return {
        "dataset_root": resolved_root or config_root,
        "images_dir": resolved_images,
        "labels_dir": resolved_labels,
        "class_names": class_names,
        "num_classes": num_classes,
    }


def require_existing_paths(**paths) -> None:
    missing = [
        f"{name}='{path}'"
        for name, path in paths.items()
        if path is not None and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))
