"""Runtime helpers for configuration and dataset path resolution.

Changes vs original
-------------------
FIX-1  infer_label_dir: parts.index('images') only finds the FIRST occurrence.
       A path like /data/images/train/images/ would replace the wrong segment.
       Fixed by scanning from the right (rindex equivalent via reversed search)
       so the innermost 'images' component is replaced — which is always the
       one that separates image files from their labels.

FIX-2  load_yaml_config: previously returned {} silently when the file did not
       exist, giving no indication to the user that their --config argument was
       ignored.  Now prints a clear warning so the user knows defaults are
       being used instead of their config.

FIX-3  resolve_detection_paths: if resolved_images is still None after all
       resolution logic, an early ValueError is raised here rather than letting
       a cryptic AttributeError surface inside the dataloader later.
"""

from __future__ import annotations

import os


LEGACY_DATASET_ROOT = (
    r'C:\Users\tsake\OneDrive\Desktop\Aircraft_dataset\content\Aircraft_dataset'
)

DEFAULT_JSON_FILES = {
    'train': 'Aircraft_train.json',
    'val':   'Aircraft_val.json',
    'test':  'Aircraft_test.json',
}

DEFAULT_IMAGE_DIRS = {
    'train': os.path.join('images', 'train'),
    'val':   os.path.join('images', 'val'),
    'test':  os.path.join('images', 'test'),
}


def coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def parse_int_tuple(path_value, field_name="value", expected_len=None):
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
    """Load a YAML config file.

    FIX-2: if path is provided but the file does not exist, print a clear
    warning instead of silently returning {}.  Callers that pass a real path
    expect it to be loaded; silent failure hides misconfiguration.
    """
    if not path:
        return {}

    if not os.path.exists(path):
        # FIX-2: was silent — now warns explicitly
        print(
            f"Warning: config file not found: '{path}'. "
            "Falling back to defaults — check your --config argument."
        )
        return {}

    try:
        import yaml
    except ImportError:
        print(f"PyYAML not installed — ignoring config file: {path}")
        return {}

    with open(path, 'r', encoding='utf-8') as handle:
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


def default_dataset_root() -> str:
    return os.getenv('SLIM_DET_DATASET_ROOT', LEGACY_DATASET_ROOT)


def resolve_path(path_value: str | None, base_dir: str | None = None) -> str | None:
    if path_value is None:
        return None
    if os.path.isabs(path_value) or base_dir is None:
        return path_value
    return os.path.join(base_dir, path_value)


def infer_label_dir(images_dir: str | None) -> str | None:
    """Derive a labels directory from an images directory.

    FIX-1: original used parts.index('images') which finds the FIRST
    occurrence of 'images' in the path.  For a path such as
        /data/images/train/images/
    this would replace the top-level 'images' instead of the inner one,
    producing /data/labels/train/images/ — wrong.

    Fix: scan from the RIGHT so the innermost (most specific) 'images'
    segment is replaced, which is always the correct one.
    """
    if images_dir is None:
        return None

    norm_path = os.path.normpath(images_dir)
    drive, tail = os.path.splitdrive(norm_path)
    parts = [part for part in tail.split(os.sep) if part]

    # FIX-1: find the LAST occurrence of 'images', not the first
    try:
        idx = len(parts) - 1 - parts[::-1].index('images')
    except ValueError:
        idx = -1

    if idx >= 0:
        parts[idx] = 'labels'
        return drive + os.sep + os.path.join(*parts)

    # Fallback: place a 'labels/<split>' sibling next to the images dir
    parent = os.path.dirname(norm_path)
    split_name = os.path.basename(norm_path)
    return os.path.join(parent, 'labels', split_name)


def resolve_dataset_paths(
    dataset_root=None,
    data_config=None,
    split='train',
    json_path=None,
    images_dir=None,
):
    data_config  = data_config or {}
    dataset_root = coalesce(
        dataset_root,
        data_config.get('dataset_root'),
        default_dataset_root(),
    )

    resolved_json = resolve_path(
        coalesce(json_path, data_config.get(f'{split}_json'), DEFAULT_JSON_FILES[split]),
        dataset_root,
    )
    resolved_images = resolve_path(
        coalesce(images_dir, data_config.get(f'{split}_images'), DEFAULT_IMAGE_DIRS[split]),
        dataset_root,
    )
    return dataset_root, resolved_json, resolved_images


def resolve_detection_paths(
    config_path=None,
    dataset_root=None,
    split='train',
    images_dir=None,
    labels_dir=None,
) -> dict:
    """Resolve all dataset paths for detection-format datasets.

    FIX-3: if resolved_images is still None after all resolution attempts,
    raise a ValueError immediately instead of returning a dict with None that
    causes an obscure crash deep inside the dataloader.
    """
    config      = load_yaml_config(config_path) if config_path else {}
    config_root = os.path.dirname(os.path.abspath(config_path)) if config_path else None

    resolved_root = coalesce(dataset_root, config.get('path'), config.get('dataset_root'))
    if resolved_root is not None and not os.path.isabs(resolved_root):
        candidates = []
        if config_root is not None:
            candidates.append(resolve_path(resolved_root, config_root))
        candidates.append(os.path.abspath(resolved_root))
        resolved_root = next(
            (p for p in candidates if os.path.exists(p)), candidates[0]
        )

    split_images = coalesce(images_dir, config.get(split), config.get(f'{split}_images'))
    if split_images is None and resolved_root is not None:
        split_images = os.path.join('images', split)
    resolved_images = resolve_path(split_images, resolved_root or config_root)

    # FIX-3: fail early with a clear message instead of returning None silently
    if resolved_images is None:
        raise ValueError(
            f"Could not resolve images directory for split='{split}'. "
            "Pass --dataset_root, --data_config, or set SLIM_DET_DATASET_ROOT."
        )

    split_labels = coalesce(labels_dir, config.get(f'{split}_labels'))
    if split_labels is not None:
        resolved_labels = resolve_path(split_labels, resolved_root or config_root)
    elif resolved_root is not None:
        resolved_labels = os.path.join(resolved_root, 'labels', split)
    else:
        resolved_labels = infer_label_dir(resolved_images)

    class_names = normalize_class_names(config.get('names'))
    num_classes = config.get('nc')
    if num_classes is None and class_names is not None:
        num_classes = len(class_names)

    return {
        'dataset_root': resolved_root or config_root,
        'images_dir':   resolved_images,
        'labels_dir':   resolved_labels,
        'class_names':  class_names,
        'num_classes':  num_classes,
    }


def require_existing_paths(**paths) -> None:
    missing = [
        f"{name}='{path}'"
        for name, path in paths.items()
        if path is not None and not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing required path(s): " + ", ".join(missing))
