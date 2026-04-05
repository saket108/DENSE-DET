# data/prompt_builder.py
"""
Converts Aircraft_dataset JSON annotation fields into
rich text prompts for SLIM-Det's text encoder.

Novelty: uses ALL structured fields — not just category name.
Other detectors only use class label. We use zone + severity
+ description + metrics to build a semantically rich prompt.
"""

# ── Zone vocabulary ───────────────────────────────────────────
ZONE_DESCRIPTIONS = {
    'central':       'central inspection region',
    'top_left':      'upper-left inspection zone',
    'top_right':     'upper-right inspection zone',
    'bottom_left':   'lower-left inspection zone',
    'bottom_right':  'lower-right inspection zone',
    'left_edge':     'left edge boundary',
    'right_edge':    'right edge boundary',
    'top_edge':      'upper edge boundary',
    'bottom_edge':   'lower edge boundary',
    'unknown':       'unspecified inspection zone',
}

# ── Severity vocabulary ───────────────────────────────────────
SEVERITY_PHRASES = {
    'low':      'low severity, monitoring recommended',
    'medium':   'moderate severity, maintenance required',
    'high':     'high severity, immediate attention required',
    'critical': 'critical severity, grounding recommended',
}

# ── Damage class descriptions ─────────────────────────────────
CLASS_CONTEXT = {
    'crack':          'structural crack or fracture line',
    'dent':           'surface deformation or dent',
    'corrosion':      'corrosion or oxidation damage',
    'scratch':        'surface scratch or abrasion',
    'missing-head':   'missing fastener or rivet head',
    'paint-peel-off': 'paint delamination or peeling',
}

# ── Class ID mapping ──────────────────────────────────────────
CLASS_ID_TO_NAME = {
    0: 'crack',
    1: 'dent',
    2: 'corrosion',
    3: 'scratch',
    4: 'missing-head',
    5: 'paint-peel-off',
}

CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}


def build_prompt(annotation: dict, mode: str = 'full') -> str:
    """
    Build a text prompt from a single annotation dict.

    Args:
        annotation: one annotation entry from Aircraft JSON
        mode:
            'full'     — all fields (best for training)
            'minimal'  — category + zone only (ablation)
            'no_desc'  — everything except free-text description
            'cat_only' — category name only (YOLO-equivalent baseline)

    Returns:
        prompt string ready for tokenization

    Example output (full mode):
        "Damage type: structural crack or fracture line.
         Location: central inspection region.
         Severity: low severity, monitoring recommended.
         Damage characteristics: area ratio 0.055,
         elongation 1.16, edge factor 0.524.
         Low severity localized surface deformation..."
    """
    cat  = annotation.get('category_name', 'unknown')
    zone = annotation.get('zone_estimation', 'unknown')
    desc = annotation.get('description', '')
    risk = annotation.get('risk_assessment', {})
    sev  = risk.get('severity_level', 'low')
    mets = annotation.get('damage_metrics', {})

    # ── cat_only — YOLO-equivalent (ablation baseline) ────────
    if mode == 'cat_only':
        return f"Damage type: {cat}."

    # ── minimal — category + zone only ────────────────────────
    if mode == 'minimal':
        zone_text = ZONE_DESCRIPTIONS.get(zone, zone)
        cat_text  = CLASS_CONTEXT.get(cat, cat)
        return f"Damage type: {cat_text}. Location: {zone_text}."

    # ── Build full / no_desc prompt ───────────────────────────
    parts = []

    # 1. Damage type — always included
    cat_text = CLASS_CONTEXT.get(cat, cat)
    parts.append(f"Damage type: {cat_text}.")

    # 2. Zone location
    zone_text = ZONE_DESCRIPTIONS.get(zone, zone)
    parts.append(f"Location: {zone_text}.")

    # 3. Severity
    sev_text = SEVERITY_PHRASES.get(sev, sev)
    parts.append(f"Severity: {sev_text}.")

    # 4. Numeric damage metrics — humanized
    if mets:
        area    = mets.get('area_ratio', 0)
        elong   = mets.get('elongation', 1)
        edge    = mets.get('edge_factor', 0)
        raw_sev = mets.get('raw_severity_score', 0)
        parts.append(
            f"Damage characteristics: "
            f"area ratio {area:.3f}, "
            f"elongation {elong:.2f}, "
            f"edge factor {edge:.3f}, "
            f"severity score {raw_sev:.3f}."
        )

    # 5. Free-text description — skip in no_desc mode
    if mode != 'no_desc' and desc:
        parts.append(desc.strip())

    return ' '.join(parts)


def build_image_prompt(image_entry: dict, mode: str = 'full') -> list:
    """
    Build prompts for ALL annotations in one image entry.

    Args:
        image_entry: one image dict from Aircraft JSON
        mode: passed to build_prompt

    Returns:
        list of (annotation_id, prompt_str) tuples
    """
    results = []
    for ann in image_entry.get('annotations', []):
        prompt = build_prompt(ann, mode=mode)
        results.append((ann['annotation_id'], prompt))
    return results


def build_batch_prompts(annotations: list, mode: str = 'full') -> list:
    """
    Build prompts for a batch of annotations.
    Used inside the DataLoader __getitem__.

    Args:
        annotations: list of annotation dicts (one image)
        mode: prompt mode

    Returns:
        list of prompt strings, one per annotation
    """
    return [build_prompt(ann, mode=mode) for ann in annotations]


def get_image_level_prompt(annotations: list, mode: str = 'full') -> str:
    """
    Build a single image-level prompt summarizing all damage
    in the image. Used when you want one prompt per image
    rather than one per annotation.

    Args:
        annotations: all annotations for one image
        mode: prompt mode

    Returns:
        combined prompt string
    """
    if not annotations:
        return "No damage detected in this inspection image."

    # Get unique damage types in this image
    cats = list({ann.get('category_name', 'unknown') for ann in annotations})
    cat_texts = [CLASS_CONTEXT.get(c, c) for c in cats]

    if mode == 'cat_only':
        return f"Damage types present: {', '.join(cats)}."

    # Get most severe annotation
    sev_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
    most_severe = max(
        annotations,
        key=lambda a: sev_order.get(
            a.get('risk_assessment', {}).get('severity_level', 'low'), 0
        )
    )
    top_sev = most_severe.get('risk_assessment', {}).get('severity_level', 'low')

    parts = [
        f"Aircraft inspection image with {len(annotations)} damage instance(s).",
        f"Damage types: {', '.join(cat_texts)}.",
        f"Highest severity: {SEVERITY_PHRASES.get(top_sev, top_sev)}.",
    ]
    return ' '.join(parts)


# ── Ablation mode registry ────────────────────────────────────
PROMPT_MODES = {
    'full':     'All fields: type + zone + severity + metrics + description',
    'no_desc':  'Structured only: type + zone + severity + metrics',
    'minimal':  'Sparse: type + zone only',
    'cat_only': 'Category name only (YOLO-equivalent ablation)',
}


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    sample_annotation = {
        'annotation_id':   'aircraft_012299_0',
        'category_id':     1,
        'category_name':   'dent',
        'zone_estimation': 'central',
        'damage_metrics': {
            'area_ratio':         0.05464,
            'elongation':         1.158,
            'edge_factor':        0.52422,
            'raw_severity_score': 0.08913,
        },
        'risk_assessment': {
            'severity_level':            'low',
            'requires_manual_validation': True,
        },
        'description': (
            'Low severity localized surface deformation identified '
            'within the central inspection region. The affected region '
            'demonstrates relatively extensive surface involvement. '
            'No visible fracture lines detected within the annotated '
            'boundary. Depth and material stress redistribution require '
            'physical inspection.'
        ),
    }

    print("=" * 70)
    for mode, desc in PROMPT_MODES.items():
        print(f"\n── mode: {mode} — {desc}")
        print(build_prompt(sample_annotation, mode=mode))
    print("\n" + "=" * 70)
    print("\n── image-level prompt (full) ──")
    print(get_image_level_prompt([sample_annotation], mode='full'))
    print("=" * 70)
