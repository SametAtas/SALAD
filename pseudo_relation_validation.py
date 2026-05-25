#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label

from relation_features import (
    canonical_stem,
    extract_relation_features,
    load_label_map,
    load_model,
    sample_key,
    score_frame,
)


def component_entries(arr, foreground_classes, min_component_area):
    entries = []
    for class_id in foreground_classes:
        labeled, num_features = label(arr == class_id)
        for component_id in range(1, num_features + 1):
            mask = labeled == component_id
            area = int(mask.sum())
            if area >= min_component_area:
                entries.append((class_id, mask, area))
    return entries


def choose_component(arr, rng, foreground_classes, min_component_area):
    entries = component_entries(arr, foreground_classes, min_component_area)
    if not entries:
        return None
    return rng.choice(entries)


def shift_mask(mask, dy, dx):
    shifted = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    if src_y0 < src_y1 and src_x0 < src_x1:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return shifted


def resize_component_mask(mask, scale):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    height, width = mask.shape
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    new_h = max(1, int(round(crop.shape[0] * scale)))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    resized = np.array(Image.fromarray(crop).resize((new_w, new_h), Image.NEAREST)) > 0

    center_y = int(round((y0 + y1 - 1) / 2.0))
    center_x = int(round((x0 + x1 - 1) / 2.0))
    out = np.zeros_like(mask, dtype=bool)
    dst_y0 = center_y - new_h // 2
    dst_x0 = center_x - new_w // 2
    src_y0 = max(0, -dst_y0)
    src_x0 = max(0, -dst_x0)
    dst_y0 = max(0, dst_y0)
    dst_x0 = max(0, dst_x0)
    dst_y1 = min(height, dst_y0 + new_h - src_y0)
    dst_x1 = min(width, dst_x0 + new_w - src_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    if dst_y0 < dst_y1 and dst_x0 < dst_x1:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1]
    return out


def corrupt_remove(arr, rng, foreground_classes, min_component_area, all_maps):
    entry = choose_component(arr, rng, foreground_classes, min_component_area)
    if entry is None:
        return arr.copy()
    _, mask, _ = entry
    out = arr.copy()
    out[mask] = 0
    return out


def corrupt_change_label(arr, rng, foreground_classes, min_component_area, all_maps):
    entry = choose_component(arr, rng, foreground_classes, min_component_area)
    if entry is None:
        return arr.copy()
    class_id, mask, _ = entry
    other_classes = [class_value for class_value in foreground_classes if class_value != class_id]
    out = arr.copy()
    out[mask] = rng.choice(other_classes)
    return out


def corrupt_shift(arr, rng, foreground_classes, min_component_area, all_maps):
    entry = choose_component(arr, rng, foreground_classes, min_component_area)
    if entry is None:
        return arr.copy()
    class_id, mask, _ = entry
    height, width = arr.shape
    dy = rng.randint(-height // 5, height // 5)
    dx = rng.randint(-width // 5, width // 5)
    if dy == 0 and dx == 0:
        dx = max(1, width // 10)
    shifted = shift_mask(mask, dy, dx)
    out = arr.copy()
    out[mask] = 0
    out[shifted] = class_id
    return out


def corrupt_duplicate(arr, rng, foreground_classes, min_component_area, all_maps):
    source = rng.choice(all_maps)
    entry = choose_component(source, rng, foreground_classes, min_component_area)
    if entry is None:
        return arr.copy()
    class_id, mask, _ = entry
    height, width = arr.shape
    dy = rng.randint(-height // 3, height // 3)
    dx = rng.randint(-width // 3, width // 3)
    shifted = shift_mask(mask, dy, dx)
    out = arr.copy()
    out[shifted] = class_id
    return out


def corrupt_scale(arr, rng, foreground_classes, min_component_area, all_maps):
    entry = choose_component(arr, rng, foreground_classes, min_component_area)
    if entry is None:
        return arr.copy()
    class_id, mask, _ = entry
    scaled = resize_component_mask(mask, rng.choice((0.6, 1.4)))
    out = arr.copy()
    out[mask] = 0
    out[scaled] = class_id
    return out


CORRUPTIONS = {
    "remove_component": corrupt_remove,
    "change_label": corrupt_change_label,
    "shift_component": corrupt_shift,
    "duplicate_component": corrupt_duplicate,
    "scale_component": corrupt_scale,
}


def row_from_array(arr, path, sample_key_value, defect_class, ground_truth, corruption, model):
    features = extract_relation_features(arr, model["foreground_classes"], model["min_component_area"])
    return {
        "Sample_Key": sample_key_value,
        "Image_Path": str(path),
        "Defect_Class": defect_class,
        "Ground_Truth": int(ground_truth),
        "Pseudo_Corruption": corruption,
        **features,
    }


def main():
    parser = argparse.ArgumentParser(description="Create pseudo logical-anomaly validation data from validation/good maps.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--seg_root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pseudo_per_map", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = load_model(args.model)
    validation_dir = Path(args.seg_root) / args.category / "validation" / "good"
    paths = sorted(validation_dir.rglob("*.png"))
    if not paths:
        raise ValueError(f"No validation/good maps found under {validation_dir}")

    all_maps = [load_label_map(path) for path in paths]
    rows = []
    for path, arr in zip(paths, all_maps):
        base_key = sample_key(path, "good")
        rows.append(row_from_array(arr, path, base_key, "good", 0, "none", model))
        for pseudo_idx in range(args.pseudo_per_map):
            corruption_name = rng.choice(sorted(CORRUPTIONS))
            corrupted = CORRUPTIONS[corruption_name](
                arr,
                rng,
                model["foreground_classes"],
                model["min_component_area"],
                all_maps,
            )
            pseudo_key = f"pseudo_{corruption_name}/{canonical_stem(path)}_{pseudo_idx:02d}"
            rows.append(row_from_array(corrupted, path, pseudo_key, f"pseudo_{corruption_name}", 1, corruption_name, model))

    df = pd.DataFrame(rows)
    df = score_frame(df, model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} pseudo-validation relation rows to {output}")


if __name__ == "__main__":
    main()
