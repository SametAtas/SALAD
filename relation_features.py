#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, label
from sklearn.covariance import LedoitWolf


DEFAULT_CLASSES = (1, 2, 3, 4, 5)
DEFAULT_MIN_COMPONENT_AREA = 200
STRIP_SUFFIXES = ("_refined_seg", "_gt", "_mask")
EPS = 1e-12


def parse_classes(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def canonical_stem(path):
    stem = Path(str(path)).stem
    changed = True
    while changed:
        changed = False
        for suffix in STRIP_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def defect_class_from_path(path):
    return Path(str(path)).parent.name


def sample_key(path, defect_class=None):
    if defect_class is None or pd.isna(defect_class):
        defect_class = defect_class_from_path(path)
    return f"{defect_class}/{canonical_stem(path)}"


def load_label_map(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


def component_masks(mask, min_component_area):
    labeled, num_features = label(mask)
    components = []
    for component_id in range(1, num_features + 1):
        component = labeled == component_id
        area = int(component.sum())
        if area >= min_component_area:
            components.append(component)
    return components


def filtered_class_mask(arr, class_id, min_component_area):
    components = component_masks(arr == class_id, min_component_area)
    if not components:
        return np.zeros(arr.shape, dtype=bool), components
    filtered = np.logical_or.reduce(components)
    return filtered, components


def class_geometry_features(mask, components, height, width):
    total_pixels = float(height * width)
    area = float(mask.sum())
    present = 1.0 if area > 0 else 0.0
    count = float(len(components))
    largest_area = float(max((component.sum() for component in components), default=0))

    if not present:
        return {
            "present": 0.0,
            "area_ratio": 0.0,
            "component_count": 0.0,
            "largest_area_ratio": 0.0,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "compactness": 0.0,
        }

    ys, xs = np.nonzero(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox_width_pixels = x_max - x_min + 1
    bbox_height_pixels = y_max - y_min + 1
    bbox_area = float(bbox_width_pixels * bbox_height_pixels)

    return {
        "present": present,
        "area_ratio": area / total_pixels,
        "component_count": count,
        "largest_area_ratio": largest_area / total_pixels,
        "centroid_x": float(xs.mean()) / max(width - 1, 1),
        "centroid_y": float(ys.mean()) / max(height - 1, 1),
        "bbox_width": bbox_width_pixels / float(width),
        "bbox_height": bbox_height_pixels / float(height),
        "compactness": area / max(bbox_area, 1.0),
    }


def contact_feature(mask_a, mask_b):
    if not mask_a.any() or not mask_b.any():
        return 0.0
    dilated_a = binary_dilation(mask_a, structure=np.ones((3, 3), dtype=bool))
    return float(np.logical_and(dilated_a, mask_b).any())


def extract_relation_features(arr, foreground_classes=DEFAULT_CLASSES, min_component_area=DEFAULT_MIN_COMPONENT_AREA):
    height, width = arr.shape[:2]
    row = {}
    class_masks = {}
    class_features = {}

    for class_id in foreground_classes:
        mask, components = filtered_class_mask(arr, class_id, min_component_area)
        class_masks[class_id] = mask
        features = class_geometry_features(mask, components, height, width)
        class_features[class_id] = features
        prefix = f"rel_c{class_id}"
        row[f"{prefix}_present"] = features["present"]
        row[f"{prefix}_area_ratio"] = features["area_ratio"]
        row[f"{prefix}_component_count"] = features["component_count"]
        row[f"{prefix}_largest_area_ratio"] = features["largest_area_ratio"]
        row[f"{prefix}_centroid_x"] = features["centroid_x"]
        row[f"{prefix}_centroid_y"] = features["centroid_y"]
        row[f"{prefix}_bbox_width"] = features["bbox_width"]
        row[f"{prefix}_bbox_height"] = features["bbox_height"]
        row[f"{prefix}_compactness"] = features["compactness"]

    for class_a, class_b in itertools.combinations(foreground_classes, 2):
        features_a = class_features[class_a]
        features_b = class_features[class_b]
        both_present = features_a["present"] and features_b["present"]
        dx = features_b["centroid_x"] - features_a["centroid_x"] if both_present else 0.0
        dy = features_b["centroid_y"] - features_a["centroid_y"] if both_present else 0.0
        dist = float(np.sqrt(dx * dx + dy * dy)) if both_present else 0.0
        prefix = f"rel_c{class_a}_c{class_b}"
        row[f"{prefix}_centroid_dx"] = dx
        row[f"{prefix}_centroid_dy"] = dy
        row[f"{prefix}_centroid_distance"] = dist
        row[f"{prefix}_contact"] = contact_feature(class_masks[class_a], class_masks[class_b])

    return row


def collect_map_paths(seg_root, category, split, defect_class=None):
    split_dir = Path(seg_root) / category / split
    if defect_class:
        split_dir = split_dir / defect_class
    return sorted(split_dir.rglob("*.png"))


def feature_frame(paths, foreground_classes, min_component_area):
    rows = []
    for path in paths:
        defect_class = defect_class_from_path(path)
        arr = load_label_map(path)
        features = extract_relation_features(arr, foreground_classes, min_component_area)
        rows.append(
            {
                "Sample_Key": sample_key(path, defect_class),
                "Image_Path": str(path),
                "Defect_Class": defect_class,
                "Ground_Truth": 0 if defect_class == "good" else 1,
                **features,
            }
        )
    if not rows:
        raise ValueError("No composition-map PNGs found for the requested input.")
    return pd.DataFrame(rows)


def feature_columns(df):
    return [column for column in df.columns if column.startswith("rel_")]


def fit_relation_model(df):
    columns = feature_columns(df)
    if not columns:
        raise ValueError("No relation feature columns found.")
    values = df.loc[:, columns].astype(float).to_numpy()
    mean = values.mean(axis=0)
    cov = LedoitWolf().fit(values).covariance_
    covinv = np.linalg.pinv(cov)
    return columns, mean, covinv


def score_feature_values(values, mean, covinv):
    deltas = values - mean
    distances = np.einsum("ij,jk,ik->i", deltas, covinv, deltas)
    distances = np.maximum(distances, 0.0)
    return np.sqrt(distances)


def save_model(path, feature_names, mean, covinv, foreground_classes, min_component_area, fit_paths):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        feature_names=np.array(feature_names),
        mean=mean,
        covinv=covinv,
        foreground_classes=np.array(foreground_classes, dtype=np.int32),
        min_component_area=np.array([min_component_area], dtype=np.int32),
        fit_paths=np.array([str(path) for path in fit_paths]),
    )


def load_model(path):
    model = np.load(path, allow_pickle=True)
    return {
        "feature_names": [str(name) for name in model["feature_names"].tolist()],
        "mean": model["mean"],
        "covinv": model["covinv"],
        "foreground_classes": tuple(int(value) for value in model["foreground_classes"].tolist()),
        "min_component_area": int(model["min_component_area"][0]),
    }


def score_frame(df, model):
    columns = model["feature_names"]
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Input features are missing model columns: {missing}")
    values = df.loc[:, columns].astype(float).to_numpy()
    df = df.copy()
    df.insert(4, "Relation_Score", score_feature_values(values, model["mean"], model["covinv"]))
    return df


def fit_command(args):
    train_paths = collect_map_paths(args.seg_root, args.category, "train", "good")
    fit_paths = list(train_paths)
    if args.include_validation_good:
        fit_paths.extend(collect_map_paths(args.seg_root, args.category, "validation", "good"))
    if not fit_paths:
        raise ValueError("No train/good composition maps found.")

    foreground_classes = parse_classes(args.foreground_classes)
    df = feature_frame(fit_paths, foreground_classes, args.min_component_area)
    columns, mean, covinv = fit_relation_model(df)
    save_model(args.output, columns, mean, covinv, foreground_classes, args.min_component_area, fit_paths)

    metadata = {
        "category": args.category,
        "seg_root": args.seg_root,
        "num_fit_maps": len(fit_paths),
        "num_features": len(columns),
        "foreground_classes": list(foreground_classes),
        "min_component_area": args.min_component_area,
        "include_validation_good": args.include_validation_good,
    }
    print(json.dumps(metadata, indent=2))
    print(f"Saved relation model to {args.output}")


def score_command(args):
    model = load_model(args.model)
    paths = collect_map_paths(args.seg_root, args.category, args.split)
    df = feature_frame(paths, model["foreground_classes"], model["min_component_area"])
    df = score_frame(df, model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Scored {len(df)} maps and saved {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and score relation-layout features from SALAD composition maps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit normal relation model from train/good maps.")
    fit_parser.add_argument("--category", required=True)
    fit_parser.add_argument("--seg_root", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--foreground_classes", default=",".join(str(value) for value in DEFAULT_CLASSES))
    fit_parser.add_argument("--min_component_area", type=int, default=DEFAULT_MIN_COMPONENT_AREA)
    fit_parser.add_argument("--include_validation_good", action="store_true")
    fit_parser.set_defaults(func=fit_command)

    score_parser = subparsers.add_parser("score", help="Score maps with a fitted relation model.")
    score_parser.add_argument("--category", required=True)
    score_parser.add_argument("--seg_root", required=True)
    score_parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.set_defaults(func=score_command)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
