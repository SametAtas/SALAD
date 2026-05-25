#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from relation_features import sample_key


BRANCH_COLUMNS = ("Appearance_Score", "Global_Score", "Composition_Score")
RELATION_COLUMN = "Relation_Score"
DINO_COLUMN = "DINO_Memory_Score"
EPS = 1e-8


def add_sample_key(df):
    if "Sample_Key" in df.columns:
        df["Sample_Key"] = df["Sample_Key"].astype(str)
        return df
    if "Image_Path" not in df.columns:
        raise ValueError("CSV must contain Sample_Key or Image_Path.")
    if "Defect_Class" in df.columns:
        df["Sample_Key"] = [sample_key(path, defect_class) for path, defect_class in zip(df["Image_Path"], df["Defect_Class"])]
    else:
        df["Sample_Key"] = [sample_key(path) for path in df["Image_Path"]]
    return df


def read_scores(path):
    df = pd.read_csv(path).copy()
    return add_sample_key(df)


def merge_relation(df, relation_csv):
    relation_df = read_scores(relation_csv)
    return merge_relation_frame(df, relation_df)


def merge_relation_frame(df, relation_df):
    if RELATION_COLUMN not in relation_df.columns:
        raise ValueError(f"Relation scores are missing {RELATION_COLUMN}.")
    return df.merge(
        relation_df[["Sample_Key", RELATION_COLUMN]],
        on="Sample_Key",
        how="inner",
        validate="one_to_one",
    )


def merge_dino(df, dino_csv):
    dino_df = read_scores(dino_csv)
    if DINO_COLUMN not in dino_df.columns:
        raise ValueError(f"DINO scores are missing {DINO_COLUMN}.")
    return df.merge(
        dino_df[["Sample_Key", DINO_COLUMN]],
        on="Sample_Key",
        how="inner",
        validate="one_to_one",
    )


def robust_stats(scores):
    values = np.asarray(scores, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median)) + EPS
    return median, 1.4826 * mad


def robust_z(scores, ref_scores):
    median, scale = robust_stats(ref_scores)
    return np.maximum(0.0, (np.asarray(scores, dtype=float) - median) / scale)


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def parse_float_grid(value):
    if value is None:
        return ()
    weights = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        weight = float(part)
        if weight <= 0:
            raise ValueError("--grid_dino values must be positive")
        weights.append(weight)
    return tuple(weights)


def weight_label(weight):
    return f"{weight:g}".replace(".", "p")


def calibrated_z_scores(test_df, normal_df, use_relation, use_dino):
    columns = list(BRANCH_COLUMNS)
    if use_relation:
        columns.append(RELATION_COLUMN)
    if use_dino:
        columns.append(DINO_COLUMN)

    missing_test = sorted(set(columns).difference(test_df.columns))
    missing_normal = sorted(set(columns).difference(normal_df.columns))
    if missing_test:
        raise ValueError(f"test CSV is missing score columns: {missing_test}")
    if missing_normal:
        raise ValueError(f"normal CSV is missing calibration columns: {missing_normal}")

    return {
        column: robust_z(test_df[column].to_numpy(), normal_df[column].to_numpy())
        for column in columns
    }


def original_score(test_df):
    if "Equal_Fusion_Score" in test_df.columns:
        return test_df["Equal_Fusion_Score"].astype(float).to_numpy()
    return (
        test_df["Appearance_Score"].astype(float).to_numpy()
        + test_df["Global_Score"].astype(float).to_numpy()
        + test_df["Composition_Score"].astype(float).to_numpy()
    )


def fusion_scores(test_df, z_scores, use_relation, use_dino, dino_grid=()):
    z_img = z_scores["Appearance_Score"]
    z_mah = z_scores["Global_Score"]
    z_comp = z_scores["Composition_Score"]
    branches = [z_img, z_mah, z_comp]

    if use_relation:
        z_rel = z_scores["Relation_Score"]
        branches.append(z_rel)
    else:
        z_rel = np.zeros_like(z_comp)

    if use_dino:
        z_dino = z_scores[DINO_COLUMN]
        branches.append(z_dino)
    else:
        z_dino = z_mah

    branch_matrix = np.vstack(branches)
    structural = np.maximum(z_img, z_dino)
    structural_all = np.maximum.reduce([z_img, z_mah, z_dino])
    logical = np.maximum(z_comp, z_rel) if use_relation else z_comp

    probabilities = sigmoid(branch_matrix)
    scores = {
        "original": original_score(test_df),
        "robust_z_sum": np.sum(branch_matrix, axis=0),
        "robust_z_max": np.max(branch_matrix, axis=0),
        "loco_gate": np.maximum(structural, logical),
        "structural_gate": structural_all,
        "dino_replace_sum": z_img + z_dino + z_comp,
        "noisy_or": 1.0 - np.prod(1.0 - probabilities, axis=0),
    }
    if use_dino:
        for weight in dino_grid:
            scores[f"structural_gate_dino_{weight_label(weight)}"] = np.maximum.reduce(
                [z_img, z_mah, weight * z_dino]
            )
    return scores


def loco_auc(df, scores):
    labels = df["Ground_Truth"].astype(int)
    if labels.nunique() != 2:
        raise ValueError("test CSV must contain both normal and anomalous rows.")
    good = df["Defect_Class"] == "good"
    defect_classes = sorted(cls for cls in df["Defect_Class"].unique() if cls != "good")
    if not defect_classes:
        raise ValueError("test CSV has no defect classes besides good.")

    per_defect = {}
    for defect_class in defect_classes:
        subset = df[good | (df["Defect_Class"] == defect_class)]
        per_defect[defect_class] = roc_auc_score(subset["Ground_Truth"].astype(int), scores[subset.index])
    mean_auc = float(np.mean(list(per_defect.values())))
    return mean_auc, per_defect


def evaluate_modes(test_df, mode_scores):
    rows = []
    for mode, scores in mode_scores.items():
        auc, per_defect = loco_auc(test_df, scores)
        row = {
            "Mode": mode,
            "AUC": auc * 100.0,
        }
        for defect_class, defect_auc in per_defect.items():
            row[f"AUC_{defect_class}"] = defect_auc * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Logic-aware calibrated fusion for MVTec LOCO branch scores.")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--normal_csv", required=True)
    parser.add_argument("--relation_csv", default=None)
    parser.add_argument("--normal_relation_csv", default=None)
    parser.add_argument("--dino_csv", default=None)
    parser.add_argument("--normal_dino_csv", default=None)
    parser.add_argument("--grid_dino", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()
    dino_grid = parse_float_grid(args.grid_dino)

    test_df = read_scores(args.test_csv)
    normal_df = read_scores(args.normal_csv)

    use_relation = args.relation_csv is not None
    if use_relation:
        test_df = merge_relation(test_df, args.relation_csv)
        if RELATION_COLUMN not in normal_df.columns:
            if args.normal_relation_csv is None:
                raise ValueError("--normal_relation_csv is required when --relation_csv is used and --normal_csv lacks Relation_Score.")
            normal_relation_df = read_scores(args.normal_relation_csv)
            normal_relation_df = normal_relation_df[normal_relation_df["Ground_Truth"].astype(int) == 0]
            normal_df = merge_relation_frame(normal_df, normal_relation_df)

    use_dino = args.dino_csv is not None
    if use_dino:
        test_df = merge_dino(test_df, args.dino_csv)
        if DINO_COLUMN not in normal_df.columns:
            if args.normal_dino_csv is None:
                raise ValueError("--normal_dino_csv is required when --dino_csv is used and --normal_csv lacks DINO_Memory_Score.")
            normal_df = merge_dino(normal_df, args.normal_dino_csv)

    test_df = test_df.reset_index(drop=True)
    normal_df = normal_df[normal_df["Ground_Truth"].astype(int) == 0].reset_index(drop=True)
    z_scores = calibrated_z_scores(test_df, normal_df, use_relation, use_dino)
    mode_scores = fusion_scores(test_df, z_scores, use_relation, use_dino, dino_grid)
    result_df = evaluate_modes(test_df, mode_scores)

    print("LOCO Fusion")
    print(f"  Test rows:       {len(test_df)}")
    print(f"  Normal rows:     {len(normal_df)}")
    print(f"  Relation branch: {'yes' if use_relation else 'no'}")
    print(f"  DINO branch:     {'yes' if use_dino else 'no'}")
    print(result_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        scored = test_df.copy()
        for mode, scores in mode_scores.items():
            scored[f"{mode}_score"] = scores
        scored.to_csv(output, index=False)

    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result_df.to_dict(orient="records"), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
