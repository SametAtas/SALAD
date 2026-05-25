#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from relation_features import sample_key


SCORE_COLUMNS = ("Appearance_Score", "Global_Score", "Composition_Score", "Relation_Score")
BRANCH_NAMES = ("img", "mah", "comp", "rel")
DEFAULT_GRID = {
    "img": (0.5, 1.0, 1.5),
    "mah": (0.5, 1.0, 1.5),
    "comp": (0.5, 1.0, 1.5, 2.0),
    "rel": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
}
EPS = 1e-12


def parse_weights(value):
    weights = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(weights) != 4:
        raise ValueError("--weights must contain exactly four values: img,mah,comp,rel")
    if any(weight < 0 for weight in weights):
        raise ValueError("--weights must be non-negative")
    if sum(weights) == 0:
        raise ValueError("--weights cannot all be zero")
    return np.array(weights, dtype=float)


def parse_grid(value, default):
    if value is None:
        return default
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Grid values cannot be empty.")
    return parsed


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


def read_branch_scores(path):
    df = pd.read_csv(path).copy()
    df = add_sample_key(df)
    missing = sorted(set(SCORE_COLUMNS[:3]).difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing branch-score columns: {missing}")
    return df


def read_relation_scores(path):
    df = pd.read_csv(path).copy()
    df = add_sample_key(df)
    if "Relation_Score" not in df.columns:
        raise ValueError(f"{path} is missing Relation_Score.")
    return df


def read_ready_scores(path):
    df = pd.read_csv(path).copy()
    df = add_sample_key(df)
    missing = sorted((set(SCORE_COLUMNS) | {"Ground_Truth", "Defect_Class"}).difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def merge_scores(branch_df, relation_df):
    relation_keep = ["Sample_Key", "Relation_Score"]
    merged = branch_df.merge(relation_df[relation_keep], on="Sample_Key", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No rows merged between branch scores and relation scores. Check Sample_Key construction.")
    missing_branch = set(branch_df["Sample_Key"]).difference(merged["Sample_Key"])
    missing_relation = set(relation_df["Sample_Key"]).difference(merged["Sample_Key"])
    if missing_branch:
        print(f"Warning: {len(missing_branch)} branch rows did not match relation rows.")
    if missing_relation:
        print(f"Warning: {len(missing_relation)} relation rows did not match branch rows.")
    return merged


def load_merged_scores(branch_scores, relation_scores):
    return merge_scores(read_branch_scores(branch_scores), read_relation_scores(relation_scores))


def validate_scores(df, name):
    labels = sorted(df["Ground_Truth"].astype(int).unique().tolist())
    if labels != [0, 1]:
        raise ValueError(f"{name} must contain both normal and anomalous rows; found labels {labels}.")
    missing = sorted(set(SCORE_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"{name} missing score columns: {missing}")


def loco_auc(df, scores):
    work = df.loc[:, ["Ground_Truth", "Defect_Class"]].copy()
    work["Ground_Truth"] = work["Ground_Truth"].astype(int)
    work["score"] = scores
    good = work["Defect_Class"] == "good"
    defect_classes = sorted(cls for cls in work["Defect_Class"].unique() if cls != "good")
    if not defect_classes:
        raise ValueError("Cannot compute LOCO AUC without defect classes besides 'good'.")

    per_defect = {}
    for defect_class in defect_classes:
        subset = work[good | (work["Defect_Class"] == defect_class)]
        if subset["Ground_Truth"].nunique() != 2:
            raise ValueError(f"Cannot compute AUC for {defect_class}: subset does not contain both labels.")
        per_defect[defect_class] = roc_auc_score(subset["Ground_Truth"], subset["score"])
    return float(np.mean(list(per_defect.values()))), per_defect


class ValidationNormalizer:
    def __init__(self, columns, means, stds):
        self.columns = tuple(columns)
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)

    @classmethod
    def fit(cls, df, columns):
        normal_df = df[df["Ground_Truth"].astype(int) == 0]
        if normal_df.empty:
            raise ValueError("Cannot fit normalizer: validation scores have no normal rows.")
        values = normal_df.loc[:, columns].astype(float).to_numpy()
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        stds = np.where(stds < EPS, 1.0, stds)
        return cls(columns, means, stds)

    @classmethod
    def identity(cls, columns):
        return cls(columns, np.zeros(len(columns)), np.ones(len(columns)))

    def transform(self, df):
        return (df.loc[:, self.columns].astype(float).to_numpy() - self.means) / self.stds

    def to_dict(self):
        return {
            "columns": list(self.columns),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }


def weighted_scores(values, weights):
    return np.matmul(values, weights)


def evaluate(df, values, weights):
    scores = weighted_scores(values, weights)
    mean_auc, per_defect = loco_auc(df, scores)
    return {
        "auc": mean_auc,
        "auc_percent": mean_auc * 100.0,
        "per_defect_auc": per_defect,
        "weights": {name: float(weight) for name, weight in zip(BRANCH_NAMES, weights)},
    }


def branch_auc_summary(df):
    summary = {}
    for column, name in zip(SCORE_COLUMNS, BRANCH_NAMES):
        auc, per_defect = loco_auc(df, df[column].astype(float).to_numpy())
        summary[name] = {"auc_percent": auc * 100.0, "per_defect_auc": per_defect}
    return summary


def search_weights(validation_df, validation_values, grids):
    best = None
    for weights in itertools.product(grids["img"], grids["mah"], grids["comp"], grids["rel"]):
        weights = np.asarray(weights, dtype=float)
        result = evaluate(validation_df, validation_values, weights)
        if best is None or result["auc"] > best["auc"]:
            best = result
    return best


def load_validation_scores(args):
    if args.validation_scores:
        return read_ready_scores(args.validation_scores)
    if args.validation_branch_scores and args.validation_relation_scores:
        return load_merged_scores(args.validation_branch_scores, args.validation_relation_scores)
    if args.validation_branch_scores or args.validation_relation_scores:
        raise ValueError("Provide both --validation_branch_scores and --validation_relation_scores, or use --validation_scores.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Fuse SALAD branch scores with relation-layout scores.")
    parser.add_argument("--branch_scores", required=True, help="Real-test branch-score CSV from train/test SALAD.")
    parser.add_argument("--relation_scores", required=True, help="Real-test relation-score CSV.")
    parser.add_argument("--weights", default="1,1,1,0.5", help="Fixed img,mah,comp,rel weights.")
    parser.add_argument("--validation_scores", default=None, help="Pseudo-validation CSV containing all four score columns.")
    parser.add_argument("--validation_branch_scores", default=None)
    parser.add_argument("--validation_relation_scores", default=None)
    parser.add_argument("--grid_img", default=None)
    parser.add_argument("--grid_mah", default=None)
    parser.add_argument("--grid_comp", default=None)
    parser.add_argument("--grid_rel", default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default=None, help="Optional scored real-test CSV output path.")
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    test_df = load_merged_scores(args.branch_scores, args.relation_scores)
    validate_scores(test_df, "test scores")
    validation_df = load_validation_scores(args)
    if validation_df is not None:
        validate_scores(validation_df, "validation scores")

    if args.normalize and validation_df is not None:
        normalizer = ValidationNormalizer.fit(validation_df, SCORE_COLUMNS)
    else:
        normalizer = ValidationNormalizer.identity(SCORE_COLUMNS)
        if args.normalize:
            print("Warning: no validation scores provided; using raw score scales for fixed-weight evaluation.")

    test_values = normalizer.transform(test_df)
    fixed_weights = parse_weights(args.weights)
    fixed_result = evaluate(test_df, test_values, fixed_weights)
    branch_summary = branch_auc_summary(test_df)

    summary = {
        "num_test_rows": int(len(test_df)),
        "normalizer": normalizer.to_dict(),
        "fixed": fixed_result,
        "branch_auc": branch_summary,
    }

    best_validation = None
    calibrated_test = None
    if validation_df is not None:
        validation_values = normalizer.transform(validation_df)
        grids = {
            "img": parse_grid(args.grid_img, DEFAULT_GRID["img"]),
            "mah": parse_grid(args.grid_mah, DEFAULT_GRID["mah"]),
            "comp": parse_grid(args.grid_comp, DEFAULT_GRID["comp"]),
            "rel": parse_grid(args.grid_rel, DEFAULT_GRID["rel"]),
        }
        best_validation = search_weights(validation_df, validation_values, grids)
        best_weights = np.array([best_validation["weights"][name] for name in BRANCH_NAMES], dtype=float)
        calibrated_test = evaluate(test_df, test_values, best_weights)
        summary["grid"] = grids
        summary["best_validation"] = best_validation
        summary["calibrated_test"] = calibrated_test

    print("Relation Fusion")
    print(f"  Rows merged:        {len(test_df)}")
    print(f"  Fixed weights:      img={fixed_weights[0]:.3f}, mah={fixed_weights[1]:.3f}, comp={fixed_weights[2]:.3f}, rel={fixed_weights[3]:.3f}")
    print(f"  Fixed test AUC:     {fixed_result['auc_percent']:.4f}")
    print(f"  Branch rel AUC:     {branch_summary['rel']['auc_percent']:.4f}")
    if calibrated_test is not None:
        weights = calibrated_test["weights"]
        print(f"  Best val AUC:       {best_validation['auc_percent']:.4f}")
        print(f"  Calibrated test AUC:{calibrated_test['auc_percent']:.4f}")
        print(
            "  Calibrated weights: "
            f"img={weights['img']:.3f}, mah={weights['mah']:.3f}, comp={weights['comp']:.3f}, rel={weights['rel']:.3f}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        scored = test_df.copy()
        scored["Relation_Fusion_Score"] = weighted_scores(test_values, fixed_weights)
        if calibrated_test is not None:
            best_weights = np.array([calibrated_test["weights"][name] for name in BRANCH_NAMES], dtype=float)
            scored["Relation_Calibrated_Fusion_Score"] = weighted_scores(test_values, best_weights)
        scored.to_csv(output, index=False)

    if args.summary_json:
        output = Path(args.summary_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
