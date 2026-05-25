#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_BRANCH_COLUMNS = ("Appearance_Score", "Global_Score", "Composition_Score")
BRANCH_NAMES = ("img", "mah", "comp")
EPS = 1e-12


@dataclass
class ScoreNormalizer:
    columns: tuple
    method: str
    means: np.ndarray
    stds: np.ndarray

    @classmethod
    def fit(cls, df, columns, method="zscore", reference="normal"):
        columns = tuple(columns)
        if method == "none":
            return cls(columns=columns, method=method, means=np.zeros(len(columns)), stds=np.ones(len(columns)))

        if reference == "normal":
            reference_df = df[df["Ground_Truth"] == 0]
            if reference_df.empty:
                raise ValueError("Cannot fit normal-only score normalization: validation CSV has no normal rows.")
        else:
            reference_df = df

        values = reference_df.loc[:, columns].astype(float).to_numpy()
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0)
        stds = np.where(stds < EPS, 1.0, stds)
        return cls(columns=columns, method=method, means=means, stds=stds)

    def transform(self, df):
        values = df.loc[:, self.columns].astype(float).to_numpy()
        return (values - self.means) / self.stds

    def to_dict(self):
        return {
            "method": self.method,
            "columns": list(self.columns),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search validation-calibrated SALAD branch-fusion weights and evaluate them on test scores."
    )
    parser.add_argument("--validation-csv", required=True, help="Labeled validation branch-score CSV.")
    parser.add_argument("--test-csv", required=True, help="Held-out test branch-score CSV.")
    parser.add_argument("--grid-step", type=float, default=0.05, help="Simplex grid step for non-negative weights.")
    parser.add_argument(
        "--branch-columns",
        default=",".join(DEFAULT_BRANCH_COLUMNS),
        help="Comma-separated score columns in img,mah,comp order.",
    )
    parser.add_argument("--normalization", choices=("zscore", "none"), default="zscore")
    parser.add_argument("--normalization-reference", choices=("normal", "all"), default="normal")
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--scored-test-csv", default=None, help="Optional test CSV with Weighted_Fusion_Score added.")
    parser.add_argument(
        "--allow-same-csv",
        action="store_true",
        help="Allow validation and test CSVs to be the same file. This is diagnostic only and leaks test labels.",
    )
    return parser.parse_args()


def read_scores(csv_path, columns):
    df = pd.read_csv(csv_path)
    required = set(columns) | {"Ground_Truth", "Defect_Class"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    df = df.copy()
    df["Ground_Truth"] = df["Ground_Truth"].astype(int)
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(columns)].isna().any().any():
        raise ValueError(f"{csv_path} contains non-numeric or NaN branch scores.")
    return df


def validate_labeled_split(df, csv_path):
    labels = sorted(df["Ground_Truth"].unique().tolist())
    if labels != [0, 1]:
        raise ValueError(
            f"{csv_path} must contain both normal and anomalous validation rows for AUC tuning; "
            f"found labels {labels}."
        )
    defect_classes = sorted(cls for cls in df["Defect_Class"].unique() if cls != "good")
    if not defect_classes:
        raise ValueError(f"{csv_path} has no defect classes besides 'good'.")


def loco_auc(df, scores):
    work = df.loc[:, ["Ground_Truth", "Defect_Class"]].copy()
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


def simplex_weights(step):
    if step <= 0 or step > 1:
        raise ValueError("--grid-step must be in (0, 1].")
    steps = int(round(1.0 / step))
    if not np.isclose(steps * step, 1.0):
        raise ValueError("--grid-step must divide 1.0 exactly, e.g. 0.1, 0.05, or 0.02.")

    for img_step in range(steps + 1):
        for mah_step in range(steps + 1 - img_step):
            comp_step = steps - img_step - mah_step
            yield np.array([img_step, mah_step, comp_step], dtype=float) / steps


def weighted_scores(values, weights):
    return np.matmul(values, weights)


def evaluate(values, df, weights):
    scores = weighted_scores(values, weights)
    mean_auc, per_defect = loco_auc(df, scores)
    return {
        "mean_auc": mean_auc,
        "mean_auc_percent": mean_auc * 100.0,
        "per_defect_auc": per_defect,
        "weights": {name: float(weight) for name, weight in zip(BRANCH_NAMES, weights)},
    }


def search_weights(validation_values, validation_df, step):
    best = None
    for weights in simplex_weights(step):
        result = evaluate(validation_values, validation_df, weights)
        if best is None or result["mean_auc"] > best["mean_auc"]:
            best = result
    return best


def main():
    args = parse_args()
    validation_path = Path(args.validation_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    if validation_path == test_path and not args.allow_same_csv:
        raise ValueError(
            "Validation and test CSV paths are identical. Use --allow-same-csv only for leakage diagnostics."
        )

    columns = tuple(column.strip() for column in args.branch_columns.split(",") if column.strip())
    if len(columns) != 3:
        raise ValueError("--branch-columns must provide exactly three columns in img,mah,comp order.")

    validation_df = read_scores(validation_path, columns)
    test_df = read_scores(test_path, columns)
    validate_labeled_split(validation_df, validation_path)
    validate_labeled_split(test_df, test_path)

    normalizer = ScoreNormalizer.fit(
        validation_df,
        columns,
        method=args.normalization,
        reference=args.normalization_reference,
    )
    validation_values = normalizer.transform(validation_df)
    test_values = normalizer.transform(test_df)

    best_validation = search_weights(validation_values, validation_df, args.grid_step)
    best_weights = np.array([best_validation["weights"][name] for name in BRANCH_NAMES])
    test_result = evaluate(test_values, test_df, best_weights)

    equal_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    validation_equal = evaluate(validation_values, validation_df, equal_weights)
    test_equal = evaluate(test_values, test_df, equal_weights)

    summary = {
        "validation_csv": str(validation_path),
        "test_csv": str(test_path),
        "grid_step": args.grid_step,
        "normalizer": normalizer.to_dict(),
        "best_validation": best_validation,
        "test": test_result,
        "equal_weight_validation": validation_equal,
        "equal_weight_test": test_equal,
    }

    print("Validation-calibrated fusion")
    print(f"  Best validation AUC: {best_validation['mean_auc_percent']:.4f}")
    print(f"  Test AUC:            {test_result['mean_auc_percent']:.4f}")
    print(
        "  Weights:             "
        f"img={best_weights[0]:.4f}, mah={best_weights[1]:.4f}, comp={best_weights[2]:.4f}"
    )
    print(f"  Equal-weight test:   {test_equal['mean_auc_percent']:.4f}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.scored_test_csv:
        output_path = Path(args.scored_test_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored = test_df.copy()
        scored["Weighted_Fusion_Score"] = weighted_scores(test_values, best_weights)
        output_path.write_text(scored.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
