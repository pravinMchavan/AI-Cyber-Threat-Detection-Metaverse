"""Evaluate detection accuracy and false positives.

Outputs `artifacts/evaluation.json`.

Examples:
- py scripts\\train_model.py
- py scripts\\evaluate_models.py
- py scripts\\evaluate_models.py --csv data\\raw\\metaverse_synthetic.csv
"""

from __future__ import annotations

# --- Make `src/...` imports work reliably ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import PATHS
from src.data.make_synthetic import SyntheticMetaverseConfig, make_synthetic_metaverse_data
from src.models.inference import predict_all


def _rates(cm: list[list[int]]) -> dict:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "false_positive_rate": fpr, "true_positive_rate": tpr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Optional evaluation CSV with label column")
    parser.add_argument("--prob-threshold", type=float, default=0.55)

    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = make_synthetic_metaverse_data(SyntheticMetaverseConfig(n_rows=12000))

    if "label" not in df.columns:
        raise SystemExit("Evaluation requires a label column.")

    pred_df = predict_all(df, prob_threshold=float(args.prob_threshold), use_unsupervised=True)

    y_true = df["label"].astype(int)
    y_pred = pred_df["pred_label"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    results = {
        "n_rows": int(len(df)),
        "prob_threshold": float(args.prob_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"labels": [0, 1], "matrix": cm},
        "rates": _rates(cm),
        "attack_type_counts": (
            pred_df.get("attack_type", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
        ),
    }

    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    PATHS.evaluation_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Evaluation saved:", PATHS.evaluation_path)
    print(json.dumps(results["rates"], indent=2))


if __name__ == "__main__":
    main()
