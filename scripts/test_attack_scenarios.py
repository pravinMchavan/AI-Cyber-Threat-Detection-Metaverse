"""Test against simulated DDoS and phishing attacks (offline scenarios).

This script builds a time-ordered sequence:
- normal window
- DDoS window
- normal window
- phishing window

Then it runs the hybrid detector and prints detection stats.

Example:
- py scripts\\train_model.py
- py scripts\\test_attack_scenarios.py
"""

from __future__ import annotations

# --- Make `src/...` imports work reliably ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.models.inference import predict_all
from src.sim.event_generator import generate_event
from src.sim.types import AttackType


def _make_window(rng: np.random.Generator, attack_type: AttackType, n: int, intensity: float) -> list[dict]:
    return [generate_event(rng=rng, attack_type=attack_type, intensity=intensity) for _ in range(int(n))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=800, help="Events per window")
    parser.add_argument("--prob-threshold", type=float, default=0.55)
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    events: list[dict] = []
    events += _make_window(rng, AttackType.none, args.n, 1.0)
    events += _make_window(rng, AttackType.ddos, args.n, 1.7)
    events += _make_window(rng, AttackType.none, args.n, 1.0)
    events += _make_window(rng, AttackType.phishing, args.n, 1.0)

    df = pd.DataFrame(events)

    pred_df = predict_all(df, prob_threshold=float(args.prob_threshold), use_unsupervised=True)

    y_true = df["label"].astype(int)
    y_pred = pred_df["pred_label"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0

    print("Confusion matrix [0,1]:")
    print(cm)
    print(f"TPR (detection rate): {tpr:.3f}")
    print(f"FPR (false positive): {fpr:.3f}")

    # Breakdown by attack type (ground truth from simulator)
    if "attack_type" in df.columns:
        for typ, sub in df.groupby("attack_type"):
            yt = sub["label"].astype(int)
            yp = pred_df.loc[sub.index, "pred_label"].astype(int)
            cm2 = confusion_matrix(yt, yp, labels=[0, 1])
            tn2, fp2, fn2, tp2 = cm2.ravel().tolist()
            tpr2 = tp2 / (tp2 + fn2) if (tp2 + fn2) else 0.0
            fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) else 0.0
            print(f"[{typ}] TPR={tpr2:.3f} FPR={fpr2:.3f} rows={len(sub)}")


if __name__ == "__main__":
    main()
