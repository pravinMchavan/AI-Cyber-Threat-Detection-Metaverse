"""Unified inference: supervised + unsupervised anomaly detection.

Outputs columns used by the dashboard and alerting:
- supervised_label, supervised_prob
- anomaly_score, anomaly_flag (if unsupervised model exists)
- pred_label, pred_text (final decision)

Decision policy (simple, adjustable):
- final attack if supervised_prob >= prob_threshold OR anomaly_flag == 1
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import PATHS
from src.data.load import validate_columns
from src.data.upgrade import upgrade_dataframe
from src.models.anomaly import load_unsupervised_model
from src.models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def load_supervised_model():
    if not PATHS.model_path.exists():
        raise FileNotFoundError(
            "Model not found. Train the model first (example: py scripts\\train_model.py)"
        )
    return joblib.load(PATHS.model_path)


def predict_all(
    df: pd.DataFrame,
    *,
    prob_threshold: float = 0.55,
    use_unsupervised: bool = True,
) -> pd.DataFrame:
    validate_columns(df, require_label=False)
    df = upgrade_dataframe(df)

    supervised = load_supervised_model()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    supervised_label = supervised.predict(X).astype(int)

    supervised_prob = None
    try:
        supervised_prob = supervised.predict_proba(X)[:, 1].astype(float)
    except Exception:
        supervised_prob = None

    out = df.copy()
    out["supervised_label"] = supervised_label
    if supervised_prob is not None:
        out["supervised_prob"] = supervised_prob
    else:
        out["supervised_prob"] = 0.0

    # Unsupervised anomaly signals (optional).
    out["anomaly_score"] = 0.0
    out["anomaly_flag"] = 0
    if use_unsupervised and PATHS.unsupervised_model_path.exists():
        bundle = load_unsupervised_model()
        pipeline = bundle["pipeline"]
        threshold = float(bundle["threshold"])
        scores = pipeline.score_samples(X)
        out["anomaly_score"] = scores.astype(float)
        out["anomaly_flag"] = (out["anomaly_score"] <= threshold).astype(int)

    final_label = ((out["supervised_prob"] >= float(prob_threshold)) | (out["anomaly_flag"] == 1)).astype(
        int
    )

    out["pred_label"] = final_label
    out["pred_text"] = out["pred_label"].map({0: "Normal", 1: "Attack"})

    return out
