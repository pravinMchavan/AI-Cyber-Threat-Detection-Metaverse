"""Unsupervised anomaly detection (IsolationForest).

Why unsupervised?
- In real systems you often have lots of "normal" telemetry and few labeled attacks.
- Unsupervised models learn what normal looks like and flag outliers.

We train IsolationForest mainly on normal samples (label==0 if available).
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_RANDOM_STATE, PATHS
from src.data.load import validate_columns
from src.data.upgrade import upgrade_dataframe
from src.models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


@dataclass(frozen=True)
class UnsupervisedConfig:
    contamination: float = 0.05
    random_state: int = DEFAULT_RANDOM_STATE
    # Threshold percentile on training scores: lower scores => more anomalous.
    threshold_percentile: float = 5.0


def train_unsupervised(
    df: pd.DataFrame,
    config: UnsupervisedConfig = UnsupervisedConfig(),
) -> dict:
    """Train IsolationForest and save to artifacts.

    If a `label` column exists, we train mostly on normal samples (label=0).
    """

    validate_columns(df, require_label=False)
    df = upgrade_dataframe(df)

    if "label" in df.columns:
        train_df = df[df["label"].astype(int) == 0].copy()
        if len(train_df) < 50:
            # Fallback: if not enough normal rows, train on full data.
            train_df = df.copy()
    else:
        train_df = df.copy()

    X = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    model = IsolationForest(
        n_estimators=300,
        contamination=float(config.contamination),
        random_state=int(config.random_state),
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
    pipeline.fit(X)

    scores = pipeline.score_samples(X)
    threshold = float(np.percentile(scores, float(config.threshold_percentile)))

    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "config": {
                "contamination": float(config.contamination),
                "threshold_percentile": float(config.threshold_percentile),
            },
            "features": {
                "numeric": NUMERIC_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
            },
        },
        PATHS.unsupervised_model_path,
    )

    return {
        "train_rows": int(len(train_df)),
        "threshold": threshold,
        "contamination": float(config.contamination),
        "threshold_percentile": float(config.threshold_percentile),
    }


def load_unsupervised_model() -> dict:
    if not PATHS.unsupervised_model_path.exists():
        raise FileNotFoundError(
            "Unsupervised model not found. Train it first (py scripts\\train_model.py)."
        )

    return joblib.load(PATHS.unsupervised_model_path)


def predict_unsupervised(df: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly score + flag columns to the input DataFrame."""

    validate_columns(df, require_label=False)
    df = upgrade_dataframe(df)

    bundle = load_unsupervised_model()
    pipeline: Pipeline = bundle["pipeline"]
    threshold: float = float(bundle["threshold"])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    scores = pipeline.score_samples(X)

    out = df.copy()
    out["anomaly_score"] = scores.astype(float)
    out["anomaly_flag"] = (out["anomaly_score"] <= threshold).astype(int)

    return out
