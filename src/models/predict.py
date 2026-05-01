"""Load a saved model and run predictions on new data."""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import PATHS
from src.data.load import validate_columns
from src.models.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def load_model():
    """Load the trained pipeline from `artifacts/model.joblib`."""

    if not PATHS.model_path.exists():
        raise FileNotFoundError(
            "Model not found. Train the model first (example: py scripts\\train_model.py)"
        )

    return joblib.load(PATHS.model_path)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Predict label for each row.

    Returns a copy of the input with two extra columns:
    - pred_label (0=normal, 1=attack)
    - pred_text (Normal/Attack)
    """

    validate_columns(df, require_label=False)

    model = load_model()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pred = model.predict(X)

    out = df.copy()
    out["pred_label"] = pred.astype(int)
    out["pred_text"] = out["pred_label"].map({0: "Normal", 1: "Attack"})

    return out
