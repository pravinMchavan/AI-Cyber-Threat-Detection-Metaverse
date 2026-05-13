"""Backward-compatible prediction helper.

The project started with a simple `predict(df)` that returns:
- pred_label (0/1)
- pred_text (Normal/Attack)

We now support a richer "hybrid" detector (supervised + unsupervised).
This file keeps the original API so the Streamlit app and scripts
continue to work.
"""

from __future__ import annotations

import pandas as pd

from src.models.inference import predict_all


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Predict Normal vs Attack.

    Returns a copy of the input with:
    - pred_label, pred_text (final decision)
    - supervised_label, supervised_prob
    - anomaly_score, anomaly_flag (if unsupervised model exists)
    """

    return predict_all(df)
