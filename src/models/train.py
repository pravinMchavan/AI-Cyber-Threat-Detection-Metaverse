"""Train a baseline ML model and save it.

What this does (in simple words):
1) Get training data (real CSV or synthetic sample).
2) Split into train/test.
3) Train a model.
4) Evaluate model (how good it is).
5) Save the trained model to `artifacts/`.

Technical word:
- pipeline = a single object that contains both preprocessing + model.
  This helps because you can "train" once and "predict" later the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DEFAULT_RANDOM_STATE, PATHS
from src.data.load import validate_columns
from src.data.make_synthetic import SyntheticMetaverseConfig, make_synthetic_metaverse_data
from src.data.upgrade import OPTIONAL_V2_DEFAULTS, upgrade_dataframe


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""

    random_state: int = DEFAULT_RANDOM_STATE
    test_size: float = 0.2


NUMERIC_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packets",
    # v2 optional numeric features
    "req_rate",
    "message_len",
    "contains_url",
]

CATEGORICAL_FEATURES = [
    "protocol",
    "service",
    "flag",
    # v2 optional categorical features
    "event_type",
    "url_domain_category",
]
LABEL_COL = "label"


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for numeric + categorical columns."""

    numeric_transformer = Pipeline(
        steps=[
            # StandardScaler makes numeric columns have similar scale.
            # (Good habit; not always required for RandomForest but fine for learning.)
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # OneHotEncoder converts text categories (tcp/udp/http/etc.) into numbers.
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline() -> Pipeline:
    """Create preprocessing + supervised model as one pipeline."""

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


def train_model(df: pd.DataFrame, config: TrainConfig = TrainConfig()) -> dict:
    """Train and evaluate, returning a metrics dictionary."""

    validate_columns(df, require_label=True)
    df = upgrade_dataframe(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Some metrics use probabilities.
    y_prob = None
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    roc_auc = None
    if y_prob is not None:
        # ROC-AUC needs both classes present.
        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
        },
        "report": classification_report(y_test, y_pred, output_dict=True),
        "n_rows": int(len(df)),
        "test_size": float(config.test_size),
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "label": LABEL_COL,
        },
    }

    # Ensure output folders exist.
    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PATHS.model_path)

    PATHS.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save schema so the app can guide users about expected columns.
    PATHS.schema_path.write_text(
        json.dumps(
            {
                # Keep the original v1 columns as the minimum requirement,
                # and list v2 columns as optional (auto-filled if missing).
                "required_columns": [
                    "duration",
                    "src_bytes",
                    "dst_bytes",
                    "packets",
                    "protocol",
                    "service",
                    "flag",
                ],
                "optional_columns": sorted(list(OPTIONAL_V2_DEFAULTS.keys())),
                "model_feature_columns": {
                    "numeric": NUMERIC_FEATURES,
                    "categorical": CATEGORICAL_FEATURES,
                },
                "label_column": LABEL_COL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return metrics


def load_training_data(csv_path: str | Path | None) -> pd.DataFrame:
    """Load real CSV if provided, otherwise create synthetic demo data."""

    if csv_path is None:
        return make_synthetic_metaverse_data(SyntheticMetaverseConfig())

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return df
