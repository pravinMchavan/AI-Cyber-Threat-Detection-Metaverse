"""Project configuration (paths + constants).

Why this file exists:
- We don't want hard-coded paths scattered everywhere.
- One place to change folders like `data/` and `artifacts/`.

Technical word:
- "artifact" = an output file produced by the project (trained model, metrics, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Root folder of the repo.
ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    """Common folders used across the project."""

    data_dir: Path = ROOT_DIR / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"

    artifacts_dir: Path = ROOT_DIR / "artifacts"

    # TLS certificates for the local HTTPS simulator.
    certs_dir: Path = artifacts_dir / "certs"

    # Where the trained model (pipeline) will be saved.
    model_path: Path = artifacts_dir / "model.joblib"

    # Unsupervised anomaly model (IsolationForest pipeline).
    unsupervised_model_path: Path = artifacts_dir / "model_unsupervised.joblib"

    # Where we save evaluation numbers (accuracy/F1 etc.).
    metrics_path: Path = artifacts_dir / "metrics.json"

    # Extra evaluation output (confusion matrix, ROC-AUC, false positives, etc.).
    evaluation_path: Path = artifacts_dir / "evaluation.json"

    # Alert log (JSON Lines): one JSON object per alert.
    alerts_log_path: Path = artifacts_dir / "alerts.jsonl"

    # Where we store information about expected input columns.
    schema_path: Path = artifacts_dir / "schema.json"


PATHS = ProjectPaths()

# Default settings for reproducibility.
DEFAULT_RANDOM_STATE = 42
