"""Load data from CSV and validate required columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "packets",
    "protocol",
    "service",
    "flag",
    # For training we also need `label`.
]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    return df


def validate_columns(df: pd.DataFrame, *, require_label: bool) -> None:
    """Validate that the DataFrame has required columns.

    - require_label=True  => needs `label` column (for training)
    - require_label=False => label is optional (for prediction)
    """

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Your CSV must contain these columns."
        )

    if require_label and "label" not in df.columns:
        raise ValueError("Training CSV must contain a `label` column (0=normal, 1=attack).")
