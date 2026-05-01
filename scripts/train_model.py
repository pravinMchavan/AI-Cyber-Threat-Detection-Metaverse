"""Train the MVP model.

This is a small command-line script.

Examples:
1) Train on sample synthetic data:
    py scripts/train_model.py

2) Train on your own CSV (must contain required columns + label):
    py scripts/train_model.py --csv data/raw/my_dataset.csv

Technical word:
- command-line = running a program by typing a command in terminal.
"""

from __future__ import annotations

# --- Make `src/...` imports work reliably ---
#
# Problem (simple words):
# - This file is inside the `scripts/` folder.
# - When Python runs a script, it may not automatically know where your project root is.
# - Our main code lives in the `src/` folder, so we add the project root to Python's path.
#
# Technical word:
# - sys.path = a list of folders where Python looks for modules/packages to import.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.models.train import load_training_data, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to training CSV. If not provided, synthetic data is used.",
    )

    args = parser.parse_args()

    df = load_training_data(args.csv)
    metrics = train_model(df)

    print("Training complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print("Saved model + metrics in artifacts/")


if __name__ == "__main__":
    main()
