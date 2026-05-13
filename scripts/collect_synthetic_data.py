"""Collect synthetic attack data (offline generator) and save as CSV.

This satisfies the task: "Collect synthetic attack data for training ML models".

Examples:
- py scripts\\collect_synthetic_data.py
- py scripts\\collect_synthetic_data.py --n-rows 20000 --out data\\raw\\metaverse_dataset.csv
"""

from __future__ import annotations

# --- Make `src/...` imports work reliably ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.data.make_synthetic import SyntheticMetaverseConfig, make_synthetic_metaverse_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=20000)
    parser.add_argument("--ddos-ratio", type=float, default=0.12)
    parser.add_argument("--phishing-ratio", type=float, default=0.08)
    parser.add_argument("--out", type=str, default="data/raw/metaverse_synthetic.csv")

    args = parser.parse_args()

    df = make_synthetic_metaverse_data(
        SyntheticMetaverseConfig(
            n_rows=int(args.n_rows),
            ddos_ratio=float(args.ddos_ratio),
            phishing_ratio=float(args.phishing_ratio),
        )
    )

    out_path = (PROJECT_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df["attack_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
