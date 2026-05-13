"""Schema upgrade helpers.

The project started with a v1 CSV schema (network-like columns).
To support metaverse-specific threats and real-time telemetry, we introduce
v2 optional columns (event_type, req_rate, etc.).

Goal:
- Keep backward compatibility: v1 CSVs should still predict/train.
- If optional v2 columns are missing, we fill safe defaults.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


OPTIONAL_V2_DEFAULTS: dict[str, object] = {
    # v2 context columns
    "timestamp": "",
    "event_type": "movement",
    "user_id": "",
    "session_id": "",
    "src_ip": "",
    "req_rate": 0.0,
    "message_len": 0,
    "contains_url": 0,
    "url_domain_category": "unknown",
    "attack_type": "none",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upgrade_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all optional v2 columns present."""

    out = df.copy()

    for col, default in OPTIONAL_V2_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default

    # If timestamp exists but is empty, fill with current time.
    if "timestamp" in out.columns:
        if out["timestamp"].astype(str).eq("").all():
            out["timestamp"] = _utc_now_iso()

    # Make sure numeric-ish columns are sane.
    for col in ["req_rate", "message_len", "contains_url"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out
