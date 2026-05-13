"""Alert notification system (local demo).

Requirements supported:
- Create alerts based on detected anomalies/attacks.
- Notify via console logs.
- Persist alerts to a file.
- Provide simple deduplication/rate limiting.

This module is intentionally simple for an MCA project demo.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


logger = logging.getLogger("metaverse.alerts")


@dataclass(frozen=True)
class Alert:
    timestamp: str
    severity: str  # low/medium/high
    rule: str
    message: str
    context: dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_parse_ts(series: pd.Series) -> pd.Series:
    # Works with ISO timestamps; invalid values become NaT.
    return pd.to_datetime(series, errors="coerce", utc=True)


class AlertManager:
    def __init__(
        self,
        *,
        log_path: Path,
        dedupe_window_sec: float = 30.0,
    ) -> None:
        self.log_path = Path(log_path)
        self.dedupe_window_sec = float(dedupe_window_sec)
        self._recent: dict[str, float] = {}

        # Ensure parent exists.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _is_duplicate(self, key: str) -> bool:
        now = time.time()
        last = self._recent.get(key)
        if last is None:
            return False
        return (now - last) < self.dedupe_window_sec

    def emit(self, alert: Alert) -> bool:
        """Emit alert (console + JSONL). Returns True if emitted (not deduped)."""

        key = f"{alert.rule}|{alert.severity}"
        if self._is_duplicate(key):
            return False

        self._recent[key] = time.time()

        # Console notification
        logger.warning("[%s] %s - %s", alert.severity.upper(), alert.rule, alert.message)

        # Persist to disk (JSON Lines)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(alert), ensure_ascii=False) + "\n")

        return True

    def evaluate(self, df: pd.DataFrame) -> list[Alert]:
        """Evaluate simple alert rules on the most recent events.

        Expects `df` to include:
        - timestamp (ISO)
        - pred_label (0/1)
        - req_rate, packets, contains_url
        """

        if df is None or len(df) == 0:
            return []

        alerts: list[Alert] = []

        ts = _safe_parse_ts(df.get("timestamp", pd.Series(["" for _ in range(len(df))])))
        df2 = df.copy()
        df2["_ts"] = ts

        now = datetime.now(timezone.utc)
        window = now - pd.Timedelta(seconds=60)
        recent = df2[df2["_ts"].notna() & (df2["_ts"] >= window)]

        # Rule 1: High attack rate in last minute
        if len(recent) > 0 and "pred_label" in recent.columns:
            attacks = int((recent["pred_label"].astype(int) == 1).sum())
            if attacks >= 10:
                alerts.append(
                    Alert(
                        timestamp=_utc_now_iso(),
                        severity="high",
                        rule="high_attack_rate",
                        message=f"{attacks} attacks detected in the last 60 seconds",
                        context={"attacks_last_60s": attacks, "events_last_60s": int(len(recent))},
                    )
                )

        # Rule 2: DDoS spike heuristic
        if all(c in df2.columns for c in ["pred_label", "req_rate", "packets"]):
            ddos_like = df2[(df2["pred_label"].astype(int) == 1) & ((df2["req_rate"] >= 120) | (df2["packets"] >= 220))]
            if len(ddos_like) > 0:
                top = ddos_like.sort_values(by=["req_rate", "packets"], ascending=False).head(1).iloc[0]
                alerts.append(
                    Alert(
                        timestamp=_utc_now_iso(),
                        severity="high",
                        rule="ddos_spike",
                        message="DDoS-like spike detected (high req_rate/packets)",
                        context={
                            "req_rate": float(top.get("req_rate", 0.0)),
                            "packets": int(top.get("packets", 0)),
                            "service": str(top.get("service", "")),
                            "protocol": str(top.get("protocol", "")),
                        },
                    )
                )

        # Rule 3: Phishing indicator heuristic
        if all(c in df2.columns for c in ["pred_label", "contains_url"]):
            phish_like = df2[(df2["pred_label"].astype(int) == 1) & (df2["contains_url"].astype(int) == 1)]
            if len(phish_like) > 0:
                top = phish_like.head(1).iloc[0]
                alerts.append(
                    Alert(
                        timestamp=_utc_now_iso(),
                        severity="medium",
                        rule="phishing_indicator",
                        message="Phishing-like chat detected (URL indicator)",
                        context={
                            "url_domain_category": str(top.get("url_domain_category", "unknown")),
                            "message_len": int(top.get("message_len", 0)),
                            "user_id": str(top.get("user_id", "")),
                        },
                    )
                )

        # Emit alerts with dedupe.
        emitted: list[Alert] = []
        for alert in alerts:
            if self.emit(alert):
                emitted.append(alert)

        return emitted
