"""Mutable simulator state (attack mode, rolling buffer).

Kept separate so it can be unit-tested more easily and reused
by the Flask app and any offline scenario runners.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from src.sim.types import AttackType


@dataclass
class AttackMode:
    attack_type: AttackType = AttackType.none
    intensity: float = 1.0
    end_time_utc: float | None = None


class SimulatorState:
    def __init__(self, *, buffer_size: int) -> None:
        self._lock = threading.Lock()
        self._events = deque(maxlen=buffer_size)
        self._mode = AttackMode()

    def append_event(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)

    def get_events(self, *, limit: int) -> list[dict]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._events)[-limit:]

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._mode = AttackMode()

    def set_attack(self, *, attack_type: AttackType, duration_sec: float | None, intensity: float) -> None:
        end_time = None
        if duration_sec is not None:
            end_time = time.time() + float(duration_sec)

        with self._lock:
            self._mode = AttackMode(
                attack_type=attack_type,
                intensity=max(0.1, float(intensity)),
                end_time_utc=end_time,
            )

    def stop_attack(self) -> None:
        with self._lock:
            self._mode = AttackMode()

    def get_effective_attack_type(self) -> AttackType:
        """Return attack type, auto-expiring time-limited modes."""

        with self._lock:
            mode = self._mode
            if mode.attack_type == AttackType.none:
                return AttackType.none

            if mode.end_time_utc is None:
                return mode.attack_type

            if time.time() <= mode.end_time_utc:
                return mode.attack_type

            # Auto-expire.
            self._mode = AttackMode()
            return AttackType.none

    def get_intensity(self) -> float:
        with self._lock:
            return float(self._mode.intensity)

    def mode_snapshot(self) -> dict:
        with self._lock:
            return {
                "attack_type": self._mode.attack_type.value,
                "intensity": float(self._mode.intensity),
                "end_time_utc": self._mode.end_time_utc,
                "buffer_size": self._events.maxlen,
                "events_in_buffer": len(self._events),
            }
