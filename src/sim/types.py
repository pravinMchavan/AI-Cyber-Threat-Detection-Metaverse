"""Shared types for the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttackType(str, Enum):
    none = "none"
    ddos = "ddos"
    phishing = "phishing"


class EventType(str, Enum):
    login = "login"
    movement = "movement"
    chat = "chat"
    asset_download = "asset_download"
    purchase = "purchase"


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration for the running simulator."""

    events_per_second: float = 5.0
    buffer_size: int = 2000
    random_state: int = 42
