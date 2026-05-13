"""Generate synthetic metaverse telemetry events.

Design goal:
- Create events that look like what a monitoring/telemetry system might log.
- Keep compatibility with the project's original "network CSV" columns.

We produce both:
- v1 network-like columns (duration, bytes, packets, protocol, service, flag)
- v2 metaverse context columns (timestamp, event_type, contains_url, req_rate, etc.)

Attack simulation:
- DDoS: spikes in packets/req_rate + suspicious flags
- Phishing: chat events with URL-like content indicators
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from src.sim.types import AttackType, EventType


@dataclass(frozen=True)
class GeneratorConfig:
    random_state: int = 42


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _random_ip(rng: np.random.Generator) -> str:
    # Private-like ranges for demo.
    return f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def generate_event(
    *,
    rng: np.random.Generator,
    attack_type: AttackType,
    intensity: float,
) -> dict:
    """Generate one telemetry event as a plain dict."""

    intensity = max(0.1, float(intensity))

    # Baseline categorical fields
    protocols = np.array(["tcp", "udp", "icmp"], dtype=object)
    services = np.array(["http", "dns", "ssh", "ftp"], dtype=object)
    flags = np.array(["SF", "S0", "REJ"], dtype=object)

    event_types = np.array(
        [
            EventType.login.value,
            EventType.movement.value,
            EventType.chat.value,
            EventType.asset_download.value,
            EventType.purchase.value,
        ],
        dtype=object,
    )

    # Choose a base event type.
    if attack_type == AttackType.phishing:
        event_type = EventType.chat.value
    else:
        event_type = str(rng.choice(event_types))

    # Defaults for v2 fields.
    message_len = 0
    contains_url = 0
    url_domain_category = "unknown"  # safe/suspicious/unknown

    # Baseline numeric magnitudes.
    duration = float(rng.gamma(shape=2.0, scale=1.0))
    req_rate = float(rng.lognormal(mean=1.0, sigma=0.4))  # roughly 1-10

    # Choose base protocol/service.
    protocol = str(rng.choice(protocols, p=[0.7, 0.25, 0.05]))
    service = str(rng.choice(services, p=[0.55, 0.25, 0.15, 0.05]))
    flag = str(rng.choice(flags, p=[0.88, 0.09, 0.03]))

    # Map event type to some network-like behavior.
    if event_type == EventType.login.value:
        src_bytes = int(rng.lognormal(mean=5.5, sigma=0.6))
        dst_bytes = int(rng.lognormal(mean=5.2, sigma=0.6))
        packets = int(rng.poisson(lam=18))
        service = "ssh" if rng.random() < 0.6 else service
    elif event_type == EventType.movement.value:
        src_bytes = int(rng.lognormal(mean=5.0, sigma=0.5))
        dst_bytes = int(rng.lognormal(mean=4.8, sigma=0.5))
        packets = int(rng.poisson(lam=12))
    elif event_type == EventType.asset_download.value:
        src_bytes = int(rng.lognormal(mean=6.2, sigma=0.7))
        dst_bytes = int(rng.lognormal(mean=7.0, sigma=0.7))
        packets = int(rng.poisson(lam=28))
        service = "http"
    elif event_type == EventType.purchase.value:
        src_bytes = int(rng.lognormal(mean=5.8, sigma=0.6))
        dst_bytes = int(rng.lognormal(mean=5.8, sigma=0.6))
        packets = int(rng.poisson(lam=16))
        service = "http"
    else:  # chat
        src_bytes = int(rng.lognormal(mean=4.6, sigma=0.5))
        dst_bytes = int(rng.lognormal(mean=4.6, sigma=0.5))
        packets = int(rng.poisson(lam=10))
        service = "http" if rng.random() < 0.7 else service
        message_len = int(rng.integers(10, 240))

        # Some normal chat contains URLs, but rarely.
        if rng.random() < 0.03:
            contains_url = 1
            url_domain_category = "safe"

    # Apply attack mutations.
    attack_label = 0
    if attack_type == AttackType.ddos:
        attack_label = 1
        # DDoS: very high req_rate/packets; more SYN/REJ flags.
        req_rate = float(rng.lognormal(mean=4.3, sigma=0.35) * intensity)  # ~ 50-1500
        packets = int(rng.poisson(lam=120 * intensity))
        src_bytes = int(src_bytes * (1.5 * intensity))
        dst_bytes = int(dst_bytes * (1.2 * intensity))
        duration = float(duration * (1.0 + 0.3 * intensity))
        service = "http" if rng.random() < 0.8 else "dns"
        protocol = "tcp" if rng.random() < 0.85 else protocol
        flag = str(rng.choice(flags, p=[0.35, 0.45, 0.20]))

    elif attack_type == AttackType.phishing:
        attack_label = 1
        event_type = EventType.chat.value
        message_len = int(rng.integers(80, 500))
        contains_url = 1
        url_domain_category = "suspicious"
        # Phishing is not necessarily high-volume.
        req_rate = float(rng.lognormal(mean=1.2, sigma=0.3))
        packets = int(max(1, rng.poisson(lam=14)))
        service = "http"
        protocol = "tcp"
        flag = "SF"

    # Minimal identity/session metadata (kept OUT of ML features by default).
    user_id = f"user_{int(rng.integers(1, 500))}"
    session_id = f"sess_{int(rng.integers(1, 2000))}"

    return {
        # v2 context
        "timestamp": _utc_now_iso(),
        "event_type": event_type,
        "user_id": user_id,
        "session_id": session_id,
        "src_ip": _random_ip(rng),
        "req_rate": float(req_rate),
        "message_len": int(message_len),
        "contains_url": int(contains_url),
        "url_domain_category": url_domain_category,
        "attack_type": attack_type.value,
        # v1 network-like columns
        "duration": float(duration),
        "src_bytes": int(src_bytes),
        "dst_bytes": int(dst_bytes),
        "packets": int(packets),
        "protocol": protocol,
        "service": service,
        "flag": flag,
        # label for ML
        "label": int(attack_label),
    }
