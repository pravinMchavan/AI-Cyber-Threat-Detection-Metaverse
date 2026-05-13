"""Create synthetic (generated) network-traffic data.

This is useful because in early stages you might not have a real dataset yet.
We generate a simple table (CSV-like) with columns such as:
- duration, src_bytes, dst_bytes, packets
- protocol, service, flag
- label (0 = normal, 1 = attack)

Important:
- Synthetic data is NOT a substitute for real data in the final report.
- But it is perfect to build the pipeline and demo UI first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sim.types import AttackType
from src.sim.event_generator import generate_event


@dataclass(frozen=True)
class SyntheticConfig:
    n_rows: int = 5000
    attack_ratio: float = 0.2
    random_state: int = 42


def make_synthetic_network_data(config: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    """Generate a simple dataset for anomaly detection.

    Returns a pandas DataFrame.

    Technical words:
    - DataFrame = a table (rows and columns) in Python.
    - anomaly = something unusual/suspicious compared to normal.
    """

    rng = np.random.default_rng(config.random_state)

    n_attack = int(config.n_rows * config.attack_ratio)
    n_normal = config.n_rows - n_attack

    # Categorical columns (text-like values)
    protocols = np.array(["tcp", "udp", "icmp"], dtype=object)
    services = np.array(["http", "dns", "ssh", "ftp"], dtype=object)
    flags = np.array(["SF", "S0", "REJ"], dtype=object)

    # Normal traffic: lower bytes/packets
    normal = pd.DataFrame(
        {
            "duration": rng.gamma(shape=2.0, scale=1.0, size=n_normal),
            "src_bytes": rng.lognormal(mean=6.0, sigma=0.6, size=n_normal).astype(int),
            "dst_bytes": rng.lognormal(mean=5.5, sigma=0.7, size=n_normal).astype(int),
            "packets": rng.poisson(lam=20, size=n_normal).astype(int),
            "protocol": rng.choice(protocols, size=n_normal, replace=True),
            "service": rng.choice(services, size=n_normal, replace=True, p=[0.55, 0.25, 0.15, 0.05]),
            "flag": rng.choice(flags, size=n_normal, replace=True, p=[0.85, 0.10, 0.05]),
            "label": np.zeros(n_normal, dtype=int),
        }
    )

    # Attack traffic: higher bytes/packets + more suspicious flags
    attack = pd.DataFrame(
        {
            "duration": rng.gamma(shape=2.0, scale=2.0, size=n_attack),
            "src_bytes": rng.lognormal(mean=7.0, sigma=0.8, size=n_attack).astype(int),
            "dst_bytes": rng.lognormal(mean=6.8, sigma=0.9, size=n_attack).astype(int),
            "packets": rng.poisson(lam=60, size=n_attack).astype(int),
            "protocol": rng.choice(protocols, size=n_attack, replace=True, p=[0.6, 0.3, 0.1]),
            "service": rng.choice(services, size=n_attack, replace=True, p=[0.35, 0.20, 0.25, 0.20]),
            "flag": rng.choice(flags, size=n_attack, replace=True, p=[0.55, 0.30, 0.15]),
            "label": np.ones(n_attack, dtype=int),
        }
    )

    df = pd.concat([normal, attack], ignore_index=True)

    # Shuffle rows so attacks are not all at the bottom.
    df = df.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)

    return df


@dataclass(frozen=True)
class SyntheticMetaverseConfig:
    """Config for generating metaverse-style synthetic telemetry."""

    n_rows: int = 8000
    ddos_ratio: float = 0.12
    phishing_ratio: float = 0.08
    random_state: int = 42


def make_synthetic_metaverse_data(
    config: SyntheticMetaverseConfig = SyntheticMetaverseConfig(),
) -> pd.DataFrame:
    """Generate a metaverse telemetry dataset with DDoS + phishing samples.

    Output columns include both:
    - original v1 network-like columns
    - v2 metaverse context columns (event_type, contains_url, req_rate, ...)
    """

    rng = np.random.default_rng(config.random_state)

    n_ddos = int(config.n_rows * config.ddos_ratio)
    n_phish = int(config.n_rows * config.phishing_ratio)
    n_normal = config.n_rows - n_ddos - n_phish

    events: list[dict] = []

    for _ in range(n_normal):
        events.append(generate_event(rng=rng, attack_type=AttackType.none, intensity=1.0))

    for _ in range(n_ddos):
        # Slight intensity variation for realism.
        intensity = float(rng.uniform(0.8, 2.2))
        events.append(generate_event(rng=rng, attack_type=AttackType.ddos, intensity=intensity))

    for _ in range(n_phish):
        events.append(generate_event(rng=rng, attack_type=AttackType.phishing, intensity=1.0))

    df = pd.DataFrame(events)
    df = df.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)
    return df
