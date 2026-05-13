"""Run the simulated metaverse server environment (Flask).

This is the "server" part of the project tasks.
It generates synthetic metaverse telemetry events and exposes a polling API.

Examples (Windows PowerShell):
- py scripts\\run_simulator.py
- py scripts\\run_simulator.py --port 5050 --https

Then in browser:
- https://127.0.0.1:5050/health
- https://127.0.0.1:5050/events?limit=20
"""

from __future__ import annotations

 # --- Make `src/...` imports work reliably ---
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.config import PATHS
from src.sim.server import create_app, start_background_generator
from src.sim.state import SimulatorState
from src.sim.tls import ensure_self_signed_cert
from src.sim.types import SimulatorConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--eps", type=float, default=5.0, help="Events per second")
    parser.add_argument("--buffer", type=int, default=2000, help="Rolling event buffer size")
    parser.add_argument("--https", action="store_true", help="Enable HTTPS (self-signed cert)")

    args = parser.parse_args()

    config = SimulatorConfig(events_per_second=float(args.eps), buffer_size=int(args.buffer))
    state = SimulatorState(buffer_size=config.buffer_size)

    start_background_generator(config=config, state=state)
    app = create_app(config=config, state=state)

    ssl_context = None
    if args.https:
        cert_path, key_path = ensure_self_signed_cert(cert_dir=PATHS.certs_dir)
        ssl_context = (str(cert_path), str(key_path))

    app.run(host=args.host, port=int(args.port), debug=False, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
