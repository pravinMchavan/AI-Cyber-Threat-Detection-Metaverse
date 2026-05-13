"""Flask simulator server (polling endpoints).

Endpoints (polling-friendly):
- GET  /health
- GET  /events?limit=100
- POST /attack/start  {"type":"ddos|phishing", "duration_sec":30, "intensity":1.0}
- POST /attack/stop
- POST /sim/reset

Run via: `py scripts/run_simulator.py`
"""

from __future__ import annotations

import threading
import time

import numpy as np
from flask import Flask, jsonify, request

from src.sim.event_generator import generate_event
from src.sim.state import SimulatorState
from src.sim.types import AttackType, SimulatorConfig


def create_app(*, config: SimulatorConfig, state: SimulatorState) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        """Interactive UI (simple HTML) for browser users."""

        html = """<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Metaverse Simulator</title>
        <style>
            body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }
            .row { display: flex; gap: 16px; flex-wrap: wrap; }
            .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; min-width: 280px; }
            button { padding: 8px 12px; margin-right: 8px; cursor: pointer; }
            input { padding: 6px; width: 90px; }
            pre { background: #f7f7f7; padding: 12px; border-radius: 10px; overflow: auto; max-height: 420px; }
            .muted { color: #666; }
        </style>
    </head>
    <body>
        <h2>Metaverse Simulator (Interactive Control Panel)</h2>
        <p class=\"muted\">This UI polls <code>/health</code> and <code>/events</code>. Use the buttons to trigger attacks.</p>

        <div class=\"row\">
            <div class=\"card\">
                <h3>Controls</h3>
                <div>
                    <label>DDoS duration (sec): <input id=\"ddos_dur\" type=\"number\" value=\"30\" min=\"5\" max=\"120\" /></label>
                </div>
                <div style=\"margin-top:8px\">
                    <label>DDoS intensity: <input id=\"ddos_int\" type=\"number\" value=\"1.5\" min=\"0.5\" max=\"3.0\" step=\"0.1\" /></label>
                </div>
                <div style=\"margin-top:10px\">
                    <button onclick=\"startAttack('ddos')\">Start DDoS</button>
                    <button onclick=\"startAttack('phishing')\">Start Phishing</button>
                </div>
                <div style=\"margin-top:10px\">
                    <button onclick=\"postJson('/attack/stop', {})\">Stop Attack</button>
                    <button onclick=\"postJson('/sim/reset', {})\">Reset Buffer</button>
                </div>
                <div style=\"margin-top:10px\">
                    <label>Events limit: <input id=\"limit\" type=\"number\" value=\"50\" min=\"5\" max=\"2000\" /></label>
                    <button onclick=\"refreshOnce()\">Refresh</button>
                </div>
            </div>

            <div class=\"card\">
                <h3>Status</h3>
                <pre id=\"health\">Loading...</pre>
            </div>
        </div>

        <h3 style=\"margin-top:18px\">Recent events</h3>
        <pre id=\"events\">Loading...</pre>

        <p class=\"muted\">API index (JSON): <a href=\"/api\">/api</a></p>

        <script>
            async function postJson(path, payload) {
                const r = await fetch(path, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload || {})
                });
                return await r.json();
            }

            async function startAttack(type) {
                const dur = Number(document.getElementById('ddos_dur').value || 30);
                const intensity = Number(document.getElementById('ddos_int').value || 1.5);
                const payload = (type === 'ddos')
                    ? { type: 'ddos', duration_sec: dur, intensity: intensity }
                    : { type: 'phishing', duration_sec: dur, intensity: 1.0 };
                await postJson('/attack/start', payload);
                await refreshOnce();
            }

            async function refreshOnce() {
                const limit = Number(document.getElementById('limit').value || 50);
                const h = await fetch('/health');
                document.getElementById('health').textContent = JSON.stringify(await h.json(), null, 2);
                const e = await fetch('/events?limit=' + encodeURIComponent(limit));
                document.getElementById('events').textContent = JSON.stringify((await e.json()).events || [], null, 2);
            }

            // Auto refresh every 2 seconds
            refreshOnce();
            setInterval(refreshOnce, 2000);
        </script>
    </body>
</html>"""

        return html, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.get("/api")
    def api_index():
        """JSON index for programmatic clients."""

        base = {
            "message": "Metaverse simulator is running.",
            "endpoints": {
                "ui": "/",
                "health": "/health",
                "events": "/events?limit=100",
                "attack_start": "/attack/start (POST)",
                "attack_stop": "/attack/stop (POST)",
                "reset": "/sim/reset (POST)",
            },
            "example_attack_start_payload": {
                "type": "ddos",
                "duration_sec": 30,
                "intensity": 1.5,
            },
        }
        return jsonify({**base, **state.mode_snapshot()})

    @app.get("/favicon.ico")
    def favicon():
        # Avoid noisy 404s when users open the base URL in a browser.
        return ("", 204)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", **state.mode_snapshot()})

    @app.get("/events")
    def events():
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 5000))
        return jsonify({"events": state.get_events(limit=limit)})

    @app.post("/attack/start")
    def attack_start():
        payload = request.get_json(silent=True) or {}
        typ = str(payload.get("type", "none")).lower().strip()
        duration_sec = payload.get("duration_sec", None)
        intensity = float(payload.get("intensity", 1.0))

        if typ not in {"ddos", "phishing"}:
            return jsonify({"error": "type must be ddos or phishing"}), 400

        attack_type = AttackType.ddos if typ == "ddos" else AttackType.phishing
        state.set_attack(attack_type=attack_type, duration_sec=duration_sec, intensity=intensity)
        return jsonify({"status": "ok", **state.mode_snapshot()})

    @app.post("/attack/stop")
    def attack_stop():
        state.stop_attack()
        return jsonify({"status": "ok", **state.mode_snapshot()})

    @app.post("/sim/reset")
    def sim_reset():
        state.reset()
        return jsonify({"status": "ok", **state.mode_snapshot()})

    return app


def start_background_generator(*, config: SimulatorConfig, state: SimulatorState) -> threading.Thread:
    """Start the event generator loop in a daemon thread."""

    rng = np.random.default_rng(config.random_state)

    def loop() -> None:
        sleep_s = max(0.01, 1.0 / max(0.1, float(config.events_per_second)))
        while True:
            attack_type = state.get_effective_attack_type()
            intensity = state.get_intensity()
            event = generate_event(rng=rng, attack_type=attack_type, intensity=intensity)
            state.append_event(event)
            time.sleep(sleep_s)

    t = threading.Thread(target=loop, name="sim-event-generator", daemon=True)
    t.start()
    return t
