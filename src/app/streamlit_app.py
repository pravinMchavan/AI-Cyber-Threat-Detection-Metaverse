"""Streamlit demo app.

Streamlit = a Python tool to make a simple web app quickly.
This app lets you:
- Upload a CSV with network records
- Run the trained model
- See which rows look suspicious

How to run (after installing requirements):
- py scripts/train_model.py
- streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

# --- Make `src/...` imports work reliably ---
# Streamlit runs this file as an app. Depending on how you start it,
# Python might not find the `src/` folder automatically.
# So we add the project root folder to `sys.path`.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import time

import pandas as pd
import requests
import streamlit as st

from src.config import PATHS
from src.data.make_synthetic import make_synthetic_network_data
from src.monitoring.alerts import AlertManager
from src.models.predict import predict


logging.basicConfig(level=logging.INFO)


st.set_page_config(page_title="Cyber Threat Detection (MVP)", layout="wide")

st.title("AI Cyber Threat Detection (MVP-A)")
st.write(
    "Upload a CSV of network traffic and get predictions (Normal vs Attack). "
    "This is an educational demo for an MCA final-year project."
)

with st.expander("What columns should my CSV have?", expanded=True):
    if PATHS.schema_path.exists():
        schema = json.loads(PATHS.schema_path.read_text(encoding="utf-8"))
        st.write("Required columns:")
        st.code(", ".join(schema["required_columns"]))
        if "optional_columns" in schema:
            st.write("Optional columns (metaverse telemetry; auto-filled if missing):")
            st.code(", ".join(schema["optional_columns"]))
        st.write("Optional for prediction: `label` (only needed during training).")
    else:
        st.warning(
            "Schema file not found yet. Train the model first (scripts/train_model.py)."
        )


tab_batch, tab_live = st.tabs(["Batch CSV", "Live Monitoring"])


with tab_batch:
    left, right = st.columns([2, 1])

    with right:
        st.subheader("1) Get data")
        use_sample = st.checkbox("Use sample (synthetic) data", value=True)
        uploaded = None
        if not use_sample:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])

        st.subheader("2) Run prediction")
        run = st.button("Predict")

    with left:
        st.subheader("Results")

        df: pd.DataFrame | None = None

        if use_sample:
            df = make_synthetic_network_data().head(500)
            st.caption("Using sample synthetic data (first 500 rows).")
        elif uploaded is not None:
            df = pd.read_csv(uploaded)

        if run:
            if df is None:
                st.error("Please upload a CSV or enable sample data.")
            else:
                try:
                    pred_df = predict(df)
                    counts = pred_df["pred_text"].value_counts()

                    c1, c2 = st.columns(2)
                    c1.metric("Normal", int(counts.get("Normal", 0)))
                    c2.metric("Attack", int(counts.get("Attack", 0)))

                    st.write("Preview (first 50 rows):")
                    st.dataframe(pred_df.head(50), use_container_width=True)

                    st.download_button(
                        "Download predictions as CSV",
                        data=pred_df.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.exception(e)
        else:
            st.info("Click Predict to run the model.")


with tab_live:
    st.subheader("Live monitoring (poll simulator)")
    st.write(
        "This mode polls the simulated metaverse server and runs hybrid detection "
        "(supervised + unsupervised)."
    )

    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AlertManager(log_path=PATHS.alerts_log_path)
    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        base_url = st.text_input("Simulator base URL", value="https://127.0.0.1:5050")
        verify_tls = st.checkbox(
            "Verify TLS certificate (disable for self-signed demo cert)", value=False
        )
        if base_url.strip().lower().startswith("https") and not verify_tls:
            st.warning("TLS verification is disabled (demo mode).")

    with col_b:
        limit = st.number_input("Events to fetch", min_value=20, max_value=2000, value=200, step=20)
        prob_threshold = st.slider(
            "Attack probability threshold", min_value=0.10, max_value=0.95, value=0.55, step=0.05
        )

    with col_c:
        auto_refresh = st.checkbox("Auto refresh", value=True)
        refresh_sec = st.number_input("Refresh seconds", min_value=1, max_value=30, value=2, step=1)

    st.divider()

    ctl1, ctl2, ctl3, ctl4 = st.columns(4)
    with ctl1:
        ddos_duration = st.number_input("DDoS duration (sec)", min_value=5, max_value=120, value=30, step=5)
        ddos_intensity = st.slider("DDoS intensity", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        start_ddos = st.button("Start DDoS")
    with ctl2:
        phish_duration = st.number_input("Phishing duration (sec)", min_value=5, max_value=120, value=30, step=5)
        start_phish = st.button("Start Phishing")
    with ctl3:
        stop_attack = st.button("Stop Attack")
    with ctl4:
        reset_sim = st.button("Reset Simulator")

    def _post(path: str, payload: dict | None = None) -> None:
        url = base_url.rstrip("/") + path
        requests.post(url, json=payload or {}, timeout=5, verify=verify_tls)

    if start_ddos:
        try:
            _post(
                "/attack/start",
                {"type": "ddos", "duration_sec": float(ddos_duration), "intensity": float(ddos_intensity)},
            )
            st.success("DDoS started")
        except Exception as e:
            st.error(f"Failed to start DDoS: {e}")

    if start_phish:
        try:
            _post(
                "/attack/start",
                {"type": "phishing", "duration_sec": float(phish_duration), "intensity": 1.0},
            )
            st.success("Phishing started")
        except Exception as e:
            st.error(f"Failed to start phishing: {e}")

    if stop_attack:
        try:
            _post("/attack/stop")
            st.success("Attack stopped")
        except Exception as e:
            st.error(f"Failed to stop attack: {e}")

    if reset_sim:
        try:
            _post("/sim/reset")
            st.success("Simulator reset")
        except Exception as e:
            st.error(f"Failed to reset simulator: {e}")

    def _fetch_events() -> pd.DataFrame:
        url = base_url.rstrip("/") + "/events"
        r = requests.get(url, params={"limit": int(limit)}, timeout=5, verify=verify_tls)
        r.raise_for_status()
        payload = r.json()
        events = payload.get("events", [])
        return pd.DataFrame(events)

    refresh_now = st.button("Refresh now")

    try:
        if refresh_now or auto_refresh:
            live_df = _fetch_events()
        else:
            live_df = pd.DataFrame([])
    except Exception as e:
        live_df = pd.DataFrame([])
        st.error(f"Failed to fetch events: {e}")

    if len(live_df) > 0:
        try:
            pred_df = predict(live_df)
        except Exception as e:
            st.exception(e)
            pred_df = live_df

        counts = pred_df.get("pred_text", pd.Series(dtype=object)).value_counts()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Events", int(len(pred_df)))
        k2.metric("Normal", int(counts.get("Normal", 0)))
        k3.metric("Attack", int(counts.get("Attack", 0)))
        k4.metric("Alert log", str(PATHS.alerts_log_path.name))

        # Alerts
        try:
            new_alerts = st.session_state.alert_manager.evaluate(pred_df)
            if new_alerts:
                st.session_state.alert_history = (new_alerts + st.session_state.alert_history)[:50]
        except Exception as e:
            st.error(f"Alert evaluation failed: {e}")

        if st.session_state.alert_history:
            st.subheader("Alerts")
            for a in st.session_state.alert_history[:10]:
                st.write(f"[{a.severity}] {a.rule}: {a.message}")
        else:
            st.info("No alerts yet.")

        # Time series chart (attacks over time)
        if "timestamp" in pred_df.columns:
            tmp = pred_df.copy()
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce", utc=True)
            tmp = tmp.dropna(subset=["timestamp"])
            if len(tmp) > 0 and "pred_label" in tmp.columns:
                tmp = tmp.sort_values("timestamp")
                per_min = (
                    tmp.set_index("timestamp")["pred_label"]
                    .astype(int)
                    .resample("10S")
                    .sum()
                    .rename("attacks")
                    .to_frame()
                )
                st.subheader("Attacks over time")
                st.line_chart(per_min)

        st.subheader("Recent events (with model outputs)")
        show_cols = [
            c
            for c in [
                "timestamp",
                "event_type",
                "service",
                "protocol",
                "req_rate",
                "packets",
                "contains_url",
                "supervised_prob",
                "anomaly_score",
                "pred_text",
                "attack_type",
            ]
            if c in pred_df.columns
        ]
        st.dataframe(pred_df[show_cols].tail(50), use_container_width=True)

    else:
        st.info("No live events yet. Start the simulator and refresh.")

    if auto_refresh:
        time.sleep(float(refresh_sec))
        st.rerun()
