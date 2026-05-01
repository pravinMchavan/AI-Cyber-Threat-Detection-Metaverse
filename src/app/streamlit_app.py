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

import pandas as pd
import streamlit as st

from src.config import PATHS
from src.data.make_synthetic import make_synthetic_network_data
from src.models.predict import predict


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
        st.write("Optional for prediction: `label` (only needed during training).")
    else:
        st.warning(
            "Schema file not found yet. Train the model first (scripts/train_model.py)."
        )

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
