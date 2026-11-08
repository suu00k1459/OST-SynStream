from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SynStream - Batch 3", layout="wide")
st.title("SynStream: Batch 3 forecasting results (UCI Air Quality)")
st.caption("Source: batch3_uci_forecasting.csv (from Hazem). Static batch results, not live data.")

# Looks for the CSV next to this file, then in the repository layout.
here = Path(__file__).resolve().parent
candidates = [
    here / "batch3_uci_forecasting.csv",
    here.parent.parent / "stream_analytics" / "results" / "batch3_uci_forecasting.csv",
]
csv_path = next((p for p in candidates if p.exists()), None)
if csv_path is None:
    st.error("batch3_uci_forecasting.csv not found. Put it next to app.py or in stream_analytics/results/.")
    st.stop()

df = pd.read_csv(csv_path)

st.subheader("All results")
st.dataframe(df)

st.subheader("Forecasting error (MAE) by training data")
models = df[df["model"] != "-"]
pivot = models.pivot(index="train_on", columns="model", values="MAE")
st.bar_chart(pivot)

baseline = df[df["model"] == "-"]
if not baseline.empty:
    st.metric("Naive last-value baseline (MAE)", round(float(baseline["MAE"].iloc[0]), 3))

st.subheader("Similarity between real and synthetic data")
sim_cols = ["mmd_score", "wasserstein", "acf_diff", "xcorr_diff"]
sim = (
    df.dropna(subset=["mmd_score"])
    .drop_duplicates("train_on")
    .set_index("train_on")[sim_cols]
)
st.dataframe(sim)
st.caption("These metrics are distances between real and synthetic data: lower means more similar.")
