from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SynStream - Batch results", layout="wide")
st.title("SynStream: Batch results (UCI Air Quality)")
st.caption("Static batch results (Batch 2 from Suyoung, Batch 3 from Hazem), not live data.")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repository root when this file is in visualization/streamlit_app/


def find_csv(name):
    """Looks next to this file, then in the repository folders of the teammates."""
    candidates = [HERE / name]
    if (ROOT / "visualization").is_dir():
        for sub in ("stream_analytics", "synthetic_gen"):
            if (ROOT / sub).is_dir():
                candidates += sorted((ROOT / sub).rglob(name))
    return next((p for p in candidates if p.exists()), None)


def load(name):
    path = find_csv(name)
    if path is None:
        st.error(f"{name} not found. Put it next to app.py or in the repository.")
        return None
    return pd.read_csv(path)


tab2, tab3 = st.tabs(["Batch 2: similarity", "Batch 3: forecasting"])

with tab2:
    df2 = load("evaluation_results.csv")
    if df2 is not None:
        st.subheader("Similarity between real and synthetic data")
        st.dataframe(df2)
        indexed = df2.set_index("model")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("MMD (lower = more similar to real data)")
            st.bar_chart(indexed[["MMD"]])
        with col2:
            st.caption("Discriminative score (lower = harder to tell apart from real data)")
            st.bar_chart(indexed[["discriminative_score"]])
        st.caption("Predictive MAE on validation data (lower is better)")
        st.bar_chart(indexed[["predictive_MAE_val"]])

with tab3:
    df3 = load("batch3_uci_forecasting.csv")
    if df3 is not None:
        st.subheader("All results")
        st.dataframe(df3)

        st.subheader("Forecasting error (MAE) by training data")
        models = df3[df3["model"] != "-"]
        pivot = models.pivot(index="train_on", columns="model", values="MAE")
        st.bar_chart(pivot)

        baseline = df3[df3["model"] == "-"]
        if not baseline.empty:
            st.metric("Naive last-value baseline (MAE)", round(float(baseline["MAE"].iloc[0]), 3))

        st.subheader("Similarity between real and synthetic data")
        sim_cols = ["mmd_score", "wasserstein", "acf_diff", "xcorr_diff"]
        sim = (
            df3.dropna(subset=["mmd_score"])
            .drop_duplicates("train_on")
            .set_index("train_on")[sim_cols]
        )
        st.dataframe(sim)
        st.caption("These metrics are distances between real and synthetic data: lower means more similar.")
