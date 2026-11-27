from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SynStream - Batch results", layout="wide")
st.title("SynStream: Batch results")
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


def find_windows(dataset):
    """Real training windows and synthetic windows of one dataset (files of the teammates)."""
    real = ROOT / "stream_analytics" / "results" / f"{dataset}_train_windows.npy"
    synthetic = {}
    out = ROOT / "synthetic_gen" / "outputs"
    if out.is_dir():
        for p in sorted(out.glob(f"{dataset}_*.npy")):
            synthetic[p.stem[len(dataset) + 1:]] = p
    return (real if real.exists() else None), synthetic


def flat(path, n, seed):
    """Loads windows, flattens each window to one row and keeps at most n random rows."""
    arr = np.nan_to_num(np.load(path, mmap_mode="r"))
    idx = np.sort(np.random.default_rng(seed).choice(len(arr), size=min(n, len(arr)), replace=False))
    return np.asarray(arr[idx], dtype=np.float64).reshape(len(idx), -1)


@st.cache_data(show_spinner="Computing the projection...")
def embed_2d(real_path, syn_paths, method, n, seed):
    """2-D projection of real and synthetic windows. Returns (DataFrame, explained variance or None)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    parts = {"real": flat(real_path, n, seed)}
    for name, p in syn_paths:
        parts[name] = flat(p, n, seed)
    dims = {k: v.shape[1] for k, v in parts.items()}
    if len(set(dims.values())) != 1:
        raise ValueError(f"Windows have different sizes: {dims}")
    X = np.vstack(list(parts.values()))
    labels = np.concatenate([[k] * len(v) for k, v in parts.items()])
    explained = None
    if method == "PCA":
        pca = PCA(n_components=2, random_state=seed).fit(parts["real"])  # fitted on real data only
        Z = pca.transform(X)
        explained = float(pca.explained_variance_ratio_.sum())
    else:
        perplexity = max(5, min(30, (len(X) - 1) // 3))
        Z = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)
    return pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "source": labels}), explained


tab2, tab3, tab4 = st.tabs(["Batch 2: similarity", "Batch 3: forecasting", "PCA / t-SNE: real vs synthetic"])

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

with tab4:
    st.subheader("Do synthetic windows cover the same region as the real ones?")
    ds = st.selectbox("Dataset", ["uci", "hai"], key="proj_ds")
    real_path, syn = find_windows(ds)
    if real_path is None or not syn:
        st.error(
            f"No windows found for {ds}. Expected stream_analytics/results/{ds}_train_windows.npy "
            f"and synthetic_gen/outputs/{ds}_*.npy in the repository."
        )
    else:
        chosen = st.multiselect("Synthetic generators", sorted(syn), default=sorted(syn), key="proj_gen")
        c1, c2 = st.columns(2)
        with c1:
            method = st.radio("Method", ["PCA", "t-SNE"], horizontal=True, key="proj_method")
        with c2:
            n = st.slider("Windows per source", 100, 1000, 300, step=100, key="proj_n")
        if chosen:
            try:
                df, explained = embed_2d(str(real_path), tuple((g, str(syn[g])) for g in chosen), method, n, 0)
            except ValueError as err:
                st.error(str(err))
            else:
                import altair as alt
                palette = ["#56B4E9", "#009E73", "#CC79A7", "#F0E442"]  # generators; real data is always orange
                order = ["real"] + [g for g in chosen]
                colors = ["#E69F00"] + palette[: len(chosen)]
                chart = (
                    alt.Chart(df)
                    .mark_circle(size=45, opacity=0.6)
                    .encode(
                        x=alt.X("x:Q", title="component 1"),
                        y=alt.Y("y:Q", title="component 2"),
                        color=alt.Color("source:N", scale=alt.Scale(domain=order, range=colors), title="source"),
                        tooltip=["source", "x", "y"],
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
                if explained is not None:
                    st.caption(f"PCA fitted on the real windows only. The two components explain {explained:.0%} of their variance.")
                else:
                    st.caption("t-SNE is fitted on all windows together. Distances between far-apart clusters are not meaningful.")
                st.caption(f"Each window is flattened to one vector. At most {n} random windows per source, fixed seed. Training windows only, not live data.")
