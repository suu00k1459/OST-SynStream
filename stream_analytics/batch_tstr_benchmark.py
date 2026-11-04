"""
BATCH 3 - Downstream utility: TRTR vs TSTR
=========================================
TRTR = Train on Real,      Test on Real   (the reference)
TSTR = Train on Synthetic, Test on Real   (does synthetic data teach a model
                                           as well as real data does?)

Task A (UCI Air Quality): forecast CO(GT) one hour ahead from the previous
        23 hours of all sensors.            Metrics: MAE, RMSE (lower = better)
Task B (HAI):             detect attacked windows with an IsolationForest
        trained on normal data only.        Metrics: AUROC, F1 (higher = better)

Synthetic data: pass Suyoung's generated windows as .npy files shaped
(N, 24, D), in the SAME z-scored space as produced here (scaler saved to
results/*_scaler.npz). Without them, a placeholder (real + noise) is used,
plus a "random noise" row as a sanity floor.

Usage
-----
  python batch_tstr_benchmark.py --uci data/AirQualityUCI.csv
  python batch_tstr_benchmark.py --hai-train "data/hai/train*.csv" --hai-test "data/hai/test*.csv"
  python batch_tstr_benchmark.py --uci data/AirQualityUCI.csv \
         --uci-synthetic results/uci_timegan.npy:TimeGAN results/uci_timevae.npy:TimeVAE
  # with the team's shared split file (Faisal's prepare_batch_splits.py):
  python batch_tstr_benchmark.py --uci-splits ../data/splits/UCI_AirQuality_windows.npz \
         --uci-synthetic results/uci_timegan.npy:TimeGAN
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score)

import data_loading as dl
from metrics import compare_windows, placeholder_synthetic

OUT = os.path.join(os.path.dirname(__file__), "results")


def synthetic_sources(train_w, specs, seed=0):
    """Return {name: windows}. specs = ['path.npy:Name', ...] or None."""
    rng = np.random.default_rng(seed)
    if specs:
        out = {}
        for s in specs:
            path, _, name = s.partition(":")
            w = np.load(path)
            assert w.shape[1:] == train_w.shape[1:], \
                f"{path}: shape {w.shape[1:]} != real {train_w.shape[1:]} (L, D)"
            out[name or os.path.basename(path)] = w
        return out
    print("  ! no synthetic files given -> using PLACEHOLDER (real + noise)")
    return {
        "placeholder(real+noise)": placeholder_synthetic(train_w, 0.15, seed),
        "random-noise (floor)": rng.normal(0, 1, train_w.shape),
    }


# ======================================================================
# Task A: UCI forecasting
# ======================================================================
def windows_to_xy(w, target_idx):
    """(N, L, D) -> X = first L-1 steps flattened, y = target at step L."""
    return w[:, :-1, :].reshape(len(w), -1), w[:, -1, target_idx]


def scaler_features(z):
    """Feature names from a scaler/split file (ours: 'features', Faisal's: 'sensor_names')."""
    key = "features" if "features" in z.files else "sensor_names"
    return [str(f) for f in z[key]]


def load_faisal_splits(path, test_part):
    """
    Read Faisal's prepare_batch_splits.py output (<DATASET>_windows.npz):
    X_train (stride 1, clipped +-4), X_test_30pct / X_test (stride 4, clipped), mean, std, sensor_names.
    We use the CLIPPED versions for train AND test: that is what the generator is
    trained on and what Flink sends live, so TRTR and TSTR see the same representation.
    """
    z = np.load(path)
    feats = scaler_features(z)
    if feats != dl.UCI_FEATURES:
        raise SystemExit(f"{path}: features {feats} differ from the agreed {dl.UCI_FEATURES}")
    test_key = {"test_30pct": "X_test_30pct", "test": "X_test"}[test_part]
    for k in ("X_train", test_key, "mean", "std"):
        if k not in z.files:
            raise SystemExit(f"{path} has no '{k}' (keys: {z.files}); regenerate with prepare_batch_splits.py")
    return (z["X_train"].astype(np.float64), z[test_key].astype(np.float64),
            z["mean"].astype(np.float64), z["std"].astype(np.float64), feats, test_key)


def run_uci(path, syn_specs, L, splits=None, test_part="test_30pct"):
    print(f"\n=== Task A: UCI forecasting ({dl.UCI_TARGET}, 1 step ahead) ===")
    t_idx = dl.UCI_FEATURES.index(dl.UCI_TARGET)
    if splits:
        # ---- the team's shared split file (Faisal) ----
        train_w, test_w, mean, std, feats, test_key = load_faisal_splits(splits, test_part)
        print(f"  using Faisal's split file {splits}: X_train + {test_key} (clipped +-4, his scaler)")
    else:
        # ---- built here from the CSV ----
        df = dl.load_uci(path)
        train_df, test_df = dl.chrono_split(df, 0.7)
        scaler = dl.ZScaler().fit(train_df.values)
        tr, te = scaler.transform(train_df.values), scaler.transform(test_df.values)
        mean, std, feats = scaler.mean_, scaler.std_, dl.UCI_FEATURES
        train_w, test_w = dl.make_windows(tr, L), dl.make_windows(te, L)
    # same file names either way, so Pipeline 5, check_synthetic.py and the test kit
    # always use the scaling and training windows that Batch 3 used
    np.savez(os.path.join(OUT, "uci_scaler.npz"), mean=mean, std=std, features=np.array(feats))
    np.save(os.path.join(OUT, "uci_train_windows.npy"), train_w)   # for Suyoung / DCR
    X_te, y_te = windows_to_xy(test_w, t_idx)
    print(f"  train windows {train_w.shape}, test windows {test_w.shape}")

    models = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "GradBoost": lambda: HistGradientBoostingRegressor(max_iter=200, random_state=0),
    }
    std_t = std[t_idx]                  # to report errors in original units (mg/m^3)

    sources = {"REAL (TRTR)": train_w, **synthetic_sources(train_w, syn_specs)}
    rows = []
    # naive baseline: tomorrow = last value
    naive = X_te.reshape(len(X_te), L - 1, -1)[:, -1, t_idx]
    rows.append({"train_on": "naive last-value", "model": "-",
                 "MAE": mean_absolute_error(y_te, naive) * std_t,
                 "RMSE": mean_squared_error(y_te, naive) ** 0.5 * std_t})
    for src, w in sources.items():
        X, y = windows_to_xy(w, t_idx)
        sim = compare_windows(train_w[:2000], w[:2000]) if src != "REAL (TRTR)" else {}
        for mname, make in models.items():
            pred = make().fit(X, y).predict(X_te)
            rows.append({"train_on": src, "model": mname,
                         "MAE": mean_absolute_error(y_te, pred) * std_t,
                         "RMSE": mean_squared_error(y_te, pred) ** 0.5 * std_t, **sim})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.to_csv(os.path.join(OUT, "batch3_uci_forecasting.csv"), index=False)
    return res


# ======================================================================
# Task B: HAI anomaly detection
# ======================================================================
def window_features(w):
    """(N, L, D) -> summary features per window: mean, std, min, max, last, slope."""
    slope = w[:, -1, :] - w[:, 0, :]
    return np.concatenate([w.mean(1), w.std(1), w.min(1), w.max(1), w[:, -1, :], slope], 1)


def run_hai(train_glob, test_glob, label_glob, syn_specs, L, stride, max_rows, repick=False):
    print("\n=== Task B: HAI anomaly detection (IsolationForest) ===")
    train_df = dl.load_hai(train_glob, max_rows)
    test_df = dl.load_hai(test_glob, max_rows, label_paths=label_glob)
    train_df = train_df[train_df["attack"] == 0]              # normal data only
    sensors = dl.pick_hai_sensors(train_df) if repick else list(dl.HAI_SENSORS)
    missing = [c for c in sensors if c not in train_df]
    if missing:
        raise SystemExit(f"HAI files lack sensors {missing}: check the HAI version (22.04 expected)")
    print(f"  sensors used ({len(sensors)}): {sensors}")
    with open(os.path.join(OUT, "hai_sensors.json"), "w") as f:
        json.dump(sensors, f, indent=1)

    scaler = dl.ZScaler().fit(train_df[sensors].values)
    tr = scaler.transform(train_df[sensors].values)
    te = scaler.transform(test_df[sensors].values)
    np.savez(os.path.join(OUT, "hai_scaler.npz"), mean=scaler.mean_, std=scaler.std_,
             features=np.array(sensors))
    train_w = dl.make_windows(tr, L, stride)
    test_w = dl.make_windows(te, L, stride)
    y_te = dl.window_labels(test_df["attack"].values, L, stride)
    np.save(os.path.join(OUT, "hai_train_windows.npy"), train_w)
    print(f"  train windows {train_w.shape}, test windows {test_w.shape}, "
          f"attacked test windows {y_te.mean():.1%}")
    F_te = window_features(test_w)

    rows = []
    sources = {"REAL (TRTR)": train_w, **synthetic_sources(train_w, syn_specs)}
    for src, w in sources.items():
        F = window_features(w)
        model = IsolationForest(n_estimators=200, random_state=0).fit(F)
        s_train = -model.score_samples(F)            # higher = more anomalous
        s_test = -model.score_samples(F_te)
        thr = np.quantile(s_train, 0.99)             # threshold from training only
        pred = (s_test > thr).astype(int)
        sim = compare_windows(train_w[:2000], w[:2000]) if src != "REAL (TRTR)" else {}
        rows.append({"train_on": src, "model": "IsolationForest",
                     "AUROC": roc_auc_score(y_te, s_test) if y_te.any() else np.nan,
                     "F1": f1_score(y_te, pred, zero_division=0),
                     "Precision": precision_score(y_te, pred, zero_division=0),
                     "Recall": recall_score(y_te, pred, zero_division=0), **sim})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.to_csv(os.path.join(OUT, "batch3_hai_anomaly.csv"), index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uci", help="path to AirQualityUCI.csv")
    ap.add_argument("--uci-synthetic", nargs="*", help="Suyoung's .npy files, 'path:Name'")
    ap.add_argument("--uci-splits", help="use Faisal's split file (e.g. ../data/splits/UCI_AirQuality_windows.npz) "
                                         "instead of building windows from the CSV")
    ap.add_argument("--uci-test-part", choices=["test_30pct", "test"], default="test_30pct",
                    help="with --uci-splits: last 30%% (default) or last 15%% as test period")
    ap.add_argument("--hai-train", help="HAI train CSV path or glob")
    ap.add_argument("--hai-test", help="HAI test CSV path or glob")
    ap.add_argument("--hai-labels", help="HAI 23.05 label CSVs (glob), if labels are separate")
    ap.add_argument("--hai-synthetic", nargs="*", help="Suyoung's .npy files, 'path:Name'")
    ap.add_argument("--hai-stride", type=int, default=10, help="window stride for 1 Hz HAI")
    ap.add_argument("--hai-max-rows", type=int, default=200_000)
    ap.add_argument("--hai-repick-sensors", action="store_true",
                    help="re-select HAI sensors automatically instead of the agreed list (EDA only)")
    ap.add_argument("--L", type=int, default=dl.WINDOW_LEN)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not (a.uci or a.uci_splits or a.hai_train):
        ap.error("give --uci (or --uci-splits) and/or --hai-train/--hai-test")
    if a.uci or a.uci_splits:
        run_uci(a.uci, a.uci_synthetic, a.L, a.uci_splits, a.uci_test_part)
    if a.hai_train:
        run_hai(a.hai_train, a.hai_test, a.hai_labels, a.hai_synthetic,
                a.L, a.hai_stride, a.hai_max_rows, a.hai_repick_sensors)
    print(f"\nresults written to {OUT}/")
