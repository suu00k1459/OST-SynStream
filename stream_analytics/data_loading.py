"""
Data loading and windowing helpers shared by all of Hazem's scripts.

Two datasets:
  * UCI Air Quality  (AirQualityUCI.csv, ';'-separated, ',' decimals, -200 = missing)
  * HAI ICS security (train*.csv / test*.csv, 1 Hz, an 'attack' label column in test files)

Everything returns plain pandas / numpy objects so the same code works offline
(on CSV files) and later inside the Kafka services.
"""
from __future__ import annotations

import glob
import re
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Settings agreed with the team (change here only, every script imports them)
# --------------------------------------------------------------------------
WINDOW_LEN = 24          # L: 24 steps (= 1 day for hourly UCI data)

UCI_FEATURES = [         # NMHC(GT) is dropped: ~90% missing
    "CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
    "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]
UCI_TARGET = "CO(GT)"    # forecasting target for Batch 3 / Pipeline 5

HAI_N_SENSORS = 15       # how many continuous HAI sensors to keep

# The AGREED HAI sensor list (from the EDA, same as common/config.py).
# Fixed on purpose: pick_hai_sensors() depends on which training rows it sees,
# so re-picking on every run could silently change the features.
HAI_SENSORS = [
    "P1_FT01", "P1_FT02", "P1_FT03", "P1_LIT01", "P1_PIT01", "P1_PIT02",
    "P1_TIT02", "P1_TIT03", "P2_VIBTR01", "P2_VIBTR02", "P2_VIBTR03",
    "P2_VIBTR04", "P3_FIT01", "P3_LIT01", "P3_PIT01",
]


# --------------------------------------------------------------------------
# UCI Air Quality
# --------------------------------------------------------------------------
def load_uci(path: str) -> pd.DataFrame:
    """Load and clean AirQualityUCI.csv -> DataFrame indexed by timestamp."""
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]      # trailing ';;'
    df = df.dropna(subset=["Date", "Time"])                     # empty tail rows
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S"
    )
    df = df.set_index("timestamp").drop(columns=["Date", "Time"])
    df = df.replace(-200, np.nan)                               # missing marker
    df = df[UCI_FEATURES].astype(float)
    # fill short gaps by time interpolation, then forward/back fill the rest
    df = df.interpolate(method="time", limit=6).ffill().bfill()
    return df


# --------------------------------------------------------------------------
# HAI
# --------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, names) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def load_hai(paths, max_rows: int | None = None, label_paths=None) -> pd.DataFrame:
    """
    Load one or several HAI CSV files (a path, a glob pattern or a list).
    Returns a DataFrame with all sensor columns plus a 0/1 'attack' column
    (all zeros for training files).

    HAI 23.05 keeps test labels in separate files (label-test1.csv ...);
    pass them via label_paths in the same order as the test files.
    """
    if isinstance(paths, str):
        paths = sorted(glob.glob(paths)) or [paths]
    if isinstance(label_paths, str):
        label_paths = sorted(glob.glob(label_paths)) or [label_paths]

    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        ts = _find_col(df, ["timestamp", "time"])
        if ts:
            df[ts] = pd.to_datetime(df[ts])
            df = df.set_index(ts)
            df.index.name = "timestamp"

        lab = _find_col(df, ["attack", "label"])
        if lab is None and label_paths:
            lab_df = pd.read_csv(label_paths[i])
            lab_col = _find_col(lab_df, ["attack", "label"])
            df["attack"] = lab_df[lab_col].values[: len(df)]
        elif lab is not None:
            df = df.rename(columns={lab: "attack"})
        else:
            df["attack"] = 0

        # older HAI versions have per-process attack columns (attack_P1 ...)
        df = df.drop(columns=[c for c in df.columns
                              if c.lower().startswith("attack_")], errors="ignore")
        frames.append(df)

    df = pd.concat(frames)
    if max_rows:
        df = df.iloc[:max_rows]
    df["attack"] = (df["attack"] > 0).astype(int)
    return df


# HAI naming: *_FT/FIT = flow, LIT = level, PIT/PT = pressure, TIT/TT = temperature,
# SIT = speed, VIBTR = vibration. These are MEASUREMENTS. Columns ending in D
# (demand/setpoint) or Z (valve position) are controller values that mostly
# duplicate a measurement, and *_FD/_LD/_PO/_PS are turbine simulator signals.
HAI_MEASUREMENT = re.compile(r"_(FT|FIT|LIT|PIT|TIT|SIT)\d+$|_(PT|TT)\d+$|_VIBTR\d+$")


def pick_hai_sensors(train_df: pd.DataFrame, n: int = HAI_N_SENSORS) -> list[str]:
    """
    Pick up to n continuous PHYSICAL MEASUREMENT sensors (flow, level,
    pressure, temperature, speed, vibration), drop constant or binary ones
    and near-duplicates (|corr| > 0.98 with an already chosen sensor).
    Falls back to a generic variance ranking for datasets with other names.
    Run once, review with Suyoung, then keep the list in results/hai_sensors.json.
    """
    meas = [c for c in train_df.columns
            if HAI_MEASUREMENT.search(c) and np.issubdtype(train_df[c].dtype, np.number)
            and train_df[c].nunique() > 50 and train_df[c].std() > 1e-6]
    if len(meas) >= 5:
        # most "informative" first: many distinct values
        meas.sort(key=lambda c: -train_df[c].nunique())
        chosen = []
        corr = train_df[meas].corr().abs()
        for c in meas:
            if all(corr.loc[c, k] <= 0.98 for k in chosen):
                chosen.append(c)
            if len(chosen) == n:
                break
        return sorted(chosen)

    cand = []
    for c in train_df.columns:
        if c == "attack" or not np.issubdtype(train_df[c].dtype, np.number):
            continue
        if train_df[c].nunique() > 50 and train_df[c].std() > 1e-6:
            # coefficient-of-variation style ranking, robust to units
            cand.append((train_df[c].std() / (abs(train_df[c].mean()) + 1e-6), c))
    cand.sort(reverse=True)
    return [c for _, c in cand[:n]]


# --------------------------------------------------------------------------
# Scaling and windowing
# --------------------------------------------------------------------------
class ZScaler:
    """z-score scaler fitted on TRAIN data only (no test leakage)."""

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


def chrono_split(df: pd.DataFrame, train_frac: float = 0.7):
    """Time-ordered split (never shuffle time series)."""
    cut = int(len(df) * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


def make_windows(arr: np.ndarray, L: int = WINDOW_LEN, stride: int = 1) -> np.ndarray:
    """(T, D) array -> (N, L, D) overlapping windows."""
    idx = np.arange(0, len(arr) - L + 1, stride)
    return np.stack([arr[i:i + L] for i in idx])


def window_labels(labels: np.ndarray, L: int = WINDOW_LEN, stride: int = 1) -> np.ndarray:
    """A window is anomalous if any of its points is under attack."""
    idx = np.arange(0, len(labels) - L + 1, stride)
    return np.array([labels[i:i + L].max() for i in idx])


def find_file(*candidates) -> str | None:
    for c in candidates:
        hits = sorted(glob.glob(c))
        if hits:
            return hits[0]
    return None


DATA_DIR = os.environ.get("SYNSTREAM_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
