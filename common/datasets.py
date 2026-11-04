import os
import zipfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from common.config import DATASET_NAME, ACTIVE_SENSORS

# Works both inside Docker (/app/data) and locally on Windows/Mac (./data)
DEFAULT_DATA_DIR = "/app/data" if os.path.exists("/app") else os.path.join(os.getcwd(), "data")
UCI_URL = "[https://archive.ics.uci.edu/static/public/360/air+quality.zip](https://archive.ics.uci.edu/static/public/360/air+quality.zip)"


def load_or_download_dataset(data_dir: str = DEFAULT_DATA_DIR):
    """Downloads and loads the active dataset with zero Kafka dependencies."""
    os.makedirs(data_dir, exist_ok=True)
    uci_csv_path = os.path.join(data_dir, "AirQualityUCI.csv")

    if not os.path.exists(uci_csv_path):
        print(f"[Dataset Loader] Downloading UCI Air Quality dataset from {UCI_URL}...")
        zip_path = os.path.join(data_dir, "air_quality.zip")
        resp = requests.get(UCI_URL, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("[Dataset Loader] Download and extraction complete.")

    df = pd.read_csv(uci_csv_path, sep=";", decimal=",")
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # Parse actual historical recording timestamp from Date (DD/MM/YYYY) and Time (HH.MM.SS)
    if "Date" in df.columns and "Time" in df.columns:
        raw_dt_str = (
            df["Date"].astype(str).str.strip()
            + " "
            + df["Time"].astype(str).str.strip().str.replace(".", ":", regex=False)
        )
        parsed_dt = pd.to_datetime(raw_dt_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        df["event_time"] = parsed_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df.dropna(subset=["event_time"]).reset_index(drop=True)
    else:
        df["event_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Convert active sensor columns to float (keep -200.0 sentinel for causal imputation)
    for col in ACTIVE_SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-200.0)

    return df, ACTIVE_SENSORS


def causal_imputation(raw_matrix: np.ndarray, fallback_means: np.ndarray = None) -> np.ndarray:
    """
    Causal streaming forward-fill:
    Replaces any <= -199.0 sentinel or NaN with the last valid reading per sensor.
    If the very first reading (row 0) is missing, uses fallback_means (or first valid reading) instead of 0.0.
    """
    n_rows, n_cols = raw_matrix.shape
    cleaned = np.zeros_like(raw_matrix, dtype=np.float64)

    # Determine safe initial fallback per column (never default to 0.0 if avoidable)
    last_valid = np.zeros(n_cols, dtype=np.float64)
    for c in range(n_cols):
        if fallback_means is not None:
            last_valid[c] = float(fallback_means[c])
        else:
            valid_col = raw_matrix[(~np.isnan(raw_matrix[:, c])) & (raw_matrix[:, c] > -199.0), c]
            last_valid[c] = float(valid_col[0]) if len(valid_col) > 0 else 0.0

    for r in range(n_rows):
        for c in range(n_cols):
            val = float(raw_matrix[r, c])
            if np.isnan(val) or val <= -199.0:
                val = last_valid[c]
            else:
                last_valid[c] = val
            cleaned[r, c] = val

    return cleaned.astype(np.float32)