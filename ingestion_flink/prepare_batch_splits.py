import os
import numpy as np
import pandas as pd
from ingestion_flink.replay_producer import load_or_download_dataset
from common.config import WINDOW_SIZE, WINDOW_SLIDE

SPLIT_DIR = "/app/data/splits"

def create_chronological_splits():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    df, cols = load_or_download_dataset()
    values = df[cols].values.astype(np.float32)

    n_total = len(values)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.85)

    train_raw = values[:n_train]
    val_raw = values[n_train:n_val]
    test_raw = values[n_val:]

    # Fit Z-score scaler strictly on training split (no look-ahead leakage!)
    mean = train_raw.mean(axis=0)
    std = np.where(train_raw.std(axis=0) < 1e-6, 1.0, train_raw.std(axis=0))

    def make_windows(arr):
        norm_arr = (arr - mean) / std
        windows = []
        for i in range(0, len(norm_arr) - WINDOW_SIZE + 1, WINDOW_SLIDE):
            windows.append(norm_arr[i : i + WINDOW_SIZE])
        return np.array(windows, dtype=np.float32)

    X_train = make_windows(train_raw)
    X_val = make_windows(val_raw)
    X_test = make_windows(test_raw)

    out_path = os.path.join(SPLIT_DIR, "UCI_AirQuality_windows.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        mean=mean,
        std=std,
        sensor_names=np.array(cols),
    )
    print(f"[Batch Split Exporter] Saved to {out_path}")
    print(f"  -> X_train shape: {X_train.shape} (70%)")
    print(f"  -> X_val   shape: {X_val.shape} (15%)")
    print(f"  -> X_test  shape: {X_test.shape} (15%)")

if __name__ == "__main__":
    create_chronological_splits()