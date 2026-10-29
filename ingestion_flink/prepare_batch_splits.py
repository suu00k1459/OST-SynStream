import os
import numpy as np
import pandas as pd
from ingestion_flink.replay_producer import load_or_download_dataset
from common.config import DATASET_NAME, WINDOW_SIZE, WINDOW_SLIDE

SPLIT_DIR = "/app/data/splits"


def causal_imputation(raw_matrix: np.ndarray) -> np.ndarray:
    """
    Replicates PyFlink's exact causal streaming forward-fill:
    Replaces any <= -199.0 sentinel or NaN with the last valid reading per sensor.
    """
    n_rows, n_cols = raw_matrix.shape
    cleaned = np.zeros_like(raw_matrix, dtype=np.float64)
    last_valid = np.zeros(n_cols, dtype=np.float64)
    has_valid = np.zeros(n_cols, dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            val = float(raw_matrix[r, c])
            if np.isnan(val) or val <= -199.0:
                val = last_valid[c] if has_valid[c] else 0.0
            else:
                last_valid[c] = val
                has_valid[c] = True
            cleaned[r, c] = val

    return cleaned.astype(np.float32)


def create_chronological_splits():
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # 1. Load dataset and 12 clean sensor columns (NMHC(GT) already excluded)
    df, cols = load_or_download_dataset()
    raw_values = df[cols].values.astype(np.float64)

    # 2. Apply causal forward-fill imputation (matching PyFlink Pipeline 2)
    imputed_values = causal_imputation(raw_values)

    # 3. Chronological 70% / 15% / 15% split (strictly ordered, no shuffling)
    n_total = len(imputed_values)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.85)

    train_raw = imputed_values[:n_train]
    val_raw = imputed_values[n_train:n_val]
    test_raw = imputed_values[n_val:]

    # 4. Fit Z-score scaler strictly on training split (zero look-ahead leakage!)
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    # 5. Build sliding windows (L=24, S=4) with +/- 4.0 sigma clipping (matching PyFlink)
    def make_windows(arr):
        norm_arr = np.clip((arr - mean) / std, -4.0, 4.0)
        norm_windows = []
        raw_windows = []
        for i in range(0, len(arr) - WINDOW_SIZE + 1, WINDOW_SLIDE):
            norm_windows.append(norm_arr[i : i + WINDOW_SIZE])
            raw_windows.append(arr[i : i + WINDOW_SIZE])
        return (
            np.array(norm_windows, dtype=np.float32),
            np.array(raw_windows, dtype=np.float32),
        )

    X_train, X_train_raw = make_windows(train_raw)
    X_val, X_val_raw = make_windows(val_raw)
    X_test, X_test_raw = make_windows(test_raw)

    out_path = os.path.join(SPLIT_DIR, f"{DATASET_NAME}_windows.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train,          # Normalized & clipped [-4, 4] for Suyoung's TimeGAN/TimeVAE
        X_val=X_val,
        X_test=X_test,
        X_train_raw=X_train_raw,  # Physical units for Hazem's Batch 3 evaluation
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        sensor_names=np.array(cols),
    )

    print("=" * 65)
    print(f"[Batch Split Exporter] Successfully saved to: {out_path}")
    print(f"  -> Sensors ({len(cols)}): {cols}")
    print(f"  -> Min/Max Normalized (Train): [{X_train.min():.2f}, {X_train.max():.2f}] (Clipped +/-4.0)")
    print(f"  -> X_train shape: {X_train.shape} (70%)")
    print(f"  -> X_val   shape: {X_val.shape} (15%)")
    print(f"  -> X_test  shape: {X_test.shape} (15%)")
    print("=" * 65)


if __name__ == "__main__":
    create_chronological_splits()