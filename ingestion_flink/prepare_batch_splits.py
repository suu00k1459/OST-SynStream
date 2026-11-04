import os
import numpy as np
from common.datasets import load_or_download_dataset, causal_imputation, DEFAULT_DATA_DIR
from common.config import DATASET_NAME, WINDOW_SIZE, WINDOW_SLIDE


def create_chronological_splits():
    split_dir = os.path.join(DEFAULT_DATA_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # 1. Load dataset and 12 clean sensor columns (no Kafka imports required)
    df, cols = load_or_download_dataset(DEFAULT_DATA_DIR)
    raw_values = df[cols].values.astype(np.float64)

    n_total = len(raw_values)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.85)

    # 2. Compute valid training column means (excluding -200) for safe row-0 fallback
    train_slice = raw_values[:n_train]
    valid_train_means = np.array([
        train_slice[train_slice[:, c] > -199.0, c].mean()
        for c in range(raw_values.shape[1])
    ])

    # 3. Apply causal forward-fill imputation (matching PyFlink Pipeline 2)
    imputed_values = causal_imputation(raw_values, fallback_means=valid_train_means)

    train_raw = imputed_values[:n_train]       # First 70%
    val_raw = imputed_values[n_train:n_val]    # Next 15% (70% -> 85%)
    test_raw = imputed_values[n_val:]          # Final 15% (85% -> 100%)
    test_30_raw = imputed_values[n_train:]     # Full 30% holdout (70% -> 100% for 70/30 setups)

    # 4. Fit Z-score scaler strictly on 70% training split (zero look-ahead leakage)
    mean = train_raw.mean(axis=0).astype(np.float32)
    std = train_raw.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    # Save standalone scaler file so all teammates share one identical scaler
    scaler_path = os.path.join(split_dir, "uci_scaler.npz")
    np.savez(scaler_path, mean=mean, std=std, sensor_names=np.array(cols))

    # 5. Window builder: stride=1 for training (4x more data), stride=4 for val/test
    def make_windows(arr, stride, clip_sigma=4.0):
        norm_unclipped = (arr - mean) / std
        norm_clipped = np.clip(norm_unclipped, -clip_sigma, clip_sigma) if clip_sigma else norm_unclipped

        win_norm_clipped = []
        win_norm_unclipped = []
        win_raw = []
        for i in range(0, len(arr) - WINDOW_SIZE + 1, stride):
            win_norm_clipped.append(norm_clipped[i : i + WINDOW_SIZE])
            win_norm_unclipped.append(norm_unclipped[i : i + WINDOW_SIZE])
            win_raw.append(arr[i : i + WINDOW_SIZE])

        return (
            np.array(win_norm_clipped, dtype=np.float32),
            np.array(win_norm_unclipped, dtype=np.float32),
            np.array(win_raw, dtype=np.float32),
        )

    # Stride=1 for training (~6,526 windows); Stride=WINDOW_SLIDE (4) for val/test
    X_train, X_train_unclipped, X_train_raw = make_windows(train_raw, stride=1)
    X_val, X_val_unclipped, X_val_raw = make_windows(val_raw, stride=WINDOW_SLIDE)
    X_test, X_test_unclipped, X_test_raw = make_windows(test_raw, stride=WINDOW_SLIDE)
    X_test_30, X_test_30_unclipped, X_test_30_raw = make_windows(test_30_raw, stride=WINDOW_SLIDE)

    out_path = os.path.join(split_dir, f"{DATASET_NAME}_windows.npz")
    np.savez_compressed(
        out_path,
        X_train=X_train,                        # (6526, 24, 12) stride=1, clipped [-4, 4]
        X_train_unclipped=X_train_unclipped,    # (6526, 24, 12) stride=1, unclipped
        X_val=X_val,                            # (346, 24, 12)  stride=4 (70-85%)
        X_test=X_test,                          # (346, 24, 12)  stride=4 (85-100%)
        X_test_30pct=X_test_30,                 # (696, 24, 12)  stride=4 (full 70-100% holdout)
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        X_test_30pct_raw=X_test_30_raw,
        mean=mean,
        std=std,
        sensor_names=np.array(cols),
    )

    print("=" * 68)
    print(f"[Batch Split Exporter] Saved splits to: {out_path}")
    print(f"[Batch Split Exporter] Saved shared scaler to: {scaler_path}")
    print(f"  -> Sensors ({len(cols)}): {cols}")
    print(f"  -> X_train (stride=1, 70%)      : {X_train.shape}")
    print(f"  -> X_val   (stride={WINDOW_SLIDE}, 70-85%)  : {X_val.shape}")
    print(f"  -> X_test  (stride={WINDOW_SLIDE}, 85-100%) : {X_test.shape}")
    print(f"  -> X_test_30pct (stride={WINDOW_SLIDE}, 30%): {X_test_30.shape}")
    print("=" * 68)


if __name__ == "__main__":
    create_chronological_splits()