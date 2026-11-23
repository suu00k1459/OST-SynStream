"""
HAI: train the same GRU TimeVAE on Hazem's HAI windows.
Only the data loading and file names differ from train_timevae.py.

Run from the repository root:
    python synthetic_gen/train_timevae_hai.py
"""
import json
from pathlib import Path
import time
import numpy as np

import train_timevae as tv

ROOT = Path(__file__).resolve().parent                        # .../synthetic_gen
HAI_DIR = ROOT.parent / "stream_analytics" / "results"


def load_hai_split(val_ratio=0.1, clip=4.0):
    """Hazem's HAI windows -> same dict shape as load_split() for UCI."""
    X = np.load(HAI_DIR / "hai_train_windows.npy").astype(np.float32)   # (9358, 24, 15)
    X = np.clip(X, -clip, clip)                                         # like Flink's live windows
    scaler = np.load(HAI_DIR / "hai_scaler.npz")
    with open(HAI_DIR / "hai_sensors.json", encoding="utf-8") as f:
        names = json.load(f)
    cut = int(len(X) * (1 - val_ratio))                                 # last 10% = validation
    return {
        "X_train": X[:cut],
        "X_val": X[cut:],
        "mean": scaler["mean"].astype(np.float32),
        "std": scaler["std"].astype(np.float32),
        "sensor_names": names,
    }


if __name__ == "__main__":
    tv.MODEL_PATH = ROOT / "models" / "timevae_hai.pt"
    tv.SYNTH_PATH = tv.OUT_DIR / "hai_timevae.npy"

    t0 = time.perf_counter()
    split = load_hai_split(clip=tv.CONFIG["clip"])
    t1 = time.perf_counter()
    model = tv.train(tv.CONFIG, split)
    t2 = time.perf_counter()
    synth = tv.generate(model, tv.CONFIG)
    t3 = time.perf_counter()
    tv.plot_real_vs_synthetic(split, synth, filename="real_vs_synthetic_hai_timevae.png")
    t4 = time.perf_counter()

    print("\n--- time ---")
    print(f"load data : {t1 - t0:7.1f} s")
    print(f"training  : {t2 - t1:7.1f} s  ({(t2 - t1) / tv.CONFIG['epochs']:.2f} s per epoch)")
    print(f"generate  : {t3 - t2:7.1f} s  ({tv.CONFIG['n_generate']} windows)")
    print(f"plot      : {t4 - t3:7.1f} s")
    print(f"total     : {t4 - t0:7.1f} s")