"""
Batch 2 - Evaluate how similar synthetic windows are to real windows.

Run from the repository root (after training at least one model):
    python synthetic_gen/evaluate_batch.py

Three metrics:
  1. MMD                 Are real and synthetic windows drawn from the same distribution?
                         (lower = better, compare with the real-vs-real baseline)
  2. Discriminative score Can a classifier tell real from synthetic?
                         |accuracy - 0.5|  (0 = cannot tell them apart = best)
  3. Predictive score    Is synthetic data useful? (TSTR: Train on Synthetic, Test on Real)
                         A GRU learns to predict the next hour; tested on the unseen replay period.
                         Compare with TRTR (Train on Real, Test on Real). Lower MAE = better.

Also saves a PCA plot so the overlap can be seen by eye.

Outputs:
    synthetic_gen/outputs/evaluation_results.csv
    synthetic_gen/outputs/pca_real_vs_synthetic.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from train_timevae import CONFIG, load_uci, split_by_time, make_windows

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

# Synthetic files to evaluate - missing files are skipped automatically
SYNTHETIC_FILES = {
    "TimeVAE": OUT_DIR / "synthetic_windows.npy",
    "TimeGAN": OUT_DIR / "synthetic_windows_timegan.npy",
}
N_SAMPLES = 500   # windows per group used in each comparison
SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_real_windows(cfg):
    """Real windows in original units: training period and the unseen replay period."""
    data, missing = load_uci(cfg["csv_path"], cfg["features"])
    train_df, test_df = split_by_time(data, cfg["train_ratio"])
    train_miss, test_miss = split_by_time(missing, cfg["train_ratio"])

    def windows(df, miss):
        w = make_windows(df.values, cfg["seq_len"])
        frac = make_windows(miss.values.astype(float), cfg["seq_len"]).mean(axis=(1, 2))
        return w[frac <= cfg["max_missing_frac"]].astype(np.float32)

    train_w = windows(train_df, train_miss)
    test_w = windows(test_df, test_miss)
    mean = train_df.values.mean(axis=0)
    std = train_df.values.std(axis=0)
    std[std < 1e-6] = 1.0
    return train_w, test_w, mean, std


def sample(x, n, rng):
    idx = rng.choice(len(x), min(n, len(x)), replace=False)
    return x[idx]


# ---------------------------------------------------------------------------
# Metric 1: MMD
# ---------------------------------------------------------------------------
def mmd_rbf(x, y):
    """Maximum Mean Discrepancy with an RBF kernel on flattened windows."""
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    xy = np.vstack([x, y])
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    bandwidth = np.median(d2[d2 > 0])                 # median heuristic
    k = np.exp(-d2 / bandwidth)
    n = len(x)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


# ---------------------------------------------------------------------------
# Metric 2: Discriminative score
# ---------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0)).squeeze(-1)


def discriminative_score(real, synth, rng, epochs=40):
    x = np.concatenate([real, synth]).astype(np.float32)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))]).astype(np.float32)
    idx = rng.permutation(len(x))
    cut = int(0.8 * len(x))
    tr, te = idx[:cut], idx[cut:]
    xt, yt = torch.from_numpy(x[tr]), torch.from_numpy(y[tr])

    model = GRUClassifier(x.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for b in torch.split(torch.from_numpy(rng.permutation(len(tr))), 64):
            loss = loss_fn(model(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (model(torch.from_numpy(x[te])) > 0).float().numpy()
    acc = float((pred == y[te]).mean())
    return abs(acc - 0.5), acc


# ---------------------------------------------------------------------------
# Metric 3: Predictive score (TSTR)
# ---------------------------------------------------------------------------
class GRUForecaster(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_features)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


def predictive_mae(train_windows, test_windows, rng, epochs=40):
    """Learn hour t -> hour t+1 on train_windows, report MAE on test_windows."""
    xtr = torch.from_numpy(train_windows.astype(np.float32))
    xte = torch.from_numpy(test_windows.astype(np.float32))
    model = GRUForecaster(xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for b in torch.split(torch.from_numpy(rng.permutation(len(xtr))), 64):
            x = xtr[b]
            loss = nn.functional.l1_loss(model(x[:, :-1]), x[:, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(nn.functional.l1_loss(model(xte[:, :-1]), xte[:, 1:]))


# ---------------------------------------------------------------------------
# PCA plot
# ---------------------------------------------------------------------------
def pca_plot(real, synths):
    flat = lambda a: a.reshape(len(a), -1)
    pca = PCA(n_components=2).fit(flat(real))
    fig, axes = plt.subplots(1, len(synths), figsize=(5 * len(synths), 4.5), squeeze=False)
    for ax, (name, s) in zip(axes[0], synths.items()):
        r2, s2 = pca.transform(flat(real)), pca.transform(flat(s))
        ax.scatter(r2[:, 0], r2[:, 1], s=8, alpha=0.4, color="tab:blue", label="real")
        ax.scatter(s2[:, 0], s2[:, 1], s=8, alpha=0.4, color="tab:orange", label="synthetic")
        ax.set_title(f"PCA: real vs {name}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_real_vs_synthetic.png", dpi=110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    train_w, test_w, mean, std = load_real_windows(CONFIG)
    z = lambda a: ((a - mean) / std).astype(np.float32)     # same scale for every group
    print(f"Real windows: train {train_w.shape}, unseen replay period {test_w.shape}")

    synths = {}
    for name, path in SYNTHETIC_FILES.items():
        if path.exists():
            s = np.load(path)
            if s.shape[1:] != train_w.shape[1:]:
                print(f"Skip {name}: shape {s.shape[1:]} != real {train_w.shape[1:]} "
                      f"(retrain it with the current features)")
                continue
            synths[name] = z(sample(s, N_SAMPLES, rng))
        else:
            print(f"Skip {name}: {path.name} not found (train the model first)")
    if not synths:
        print("No synthetic data to evaluate.")
        return

    real = z(sample(train_w, N_SAMPLES, rng))
    real_other = z(sample(train_w, N_SAMPLES, rng))
    test = z(test_w)

    # Baseline row: real vs another real sample = the best score we can hope for
    rows = []
    print("\n[Baseline] real vs real ...")
    trtr = predictive_mae(z(sample(train_w, N_SAMPLES, rng)), test, rng)
    d_score, d_acc = discriminative_score(real, real_other, rng)
    rows.append({"model": "Real (baseline)", "MMD": mmd_rbf(real, real_other),
                 "discriminative_score": d_score, "discriminator_acc": d_acc,
                 "predictive_MAE": trtr})

    for name, s in synths.items():
        print(f"[{name}] computing metrics ...")
        d_score, d_acc = discriminative_score(real, s, rng)
        rows.append({"model": name, "MMD": mmd_rbf(real, s),
                     "discriminative_score": d_score, "discriminator_acc": d_acc,
                     "predictive_MAE": predictive_mae(s, test, rng)})

    df = pd.DataFrame(rows).set_index("model").round(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "evaluation_results.csv")
    pca_plot(real, synths)

    print("\n" + df.to_string())
    print("\nHow to read:")
    print("  MMD                  lower = closer to real (baseline shows the noise floor)")
    print("  discriminative_score 0 = classifier cannot tell real from synthetic (best), 0.5 = always can")
    print("  predictive_MAE       train on this data, test on the unseen replay period; close to baseline = useful")
    print(f"\nSaved -> {OUT_DIR / 'evaluation_results.csv'}")
    print(f"Saved -> {OUT_DIR / 'pca_real_vs_synthetic.png'}")


if __name__ == "__main__":
    main()
