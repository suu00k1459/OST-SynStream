"""
Batch 1 - Train a TimeVAE-style generator on UCI Air Quality windows.

Run from the repository root:
    python synthetic_gen/train_timevae.py

Outputs:
    synthetic_gen/models/timevae_uci.pt        trained model + settings + scaler
    synthetic_gen/outputs/synthetic_windows.npy synthetic windows (original units)
    synthetic_gen/outputs/real_vs_synthetic.png quick visual check
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Settings - change FEATURES to switch between the 6-column and 12-column setup
# ---------------------------------------------------------------------------
CONFIG = {
    "csv_path": "../stream_analytics/AirQuality.csv",
    # Same 6 columns as stream_analytics/Processed/real_windows.csv
    "features": ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)", "T", "RH"],
    # 12-column option (Flink's 13 sensors minus NMHC(GT), which is 90% missing):
    # "features": ["CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
    #              "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"],
    "seq_len": 24,          # 24 hourly rows = one day (matches WINDOW_SIZE in common/config.py)
    "train_ratio": 0.7,     # first 70% for training, last 30% kept for the live replay
    "hidden_dim": 64,
    "latent_dim": 16,
    "epochs": 60,
    "batch_size": 64,
    "lr": 1e-3,
    "beta": 0.05,           # weight of the KL term (lower = sharper but less varied)
    "max_missing_frac": 0.2,  # drop windows where more than 20% of cells were originally missing
    "seed": 42,
}

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "timevae_uci.pt"
OUT_DIR = ROOT / "outputs"


# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
def load_uci(csv_path, features):
    """Read the raw UCI CSV, turn -200 into missing, fill gaps by interpolation."""
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    data = df[features].apply(pd.to_numeric, errors="coerce").replace(-200, np.nan)
    missing = data.isna()                       # remember which cells were filled in
    data = data.interpolate(method="time", limit_direction="both")
    return data, missing


def split_by_time(data, train_ratio):
    """Chronological split: no shuffling, the model never sees the replay period."""
    cut = int(len(data) * train_ratio)
    return data.iloc[:cut], data.iloc[cut:]


def make_windows(array, seq_len, stride=1):
    """Slide a seq_len-row window down the table -> (n_windows, seq_len, n_features)."""
    starts = range(0, len(array) - seq_len + 1, stride)
    return np.stack([array[s:s + seq_len] for s in starts])


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class TimeVAE(nn.Module):
    """
    Encoder: reads a window and squeezes it into a small code z (latent_dim numbers).
    Decoder: unfolds z back into a full window.
    After training, feeding random z to the decoder produces new, realistic windows.
    """

    def __init__(self, n_features, seq_len, hidden_dim, latent_dim):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.encoder = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.from_z = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.to_out = nn.Linear(hidden_dim, n_features)

    def encode(self, x):
        _, h = self.encoder(x)              # h: (1, batch, hidden)
        h = h.squeeze(0)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        h = self.from_z(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder(h)
        return self.to_out(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def generate(self, n):
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def train(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    data, missing = load_uci(cfg["csv_path"], cfg["features"])
    train_df, replay_df = split_by_time(data, cfg["train_ratio"])
    train_missing, _ = split_by_time(missing, cfg["train_ratio"])
    print(f"Train period : {train_df.index[0]} -> {train_df.index[-1]} ({len(train_df)} rows)")
    print(f"Replay period: {replay_df.index[0]} -> {replay_df.index[-1]} ({len(replay_df)} rows)")

    # Z-score using training statistics only (same idea as Flink's normalization)
    mean = train_df.values.mean(axis=0)
    std = train_df.values.std(axis=0)
    std[std < 1e-6] = 1.0
    train_scaled = (train_df.values - mean) / std

    windows = make_windows(train_scaled, cfg["seq_len"]).astype(np.float32)
    # Long gaps become straight interpolated lines - don't teach those to the model
    miss_frac = make_windows(train_missing.values.astype(float), cfg["seq_len"]).mean(axis=(1, 2))
    windows = windows[miss_frac <= cfg["max_missing_frac"]]
    print(f"Training windows: {windows.shape}  (n_windows, seq_len, n_features)")

    model = TimeVAE(len(cfg["features"]), cfg["seq_len"], cfg["hidden_dim"], cfg["latent_dim"])
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = torch.utils.data.DataLoader(
        torch.from_numpy(windows), batch_size=cfg["batch_size"], shuffle=True
    )

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot = rec = kld = 0.0
        for x in loader:
            recon, mu, logvar = model(x)
            loss, r, k = vae_loss(recon, x, mu, logvar, cfg["beta"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            tot += loss.item() * len(x)
            rec += r.item() * len(x)
            kld += k.item() * len(x)
        if epoch == 1 or epoch % 10 == 0:
            n = len(windows)
            print(f"epoch {epoch:3d} | loss {tot/n:.4f} | recon {rec/n:.4f} | kl {kld/n:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg,
         "scaler_mean": mean, "scaler_std": std},
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")
    return model, windows, mean, std


# ---------------------------------------------------------------------------
# 4. Generate + quick visual check
# ---------------------------------------------------------------------------
def generate_and_plot(model, real_windows, mean, std, cfg, n=500):
    model.eval()
    synth_scaled = model.generate(n).numpy()
    synth = synth_scaled * std + mean                      # back to original units
    real = real_windows * std + mean

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "synthetic_windows.npy", synth)
    print(f"Saved {n} synthetic windows -> {OUT_DIR / 'synthetic_windows.npy'}")

    feats = cfg["features"]
    hours = np.arange(cfg["seq_len"])
    fig, axes = plt.subplots(2, len(feats), figsize=(3.2 * len(feats), 6))
    idx_r = np.random.choice(len(real), 5, replace=False)
    for j, name in enumerate(feats):
        ax = axes[0, j]
        ax.set_xlabel("hour in window")
        for i in idx_r:
            ax.plot(hours, real[i, :, j], color="tab:blue", alpha=0.6, lw=1)
        for i in range(5):
            ax.plot(hours, synth[i, :, j], color="tab:orange", alpha=0.6, lw=1)
        ax.set_title(name)
        if j == 0:
            ax.set_ylabel("5 sample windows")
        ax = axes[1, j]
        bins = np.linspace(real[:, :, j].min(), real[:, :, j].max(), 40)
        ax.hist(real[:, :, j].ravel(), bins=bins, density=True, alpha=0.5, color="tab:blue", label="real")
        ax.hist(synth[:, :, j].ravel(), bins=bins, density=True, alpha=0.5, color="tab:orange", label="synthetic")
        ax.set_xlabel("value")
        if j == 0:
            ax.set_ylabel("value distribution")
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_vs_synthetic.png", dpi=110)
    print(f"Saved plot -> {OUT_DIR / 'real_vs_synthetic.png'}")


if __name__ == "__main__":
    model, windows, mean, std = train(CONFIG)
    generate_and_plot(model, windows, mean, std, CONFIG)