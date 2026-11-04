"""
Batch 1 - Train a TimeVAE-style generator on the team's shared UCI split.

Run from the repository root (or anywhere):
    python synthetic_gen/train_timevae.py

Input (Faisal's prepare_batch_splits.py):
    data/splits/UCI_AirQuality_windows.npz
    Only X_train, X_val, mean, std and sensor_names are read.
    X_test / X_test_30pct are never loaded: they are the Batch 3 test period.

Outputs:
    synthetic_gen/models/timevae_uci.pt         trained model + settings
    synthetic_gen/outputs/uci_timevae.npy       synthetic windows (N, 24, 12), same z-scale as X_train
    synthetic_gen/outputs/real_vs_synthetic_timevae.png quick visual check in real units
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent            # .../synthetic_gen
REPO = ROOT.parent                                 # repository root
SPLIT_PATH = REPO / "data" / "splits" / "UCI_AirQuality_windows.npz"
MODEL_PATH = ROOT / "models" / "timevae_uci.pt"
OUT_DIR = ROOT / "outputs"
SYNTH_PATH = OUT_DIR / "uci_timevae.npy"

CONFIG = {
    "hidden_dim": 64,
    "latent_dim": 16,
    "epochs": 60,
    "batch_size": 64,
    "lr": 1e-3,
    "beta": 0.05,        # weight of the KL term (lower = sharper but less varied)
    "n_generate": 3000,  # number of synthetic windows to save
    "clip": 4.0,         # X_train is clipped to +-4, so synthetic output is clipped the same way
    "seed": 42,
}


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def load_split(path=SPLIT_PATH):
    """Read only the arrays we are allowed to use. np.load on .npz is lazy,
    so the test arrays are never touched."""
    d = np.load(path)
    return {
        "X_train": d["X_train"].astype(np.float32),   # (6526, 24, 12) scaled + clipped
        "X_val": d["X_val"].astype(np.float32),       # (346, 24, 12)  for tuning only
        "mean": d["mean"].astype(np.float32),         # (12,) real = x * std + mean
        "std": d["std"].astype(np.float32),
        "sensor_names": [str(s) for s in d["sensor_names"]],
    }


def to_real_units(x, mean, std):
    return x * std + mean


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
# 3. Training (X_train to learn, X_val only to watch / pick the best epoch)
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_loss(model, x_val, beta):
    model.eval()
    recon, mu, logvar = model(x_val)
    loss, rec, _ = vae_loss(recon, x_val, mu, logvar, beta)
    return loss.item(), rec.item()


def train(cfg, split):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    x_train = split["X_train"]
    x_val = torch.from_numpy(split["X_val"])
    n_windows, seq_len, n_features = x_train.shape
    print(f"X_train {x_train.shape} | X_val {tuple(x_val.shape)} | sensors: {split['sensor_names']}")

    model = TimeVAE(n_features, seq_len, cfg["hidden_dim"], cfg["latent_dim"])
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = torch.utils.data.DataLoader(
        torch.from_numpy(x_train), batch_size=cfg["batch_size"], shuffle=True
    )

    best_val, best_state, best_epoch = float("inf"), None, 0
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

        v_loss, v_rec = val_loss(model, x_val, cfg["beta"])
        if v_loss < best_val:
            best_val, best_epoch = v_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train loss {tot/n_windows:.4f} "
                  f"(recon {rec/n_windows:.4f}, kl {kld/n_windows:.4f}) | "
                  f"val loss {v_loss:.4f} (recon {v_rec:.4f})")

    model.load_state_dict(best_state)
    print(f"Best epoch by X_val loss: {best_epoch} (val loss {best_val:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "best_epoch": best_epoch,
         "seq_len": seq_len, "n_features": n_features,
         "sensor_names": split["sensor_names"],
         "scaler_mean": split["mean"], "scaler_std": split["std"]},
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# 4. Generate (scaled) + plot (real units)
# ---------------------------------------------------------------------------
def generate(model, cfg):
    model.eval()
    synth = model.generate(cfg["n_generate"]).numpy()
    synth = np.clip(synth, -cfg["clip"], cfg["clip"]).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SYNTH_PATH, synth)
    print(f"Saved {synth.shape} synthetic windows (scaled like X_train) -> {SYNTH_PATH}")
    return synth


def plot_real_vs_synthetic(split, synth, filename="real_vs_synthetic_timevae.png", n_lines=5):
    """Shared by train_timevae.py and train_timegan.py (each passes its own filename)."""
    mean, std, names = split["mean"], split["std"], split["sensor_names"]
    real = to_real_units(split["X_train"], mean, std)
    fake = to_real_units(synth, mean, std)
    seq_len = real.shape[1]
    hours = np.arange(seq_len)
    rng = np.random.default_rng(0)
    idx_r = rng.choice(len(real), n_lines, replace=False)

    n_cols = 6
    n_groups = int(np.ceil(len(names) / n_cols))       # 12 sensors -> 2 groups of 6
    fig, axes = plt.subplots(2 * n_groups, n_cols, figsize=(3.2 * n_cols, 3.0 * 2 * n_groups))
    for j, name in enumerate(names):
        g, c = divmod(j, n_cols)
        ax = axes[2 * g, c]                              # top: sample windows
        for i in idx_r:
            ax.plot(hours, real[i, :, j], color="tab:blue", alpha=0.6, lw=1)
        for i in range(n_lines):
            ax.plot(hours, fake[i, :, j], color="tab:orange", alpha=0.6, lw=1)
        ax.set_title(name)
        ax.set_xlabel("hour in window")

        ax = axes[2 * g + 1, c]                          # bottom: value distribution
        bins = np.linspace(real[:, :, j].min(), real[:, :, j].max(), 40)
        ax.hist(real[:, :, j].ravel(), bins=bins, density=True, alpha=0.5, color="tab:blue", label="real")
        ax.hist(fake[:, :, j].ravel(), bins=bins, density=True, alpha=0.5, color="tab:orange", label="synthetic")
        ax.set_xlabel("value (real units)")
        if c == 0:
            ax.legend()
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, dpi=110)
    plt.close(fig)
    print(f"Saved plot -> {OUT_DIR / filename}")


if __name__ == "__main__":
    split = load_split()
    model = train(CONFIG, split)
    synth = generate(model, CONFIG)
    plot_real_vs_synthetic(split, synth)