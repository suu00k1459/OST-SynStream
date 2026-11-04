"""
Batch 1 - Train a diffusion model (DDPM, Ho et al. 2020) on the team's shared UCI split.

Run from the repository root (or anywhere):
    python synthetic_gen/train_diffusion.py

Input (Faisal's prepare_batch_splits.py):
    data/splits/UCI_AirQuality_windows.npz
    Only X_train, X_val, mean, std and sensor_names are read (via train_timevae.load_split).
    X_test / X_test_30pct are never loaded: they are the Batch 3 test period.

Outputs:
    synthetic_gen/models/diffusion_uci.pt                 trained denoiser + settings
    synthetic_gen/outputs/uci_diffusion.npy               synthetic windows (N, 24, 12), same z-scale as X_train
    synthetic_gen/outputs/real_vs_synthetic_diffusion.png quick visual check in real units

How it works:
    Forward (fixed, no learning): add a little Gaussian noise to a real window, step by step,
        until after T steps it is pure noise.
    Denoiser (learned): given a noisy window and the step number t, predict the noise that was added.
    Generation: start from pure noise and remove the predicted noise one step at a time
        (T steps) -> a brand-new window.

    X_val is used only to watch the denoising loss and keep the best epoch.
"""

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_timevae import load_split, plot_real_vs_synthetic

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent            # .../synthetic_gen
MODEL_PATH = ROOT / "models" / "diffusion_uci.pt"
OUT_DIR = ROOT / "outputs"
SYNTH_PATH = OUT_DIR / "uci_diffusion.npy"

CONFIG = {
    "T": 200,              # number of noise steps (more = finer, but slower generation)
    "channels": 64,        # width of the denoiser
    "n_blocks": 4,         # residual blocks (dilations 1, 2, 4, 8 -> sees the whole 24h window)
    "epochs": 100,
    "batch_size": 128,
    "lr": 2e-4,
    "ema_decay": 0.999,    # smoothed copy of the weights, usually gives cleaner samples
    "n_generate": 3000,
    "gen_batch": 500,      # windows generated at once
    "clip": 4.0,           # X_train is clipped to +-4
    "seed": 42,
}


# ---------------------------------------------------------------------------
# 1. Noise schedule
# ---------------------------------------------------------------------------
def cosine_schedule(T, s=0.008):
    """How much noise is added at each step (Nichol & Dhariwal 2021). Returns betas (T,)."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class Diffusion:
    def __init__(self, T):
        self.T = T
        self.betas = cosine_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        """Jump straight to step t: x_t = sqrt(a_bar) * x0 + sqrt(1 - a_bar) * noise"""
        a = self.alpha_bar[t].view(-1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, seq_len, n_features, clip):
        """Start from pure noise and denoise T times.
        Each step: (1) estimate the clean window x0 from the predicted noise,
        (2) clip it to the data range +-clip, (3) move one step back towards it.
        Clipping x0 keeps the first (very noisy) steps from blowing up."""
        x = torch.randn(n, seq_len, n_features)
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, dtype=torch.long)
            eps = model(x, tt)
            a_bar = self.alpha_bar[t]
            a_bar_prev = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
            b, a = self.betas[t], self.alphas[t]

            x0_hat = ((x - (1 - a_bar).sqrt() * eps) / a_bar.sqrt()).clamp(-clip, clip)
            coef_x0 = b * a_bar_prev.sqrt() / (1 - a_bar)
            coef_xt = (1 - a_bar_prev) * a.sqrt() / (1 - a_bar)
            mean = coef_x0 * x0_hat + coef_xt * x
            if t > 0:
                var = b * (1 - a_bar_prev) / (1 - a_bar)              # posterior variance
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


# ---------------------------------------------------------------------------
# 2. Denoiser network
# ---------------------------------------------------------------------------
def time_embedding(t, dim):
    """Sinusoidal encoding of the step number t -> (batch, dim)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    """Dilated 1D convolution over time, told which noise step we are at."""

    def __init__(self, ch, dilation, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.t_proj = nn.Linear(t_dim, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, h, t_emb):
        out = self.conv1(F.silu(self.norm1(h)))
        out = out + self.t_proj(t_emb)[:, :, None]
        out = self.conv2(F.silu(self.norm2(out)))
        return h + out


class Denoiser(nn.Module):
    """Input: noisy window (batch, 24, 12) and step t.  Output: predicted noise (batch, 24, 12)."""

    def __init__(self, n_features, ch, n_blocks):
        super().__init__()
        self.t_dim = ch
        self.t_mlp = nn.Sequential(nn.Linear(ch, ch * 2), nn.SiLU(), nn.Linear(ch * 2, ch))
        self.inp = nn.Conv1d(n_features, ch, 1)
        self.blocks = nn.ModuleList([ResBlock(ch, 2 ** i, ch) for i in range(n_blocks)])
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv1d(ch, n_features, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        t_emb = self.t_mlp(time_embedding(t, self.t_dim))
        h = self.inp(x.transpose(1, 2))          # (batch, features, time) for Conv1d
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h).transpose(1, 2)       # back to (batch, time, features)


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def denoise_loss(model, diff, x0):
    t = torch.randint(0, diff.T, (len(x0),))
    noise = torch.randn_like(x0)
    return F.mse_loss(model(diff.add_noise(x0, t, noise), t), noise)


@torch.no_grad()
def val_loss(model, diff, x_val, seed):
    """Fixed random steps/noise so epochs are comparable."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(0, diff.T, (len(x_val),), generator=g)
    noise = torch.randn(x_val.shape, generator=g)
    return F.mse_loss(model(diff.add_noise(x_val, t, noise), t), noise).item()


def train(cfg, split):
    torch.manual_seed(cfg["seed"])
    x_train = torch.from_numpy(split["X_train"])
    x_val = torch.from_numpy(split["X_val"])
    n_windows, seq_len, n_features = x_train.shape
    print(f"X_train {tuple(x_train.shape)} | X_val {tuple(x_val.shape)} | sensors: {split['sensor_names']}")

    diff = Diffusion(cfg["T"])
    model = Denoiser(n_features, cfg["channels"], cfg["n_blocks"])
    ema = copy.deepcopy(model).eval()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loader = torch.utils.data.DataLoader(x_train, batch_size=cfg["batch_size"], shuffle=True)

    best_val, best_state, best_epoch = float("inf"), None, 0
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for x0 in loader:
            loss = denoise_loss(model, diff, x0)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            # Smoothed (EMA) copy. Warm-up: follows the model closely at first,
            # then averages over more and more steps (up to ema_decay).
            decay = min(cfg["ema_decay"], (1 + step) / (10 + step))
            with torch.no_grad():
                for p_e, p in zip(ema.parameters(), model.parameters()):
                    p_e.mul_(decay).add_(p, alpha=1 - decay)
            total += loss.item() * len(x0)

        v = val_loss(ema, diff, x_val, cfg["seed"])
        flag = ""
        if v < best_val:
            best_val, best_epoch = v, epoch
            best_state = copy.deepcopy(ema.state_dict())
            flag = "  <- best so far"
        if epoch == 1 or epoch % 10 == 0 or flag:
            print(f"epoch {epoch:3d} | train loss {total / n_windows:.4f} | val loss (EMA) {v:.4f}{flag}")

    ema.load_state_dict(best_state)
    print(f"Best epoch by X_val loss: {best_epoch} (val loss {best_val:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": ema.state_dict(), "config": cfg, "best_epoch": best_epoch,
         "seq_len": seq_len, "n_features": n_features,
         "sensor_names": split["sensor_names"],
         "scaler_mean": split["mean"], "scaler_std": split["std"]},
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")
    return ema, diff, seq_len, n_features


# ---------------------------------------------------------------------------
# 4. Generate (same z-scale as X_train)
# ---------------------------------------------------------------------------
def generate(model, diff, seq_len, n_features, cfg):
    model.eval()
    parts, done = [], 0
    while done < cfg["n_generate"]:
        n = min(cfg["gen_batch"], cfg["n_generate"] - done)
        parts.append(diff.sample(model, n, seq_len, n_features, cfg["clip"]).numpy())
        done += n
        print(f"  generated {done}/{cfg['n_generate']}")
    synth = np.clip(np.concatenate(parts), -cfg["clip"], cfg["clip"]).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SYNTH_PATH, synth)
    print(f"Saved {synth.shape} synthetic windows (scaled like X_train) -> {SYNTH_PATH}")
    return synth


if __name__ == "__main__":
    split = load_split()
    model, diff, seq_len, n_features = train(CONFIG, split)
    synth = generate(model, diff, seq_len, n_features, CONFIG)
    plot_real_vs_synthetic(split, synth, filename="real_vs_synthetic_diffusion.png")