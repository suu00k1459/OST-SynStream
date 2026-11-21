"""
Batch 1 - Train TimeGAN (Yoon et al., 2019) on the team's shared UCI split.

Run from the repository root (or anywhere):
    python synthetic_gen/train_timegan.py

Input (Faisal's prepare_batch_splits.py):
    data/splits/UCI_AirQuality_windows.npz
    Only X_train, X_val, mean, std and sensor_names are read (via train_timevae.load_split).
    X_test / X_test_30pct are never loaded: they are the Batch 3 test period.

Outputs:
    synthetic_gen/models/timegan_uci.pt                 trained networks + settings
    synthetic_gen/outputs/uci_timegan.npy               synthetic windows (N, 24, 12), same z-scale as X_train
    synthetic_gen/outputs/real_vs_synthetic_timegan.png quick visual check in real units

How TimeGAN works (5 small GRU networks):
    Embedder   real window X        -> hidden code H        (compress)
    Recovery   hidden code H        -> window X~            (decompress)
    Generator  random noise Z       -> fake hidden code E^
    Supervisor hidden code at t     -> hidden code at t+1   (teaches "what comes next")
    Discriminator hidden code       -> real or fake?

Training has 3 phases:
    1. Autoencoder  : Embedder + Recovery learn to compress and rebuild real windows.
    2. Supervised   : Supervisor learns the step-to-step dynamics on real hidden codes.
    3. Joint (GAN)  : Generator tries to fool the Discriminator, while keeping the dynamics
                      and the mean/variance of real data. X_val is used only to pick the
                      best checkpoint (lowest MMD to X_val).
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_timevae import load_split, plot_real_vs_synthetic

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent            # .../synthetic_gen
MODEL_PATH = ROOT / "models" / "timegan_uci.pt"
OUT_DIR = ROOT / "outputs"
SYNTH_PATH = OUT_DIR / "uci_timegan.npy"

CONFIG = {
    "hidden_dim": 24,          # size of the hidden code H
    "num_layers": 3,           # GRU layers in every network
    "batch_size": 128,
    "lr": 1e-3,
    "steps_embedding": 2000,   # phase 1 iterations
    "steps_supervised": 2000,  # phase 2 iterations
    "steps_joint": 2000,       # phase 3 iterations
    "gamma": 1.0,              # weight of the "fake hidden code from generator" loss
    "eval_every": 200,         # check MMD on X_val every N joint steps
    "n_generate": 3000,        # number of synthetic windows to save
    "clip": 4.0,               # X_train is clipped to +-4
    "seed": 42,
}


# ---------------------------------------------------------------------------
# 1. Scaling: TimeGAN works in [0, 1] (sigmoid outputs)
# ---------------------------------------------------------------------------
def to_unit(x, clip):
    """z-scale clipped to [-clip, clip]  ->  [0, 1]"""
    return (x + clip) / (2 * clip)


def from_unit(x01, clip):
    """[0, 1]  ->  z-scale in [-clip, clip] (same scale as X_train)"""
    return x01 * (2 * clip) - clip


# ---------------------------------------------------------------------------
# 2. Networks
# ---------------------------------------------------------------------------
class RNNBlock(nn.Module):
    """GRU stack + linear layer (+ optional sigmoid). The building block of every TimeGAN part."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, sigmoid=True):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.sigmoid = sigmoid

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return torch.sigmoid(out) if self.sigmoid else out


class TimeGAN(nn.Module):
    def __init__(self, n_features, hidden_dim, num_layers):
        super().__init__()
        self.n_features = n_features
        self.embedder = RNNBlock(n_features, hidden_dim, hidden_dim, num_layers)
        self.recovery = RNNBlock(hidden_dim, hidden_dim, n_features, num_layers)
        self.generator = RNNBlock(n_features, hidden_dim, hidden_dim, num_layers)
        self.supervisor = RNNBlock(hidden_dim, hidden_dim, hidden_dim, max(num_layers - 1, 1))
        self.discriminator = RNNBlock(hidden_dim, hidden_dim, 1, num_layers, sigmoid=False)

    def noise(self, n, seq_len):
        return torch.rand(n, seq_len, self.n_features)

    @torch.no_grad()
    def generate(self, n, seq_len):
        e_hat = self.generator(self.noise(n, seq_len))
        h_hat = self.supervisor(e_hat)
        return self.recovery(h_hat)          # in [0, 1]


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def batch_iter(x, batch_size, rng):
    """Endless random mini-batches from X_train."""
    while True:
        idx = rng.choice(len(x), batch_size, replace=False)
        yield x[idx]


def moment_loss(x_hat, x):
    """Match the mean and standard deviation of fake and real batches."""
    std_fake = torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6)
    std_real = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
    v1 = torch.mean(torch.abs(std_fake - std_real))
    v2 = torch.mean(torch.abs(x_hat.mean(dim=0) - x.mean(dim=0)))
    return v1 + v2


def mmd_rbf(x, y):
    """Maximum Mean Discrepancy with an RBF kernel (median heuristic). Lower = closer."""
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    xy = np.vstack([x, y])
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    k = np.exp(-d2 / np.median(d2[d2 > 0]))
    n = len(x)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean())


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def train(cfg, split):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    x_train = to_unit(split["X_train"], cfg["clip"]).astype(np.float32)
    x_val = split["X_val"]                                  # z-scale, only for checkpoint choice
    n_windows, seq_len, n_features = x_train.shape
    print(f"X_train {x_train.shape} | X_val {x_val.shape} | sensors: {split['sensor_names']}")

    model = TimeGAN(n_features, cfg["hidden_dim"], cfg["num_layers"])
    batches = batch_iter(torch.from_numpy(x_train), cfg["batch_size"], rng)

    opt_er = torch.optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=cfg["lr"])
    opt_gs = torch.optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=cfg["lr"])
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg["lr"])

    # ---- Phase 1: autoencoder (Embedder + Recovery) ----
    print("\n[Phase 1/3] Embedding network")
    for step in range(1, cfg["steps_embedding"] + 1):
        x = next(batches)
        x_tilde = model.recovery(model.embedder(x))
        loss = 10 * torch.sqrt(mse(x_tilde, x))
        opt_er.zero_grad(); loss.backward(); opt_er.step()
        if step == 1 or step % 500 == 0:
            print(f"  step {step:5d} | recon loss {loss.item():.4f}")

    # ---- Phase 2: supervised dynamics (Supervisor, + Generator params in optimizer) ----
    print("\n[Phase 2/3] Supervised dynamics")
    for step in range(1, cfg["steps_supervised"] + 1):
        x = next(batches)
        with torch.no_grad():
            h = model.embedder(x)
        h_sup = model.supervisor(h)
        loss = mse(h_sup[:, :-1], h[:, 1:])                 # predict the next hidden step
        opt_gs.zero_grad(); loss.backward(); opt_gs.step()
        if step == 1 or step % 500 == 0:
            print(f"  step {step:5d} | supervised loss {loss.item():.4f}")

    # ---- Phase 3: joint adversarial training ----
    print("\n[Phase 3/3] Joint training (X_val used only to pick the best checkpoint)")
    best_mmd, best_state, best_step = float("inf"), None, 0
    rng_eval = np.random.default_rng(cfg["seed"] + 1)

    for step in range(1, cfg["steps_joint"] + 1):
        # (a) Generator + Supervisor: twice per step, as in the original paper
        for _ in range(2):
            x = next(batches)
            z = model.noise(len(x), seq_len)
            h = model.embedder(x)
            e_hat = model.generator(z)
            h_hat = model.supervisor(e_hat)
            h_sup = model.supervisor(h)
            x_hat = model.recovery(h_hat)
            y_fake = model.discriminator(h_hat)
            y_fake_e = model.discriminator(e_hat)

            g_adv = bce(y_fake, torch.ones_like(y_fake))
            g_adv_e = bce(y_fake_e, torch.ones_like(y_fake_e))
            g_sup = mse(h_sup[:, :-1], h[:, 1:])
            g_mom = moment_loss(x_hat, x)
            g_loss = g_adv + cfg["gamma"] * g_adv_e + 100 * torch.sqrt(g_sup) + 100 * g_mom
            opt_gs.zero_grad(); g_loss.backward(); opt_gs.step()

            # Embedder + Recovery keep reconstructing well (with a little supervised signal)
            h = model.embedder(x)
            x_tilde = model.recovery(h)
            h_sup = model.supervisor(h)
            e_loss = 10 * torch.sqrt(mse(x_tilde, x)) + 0.1 * mse(h_sup[:, :-1], h[:, 1:])
            opt_er.zero_grad(); e_loss.backward(); opt_er.step()

        # (b) Discriminator: only update when it is not already winning easily
        x = next(batches)
        z = model.noise(len(x), seq_len)
        with torch.no_grad():
            h = model.embedder(x)
            e_hat = model.generator(z)
            h_hat = model.supervisor(e_hat)
        y_real = model.discriminator(h)
        y_fake = model.discriminator(h_hat)
        y_fake_e = model.discriminator(e_hat)
        d_loss = (bce(y_real, torch.ones_like(y_real))
                  + bce(y_fake, torch.zeros_like(y_fake))
                  + cfg["gamma"] * bce(y_fake_e, torch.zeros_like(y_fake_e)))
        if d_loss.item() > 0.15:
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # (c) Checkpoint choice on X_val (tuning set only)
        if step % cfg["eval_every"] == 0 or step == cfg["steps_joint"]:
            fake = from_unit(model.generate(len(x_val), seq_len).numpy(), cfg["clip"])
            idx = rng_eval.choice(len(fake), min(300, len(fake)), replace=False)
            val_idx = rng_eval.choice(len(x_val), min(300, len(x_val)), replace=False)
            mmd = mmd_rbf(x_val[val_idx], fake[idx])
            flag = ""
            if mmd < best_mmd:
                best_mmd, best_step = mmd, step
                best_state = copy.deepcopy(model.state_dict())
                flag = "  <- best so far"
            print(f"  step {step:5d} | D {d_loss.item():.4f} | G_adv {g_adv.item():.4f} "
                  f"| G_sup {g_sup.item():.4f} | G_mom {g_mom.item():.4f} | MMD(val) {mmd:.4f}{flag}")

    model.load_state_dict(best_state)
    print(f"\nBest joint step by MMD to X_val: {best_step} (MMD {best_mmd:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "best_step": best_step,
         "seq_len": seq_len, "n_features": n_features,
         "sensor_names": split["sensor_names"],
         "scaler_mean": split["mean"], "scaler_std": split["std"]},
        MODEL_PATH,
    )
    print(f"Saved model -> {MODEL_PATH}")
    return model, seq_len


# ---------------------------------------------------------------------------
# 5. Generate (same z-scale as X_train)
# ---------------------------------------------------------------------------
def generate(model, seq_len, cfg):
    model.eval()
    synth01 = model.generate(cfg["n_generate"], seq_len).numpy()
    synth = np.clip(from_unit(synth01, cfg["clip"]), -cfg["clip"], cfg["clip"]).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SYNTH_PATH, synth)
    print(f"Saved {synth.shape} synthetic windows (scaled like X_train) -> {SYNTH_PATH}")
    return synth


if __name__ == "__main__":
    split = load_split()
    model, seq_len = train(CONFIG, split)
    synth = generate(model, seq_len, CONFIG)
    plot_real_vs_synthetic(split, synth, filename="real_vs_synthetic_timegan.png")