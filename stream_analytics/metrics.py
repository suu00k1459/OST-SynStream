"""
Similarity, temporal and privacy metrics between REAL and SYNTHETIC windows.

Used in:
  * Batch 2 (shared with Suyoung) - on many windows at once
  * Pipeline 5 (window_comparator.py) - on one real/synthetic pair at a time

All functions accept windows shaped (N, L, D) or a single window (L, D).
Lower is better (= more similar) for every metric here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


def _as3d(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return w[None] if w.ndim == 2 else w


def _points(w) -> np.ndarray:
    """(N, L, D) -> (N*L, D): every time step is one D-dimensional sample."""
    w = _as3d(w)
    return w.reshape(-1, w.shape[-1])


# --------------------------------------------------------------------------
# 1. Distribution similarity
# --------------------------------------------------------------------------
def mmd_rbf(X, Y, gamma: float | None = None, max_n: int = 2000, seed: int = 0) -> float:
    """
    Maximum Mean Discrepancy with an RBF kernel (biased estimator).
    X, Y: 2-D sample matrices (n, d). gamma defaults to the median heuristic.
    0 means identical distributions.
    """
    rng = np.random.default_rng(seed)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    def sqdist(A, B):
        return (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T

    dxx, dyy, dxy = sqdist(X, X), sqdist(Y, Y), sqdist(X, Y)
    if gamma is None:
        med = np.median(dxy[dxy > 0]) if np.any(dxy > 0) else 1.0
        gamma = 1.0 / med
    kxx, kyy, kxy = np.exp(-gamma * dxx), np.exp(-gamma * dyy), np.exp(-gamma * dxy)
    return float(max(kxx.mean() + kyy.mean() - 2 * kxy.mean(), 0.0))


def mmd_windows(real, syn, **kw) -> float:
    """MMD treating every time step as a sample (distribution of values)."""
    return mmd_rbf(_points(real), _points(syn), **kw)


def wasserstein_per_feature(real, syn) -> float:
    """Mean 1-D Wasserstein distance over features (marginal distributions)."""
    r, s = _points(real), _points(syn)
    return float(np.mean([wasserstein_distance(r[:, d], s[:, d]) for d in range(r.shape[1])]))


# --------------------------------------------------------------------------
# 2. Temporal dynamics
# --------------------------------------------------------------------------
def _acf_1d(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    denom = (x**2).sum()
    if denom < 1e-12:
        return np.zeros(nlags)
    return np.array([(x[:-k] * x[k:]).sum() / denom for k in range(1, nlags + 1)])


def acf_matrix(windows, nlags: int = 5) -> np.ndarray:
    """Average autocorrelation per feature -> (D, nlags)."""
    w = _as3d(windows)
    N, L, D = w.shape
    nlags = min(nlags, L - 1)
    out = np.zeros((D, nlags))
    for d in range(D):
        out[d] = np.mean([_acf_1d(w[n, :, d], nlags) for n in range(N)], axis=0)
    return out


def acf_diff(real, syn, nlags: int = 5) -> float:
    """Mean absolute difference of the autocorrelation functions."""
    return float(np.abs(acf_matrix(real, nlags) - acf_matrix(syn, nlags)).mean())


def cross_corr_diff(real, syn) -> float:
    """
    Frobenius norm of the difference between feature correlation matrices,
    divided by D so the value does not grow with the number of sensors.
    Checks whether relations BETWEEN sensors are preserved.
    """
    r, s = _points(real), _points(syn)
    # a sensor that is flat inside the window has no defined correlation:
    # numpy warns and returns NaN, which we treat as 0
    with np.errstate(invalid="ignore", divide="ignore"):
        cr = np.nan_to_num(np.corrcoef(r, rowvar=False))
        cs = np.nan_to_num(np.corrcoef(s, rowvar=False))
    return float(np.linalg.norm(cr - cs, "fro") / r.shape[1])


# --------------------------------------------------------------------------
# 3. Privacy: Distance to Closest Record (DCR)
# --------------------------------------------------------------------------
class DCR:
    """
    For each synthetic window: distance to the nearest REAL TRAINING window.
    Very small distances mean the generator copies training data (privacy leak).

    Interpretation: compare with the same distance computed for a real
    HOLD-OUT window (`baseline_`). ratio = dcr_syn / baseline:
        ratio << 1  -> synthetic is suspiciously close to training data
        ratio ~ 1   -> about as novel as genuinely new real data (good)
    """

    def __init__(self, train_windows, holdout_windows=None):
        tw = _as3d(train_windows)
        self.nn = NearestNeighbors(n_neighbors=1).fit(tw.reshape(len(tw), -1))
        self.baseline_ = None
        if holdout_windows is not None:
            self.baseline_ = float(np.median(self.distances(holdout_windows)))

    def distances(self, windows) -> np.ndarray:
        w = _as3d(windows)
        d, _ = self.nn.kneighbors(w.reshape(len(w), -1))
        return d[:, 0]

    def score(self, windows) -> dict:
        d = float(np.median(self.distances(windows)))
        out = {"dcr": d}
        if self.baseline_:
            out["dcr_ratio"] = d / self.baseline_
        return out


# --------------------------------------------------------------------------
# All-in-one
# --------------------------------------------------------------------------
def compare_windows(real, syn, nlags: int = 5) -> dict:
    """Every similarity metric for a real/synthetic pair (or two sets)."""
    return {
        "mmd_score": mmd_windows(real, syn),
        "wasserstein": wasserstein_per_feature(real, syn),
        "acf_diff": acf_diff(real, syn, nlags),
        "xcorr_diff": cross_corr_diff(real, syn),
    }


def placeholder_synthetic(real_windows, noise: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    STAND-IN for Suyoung's generator until her TimeGAN/TimeVAE output exists:
    real windows + Gaussian noise (in z-score units). Lets every script run now.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(real_windows, float)
    return r + rng.normal(0, noise, r.shape)


if __name__ == "__main__":
    # sanity check: metrics must grow as synthetic data gets worse
    rng = np.random.default_rng(1)
    t = np.arange(24)
    real = np.stack([np.stack([np.sin(2 * np.pi * (t + p) / 24) for p in range(3)], -1)
                     + rng.normal(0, 0.1, (24, 3)) for _ in range(200)])
    for name, syn in [("good", placeholder_synthetic(real, 0.1)),
                      ("noisy", placeholder_synthetic(real, 0.8)),
                      ("random", rng.normal(0, 1, real.shape))]:
        print(f"{name:7s}", {k: round(v, 4) for k, v in compare_windows(real, syn).items()})
