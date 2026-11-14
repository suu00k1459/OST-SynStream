"""
Check a synthetic-data file BEFORE it is used in Batch 3.
For Suyoung: run this on your .npy file before sending it.

  python check_synthetic.py uci_timegan.npy --dataset uci
  python check_synthetic.py hai_timevae.npy --dataset hai

Needs results/<dataset>_scaler.npz and results/<dataset>_train_windows.npy
(the files Hazem shares). Prints PASS / WARN / FAIL per check.
"""
import argparse
import os
import sys

import numpy as np

from metrics import DCR, compare_windows

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="synthetic windows (.npy)")
    ap.add_argument("--dataset", choices=["uci", "hai"], required=True)
    a = ap.parse_args(argv)

    scaler = np.load(os.path.join(OUT, f"{a.dataset}_scaler.npz"))
    real = np.load(os.path.join(OUT, f"{a.dataset}_train_windows.npy"))
    feats = [str(f) for f in scaler["features" if "features" in scaler.files else "sensor_names"]]
    L, D = real.shape[1], real.shape[2]
    res = []

    def check(name, status, detail):
        res.append(status)
        print(f"{status:4s}  {name:28s} {detail}")

    try:
        syn = np.load(a.file)
    except Exception as e:
        print(f"FAIL  could not load {a.file}: {e}")
        return 1

    ok_shape = syn.ndim == 3 and syn.shape[1:] == (L, D)
    check("shape (N, L, D)", "PASS" if ok_shape else "FAIL",
          f"got {syn.shape}, expected (N, {L}, {D})  [D = {D} features in this order: {feats}]")
    if not ok_shape:
        return 1
    check("number of windows", "PASS" if len(syn) >= 1000 else "WARN",
          f"N = {len(syn)} (1000+ recommended for stable Batch 3 results)")
    finite = np.isfinite(syn).all()
    check("no NaN / inf", "PASS" if finite else "FAIL", "" if finite else f"{(~np.isfinite(syn)).sum()} bad values")
    if not finite:
        return 1

    # scale: data in the fixed training scale has per-feature mean ~0 and std ~1
    m, s = syn.reshape(-1, D).mean(0), syn.reshape(-1, D).std(0)
    raw_like = np.abs(m).max() > 5 or s.max() > 10
    check("fixed z-scale (scaler.npz)", "FAIL" if raw_like else ("PASS" if np.abs(m).max() < 1 and 0.3 < s.min() and s.max() < 3 else "WARN"),
          f"feature means {m.min():+.2f}..{m.max():+.2f}, stds {s.min():.2f}..{s.max():.2f} "
          f"(expected about 0 and 1{'; looks like RAW units: scale with the scaler file' if raw_like else ''})")
    flat = (syn.std(axis=1) < 1e-6).all(axis=1).mean()
    check("windows are not flat", "PASS" if flat < 0.01 else "WARN", f"{flat:.1%} completely flat windows")

    # similarity and privacy (information, not pass/fail on quality)
    rng = np.random.default_rng(0)
    r = real[rng.choice(len(real), min(2000, len(real)), replace=False)]
    sy = syn[rng.choice(len(syn), min(2000, len(syn)), replace=False)]
    sim = compare_windows(r, sy)
    print("INFO  similarity to real training windows (lower = closer): "
          + ", ".join(f"{k} {v:.4f}" for k, v in sim.items()))
    cut = int(len(real) * 0.9)
    dcr = DCR(real[:cut], real[cut:]).score(sy[:500])
    copy = dcr["dcr_ratio"] < 0.2
    check("not copying training data", "WARN" if copy else "PASS",
          f"DCR ratio {dcr['dcr_ratio']:.2f} (about 1 = as novel as new real data; near 0 = copies)")

    final = "FAIL" if "FAIL" in res else ("WARN" if "WARN" in res else "PASS")
    print(f"\nRESULT: {final}" + ("  -> ready for Batch 3" if final != "FAIL" else "  -> please fix before sending"))
    return 1 if final == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
