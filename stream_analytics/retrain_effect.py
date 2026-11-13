"""
Phase 4 evidence: does retraining the generator after a drift alert make the
synthetic data closer to the real data again?

Input: the two Kafka topics dumped to files (one JSON message per line):
  analytics.alerts   -> alerts.jsonl   (written by river_anomaly_drift.py)
  analytics.metrics  -> metrics.jsonl  (written by window_comparator.py)

For every drift alert with trigger_retrain = true it compares Pipeline 5's
metrics in the windows just BEFORE the alert with the windows just AFTER the
generator has retrained:
  before = the BEFORE metric messages up to the alert
  after  = BEFORE metric messages, starting SKIP messages after the alert
           (MMD is computed over the last 10 window pairs, so the first 10
            "after" values still contain pre-retrain windows -> skipped)
Lower MMD / Wasserstein / forecast_mae_syn after = the retrain helped.

Because these metrics also move on their own, the same before/after change is
computed at many random points WITHOUT a retrain (the "control"). Retraining
helps only if the change after retrains is clearly better than the control.

  python retrain_effect.py --alerts alerts.jsonl --metrics metrics.jsonl
Writes results/retrain_effect.csv and results/retrain_effect.png
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
METRICS = ["mmd_score", "wasserstein", "forecast_mae_syn"]


def read_jsonl(path):
    """Read one JSON message per line. Handles files saved by Windows PowerShell 5
    (its '>' writes UTF-16) as well as normal UTF-8 files."""
    raw = open(path, "rb").read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                                   # half-written last line etc.
    return pd.DataFrame(rows)


def add_pass(df, col="ts"):
    """The replay may loop (2005 -> back to 2004). Number each pass through the data,
    so an alert is matched with the metrics of the SAME pass."""
    jump_back = df[col].diff() < pd.Timedelta(days=-30)
    df["pass"] = jump_back.cumsum()
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--before", type=int, default=30, help="metric messages before / after (30 = 5 days)")
    ap.add_argument("--skip", type=int, default=10, help="metric messages skipped after the alert")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)

    al = read_jsonl(a.alerts)
    me = read_jsonl(a.metrics)
    print(f"read {len(al)} alerts and {len(me)} metric messages")
    me = me.dropna(subset=["timestamp"]).reset_index(drop=True)
    me["ts"] = pd.to_datetime(me["timestamp"], utc=True)
    me = add_pass(me)
    al["ts"] = pd.to_datetime(al["timestamp"], utc=True)
    al = add_pass(al)
    rt = al[(al["type"].str.upper() == "DRIFT") & (al["trigger_retrain"] == True)].reset_index(drop=True)
    print(f"{len(rt)} drift alerts with trigger_retrain = true")

    def change(i):
        """after-mean minus before-mean of each metric around metric position i (None if not enough data)."""
        b = me.iloc[max(0, i - a.before):i]
        f = me.iloc[i + a.skip:i + a.skip + a.before]
        if len(b) < a.before // 2 or len(f) < a.before // 2:
            return None
        if b["pass"].nunique() > 1 or f["pass"].nunique() > 1:  # do not mix two replay passes
            return None
        return {m: (b[m].mean(), f[m].mean()) for m in METRICS}

    rows, positions = [], []
    for _, r in rt.iterrows():
        same = me[(me["pass"] == r["pass"]) & (me["ts"] >= r["ts"])]
        if same.empty:
            continue
        i = same.index[0]                                    # first metric at / after the alert time
        if me.loc[i, "ts"] - r["ts"] > pd.Timedelta(days=1):
            continue      # no metrics around this alert (e.g. the generator was not running then)
        c = change(i)
        if c is None:
            continue
        positions.append(i)
        row = {"alert_time": r["timestamp"], "sensor": r.get("sensor"), "replay_pass": int(r["pass"]) + 1}
        for m, (bv, fv) in c.items():
            row[f"{m}_before"], row[f"{m}_after"] = round(bv, 4), round(fv, 4)
            row[f"{m}_change_%"] = round(100 * (fv - bv) / bv, 1) if bv else np.nan
        rows.append(row)
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("No retrain had enough metric messages before and after it. "
                         "Did Pipeline 5 run while the generator was retraining?")

    # control: the same before/after change at random positions far from any retrain
    rng = np.random.default_rng(a.seed)
    far = [i for i in range(a.before, len(me) - a.before - a.skip)
           if all(abs(i - p) > a.before + a.skip for p in positions)]
    ctrl = []
    for i in rng.choice(far, size=min(300, len(far)), replace=False) if far else []:
        c = change(int(i))
        if c:
            ctrl.append({m: 100 * (fv - bv) / bv for m, (bv, fv) in c.items() if bv})
    ctrl = pd.DataFrame(ctrl)

    res.to_csv(os.path.join(OUT, "retrain_effect.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n--- each retrain: mean before -> mean after (negative change = closer to real) ---")
    show = ["alert_time", "sensor", "replay_pass"] + [f"{m}_{s}" for m in ("mmd_score", "forecast_mae_syn")
                                                       for s in ("before", "after", "change_%")]
    print(res[show].to_string(index=False))

    print("\n--- summary ---")
    print(f"{'metric':18s} {'retrains improved':>18s} {'median change':>14s} {'control median':>15s} {'control improved':>17s}")
    for m in METRICS:
        ch = res[f"{m}_change_%"].dropna()
        cm = ctrl[m].dropna() if m in ctrl else pd.Series(dtype=float)
        print(f"{m:18s} {f'{(ch < 0).sum()}/{len(ch)}':>18s} {ch.median():>13.1f}% "
              f"{(f'{cm.median():.1f}%' if len(cm) else 'n/a'):>15s} "
              f"{(f'{(cm < 0).mean():.0%}' if len(cm) else 'n/a'):>17s}")
    print("(control = the same before/after comparison at random times without a retrain)")

    try:
        from scipy.stats import mannwhitneyu
        ch, cm = res["mmd_score_change_%"].dropna(), ctrl["mmd_score"].dropna()
        if len(ch) >= 3 and len(cm) >= 3:
            p = mannwhitneyu(ch, cm, alternative="less").pvalue
            print(f"\nMMD: is the change after retrains lower than at random times?  "
                  f"Mann-Whitney p = {p:.3f}  ({'yes, significant' if p < 0.05 else 'not significant'})")
    except ImportError:
        pass

    # plot: MMD over the stream with the retrain points
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4))
    sm = me["mmd_score"].rolling(10, min_periods=1).mean()
    ax.plot(me.index, me["mmd_score"], color="#9aa5b1", lw=0.6, label="MMD (each window)")
    ax.plot(me.index, sm, color="#2b6cb0", lw=1.6, label="MMD (rolling mean of 10)")
    for k, p in enumerate(positions):
        ax.axvline(p, color="#c53030", lw=1, alpha=0.7, label="drift alert -> retrain" if k == 0 else None)
    ax.set_xlabel("metric message (stream order)")
    ax.set_ylabel("MMD real vs synthetic (lower = closer)")
    ax.set_title("Pipeline 5: similarity of TimeVAE windows to the real stream, with generator retrains")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = os.path.join(OUT, "retrain_effect.png")
    fig.savefig(out, dpi=120)
    print(f"\nsaved {os.path.join(OUT, 'retrain_effect.csv')} and {out}")


if __name__ == "__main__":
    main()