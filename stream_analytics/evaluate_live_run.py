"""
Evaluate one live SynStream run from Kafka dumps (Pipelines 3, 4 and 5).

Inputs (JSON lines, one Kafka message per line):
  --alerts   dump of analytics.alerts
  --metrics  dump of analytics.metrics
Optional:
  --labels   HAI test file with the attack column (e.g. data/hai-22.04/test1.csv)
             -> event-level evaluation of the live ANOMALY alerts
  --dataset  HAI or UCI_AirQuality (default HAI)
  --warmup   number of first Pipeline 5 windows to skip in the averages (default 50)
  --out      output prefix (default stream_analytics/results/live_<dataset>)

Writes <out>_summary.csv and <out>_metrics.png and prints a summary.

Example:
  python stream_analytics/evaluate_live_run.py --alerts alerts_hai.jsonl --metrics metrics_hai.jsonl --labels data/hai-22.04/test1.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def read_jsonl(path, dataset):
    rows = []
    with open(path, "rb") as fb:                  # PowerShell ">" writes UTF-16
        enc = "utf-16" if fb.read(2) in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    with open(path, encoding=enc) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue                      # skips "Processed a total of N messages" etc.
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "dataset" in df:
        df = df[df["dataset"].astype(str).str.upper().str.startswith(dataset.upper()[:3])]
    return df.reset_index(drop=True)


def evaluate_alerts(alerts, labels_path, tolerance=60):
    out = {}
    anom = alerts[alerts["type"] == "ANOMALY"] if not alerts.empty else alerts
    drift = alerts[alerts["type"] == "DRIFT"] if not alerts.empty else alerts
    out["anomaly_alerts"] = len(anom)
    out["drift_alerts"] = len(drift)
    if not drift.empty:
        out["drift_times"] = "; ".join(f"{t} ({s})" for t, s in zip(drift["timestamp"], drift["sensor"]))
        if "trigger_retrain" in drift:
            out["drift_retrain_requests"] = int(drift["trigger_retrain"].fillna(False).astype(bool).sum())
    if not labels_path or anom.empty:
        return out

    lab = pd.read_csv(labels_path)
    cols = {c.lower(): c for c in lab.columns}
    tcol, acol = cols.get("timestamp") or cols.get("time"), cols.get("attack")
    if tcol is None or acol is None:
        print(f"[warn] {labels_path}: no timestamp/attack column, skipping label evaluation")
        return out
    t = pd.to_datetime(lab[tcol])
    y = (lab[acol].values > 0).astype(int)
    pos = pd.Series(np.arange(len(t)), index=t)
    idx = pos.reindex(pd.to_datetime(anom["timestamp"])).dropna().astype(int).values
    if len(idx) < len(anom):
        print(f"[warn] {len(anom) - len(idx)} alert timestamps not found in the label file")
    flag = np.zeros(len(y), int)
    flag[idx] = 1

    # attack events = runs of consecutive attack records
    edges = np.diff(np.r_[0, y, 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]   # [start, end)
    hits, delays = [], []
    for s, e in zip(starts, ends):
        w = np.where(flag[s:min(e + tolerance, len(y))])[0]
        hits.append(len(w) > 0)
        if len(w):
            delays.append(int(w[0]))
    in_attack = int(flag[y == 1].sum())
    out.update({
        "attack_events": len(starts),
        "attacks_detected": int(sum(hits)),
        "event_recall": round(sum(hits) / max(len(starts), 1), 3),
        "alerts_during_attacks": in_attack,
        "alert_precision_point": round(in_attack / max(flag.sum(), 1), 3),
        "median_detection_delay_records": (int(np.median(delays)) if delays else None),
    })
    return out


def summarize_metrics(m, warmup):
    out = {}
    if m.empty:
        return out, m
    m = m.sort_values("window_id").reset_index(drop=True)
    n_msgs = len(m)
    first, last = int(m["window_id"].min()), int(m["window_id"].max())
    out["metric_messages"] = n_msgs
    out["unique_windows_paired"] = int(m["window_id"].nunique())
    out["window_id_range"] = f"{first}-{last}"
    out["pairing_rate"] = round(m["window_id"].nunique() / (last - first + 1), 3)
    out["missing_timestamps"] = int(m["timestamp"].isna().sum()) if "timestamp" in m else None
    s = m[m["window_id"] >= first + warmup]
    for col in ["mmd_score", "wasserstein", "acf_diff", "xcorr_diff",
                "forecast_mae_real", "forecast_mae_syn", "dcr", "dcr_ratio"]:
        if col in s:
            out[f"{col}_mean"] = round(float(s[col].mean()), 4)
            out[f"{col}_median"] = round(float(s[col].median()), 4)
    if {"forecast_mae_real", "forecast_mae_syn"} <= set(s.columns):
        out["tstr_over_trtr"] = round(float(s["forecast_mae_syn"].mean() / s["forecast_mae_real"].mean()), 3)
        out["share_windows_tstr_le_trtr"] = round(float((s["forecast_mae_syn"] <= s["forecast_mae_real"]).mean()), 3)
    return out, m


def plot(m, path, dataset, warmup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = m[m["window_id"] >= m["window_id"].min() + warmup]
    roll = max(len(s) // 200, 1)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(s["window_id"], s["mmd_score"].rolling(roll).mean(), color="#4c72b0")
    ax[0].set_ylabel("MMD (smoothed)")
    ax[1].plot(s["window_id"], s["forecast_mae_real"].rolling(roll).mean(), label="TRTR (real-trained)", color="#55a868")
    ax[1].plot(s["window_id"], s["forecast_mae_syn"].rolling(roll).mean(), label="TSTR (synthetic-trained)", color="#dd8452")
    ax[1].set_ylabel("forecast MAE")
    ax[1].legend(frameon=False)
    ax[2].plot(s["window_id"], s["dcr_ratio"].rolling(roll).mean(), color="#8172b2")
    ax[2].axhline(1.0, color="grey", lw=0.8, ls="--")
    ax[2].set_ylabel("DCR ratio")
    ax[2].set_xlabel("window_id")
    fig.suptitle(f"Live Pipeline 5 on {dataset} (rolling mean over {roll} windows, warm-up {warmup} skipped)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--dataset", default="HAI")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    out = a.out or os.path.join("stream_analytics", "results", f"live_{a.dataset.lower()}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    alerts = read_jsonl(a.alerts, a.dataset)
    metrics = read_jsonl(a.metrics, a.dataset)
    summary = {"dataset": a.dataset}
    summary.update(evaluate_alerts(alerts, a.labels) if not alerts.empty else {"anomaly_alerts": 0, "drift_alerts": 0})
    msum, m = summarize_metrics(metrics, a.warmup)
    summary.update(msum)

    print(f"\n=== Live run evaluation: {a.dataset} ===")
    for k, v in summary.items():
        print(f"{k:32s} {v}")
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(f"{out}_summary.csv")
    if not m.empty:
        plot(m, f"{out}_metrics.png", a.dataset, a.warmup)
    print(f"\nwrote {out}_summary.csv" + (f" and {out}_metrics.png" if not m.empty else ""))


if __name__ == "__main__":
    main()