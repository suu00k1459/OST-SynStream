"""
PIPELINE 5 - Continuous real vs synthetic window comparison
==========================================================
Inputs (Kafka topics in Phase 3, simulated offline now):
  sensors.windowed.real       {"window_id": 104, "data": [[...24 rows x D...]], ...}   (Faisal)
  sensors.windowed.synthetic  {"window_id": 104, "generator": "TimeGAN", "data": [...]} (Suyoung)

For every (real, synthetic) pair with the same window_id it publishes to
`analytics.metrics`:
  statistical : mmd_score, wasserstein, acf_diff, xcorr_diff
                (computed over the last K pairs, one 24-step window alone is too noisy)
  downstream  : forecast_mae_real / forecast_mae_syn  - online TRTR vs TSTR:
                two River regressors predict the next target value of each NEW
                real window; one learns only from real windows, the other only
                from that generator's synthetic windows. If synthetic data is
                useful, forecast_mae_syn stays close to forecast_mae_real.
  privacy     : dcr, dcr_ratio (if training windows are provided)

Usage
-----
  # offline demo on UCI (placeholder generators until Suyoung's service exists)
  python window_comparator.py --uci data/AirQualityUCI.csv
  # live
  python window_comparator.py --kafka --bootstrap localhost:9092
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict, deque

import numpy as np
import pandas as pd
from river import linear_model, optim, preprocessing

import data_loading as dl
from metrics import DCR, compare_windows

OUT = os.path.join(os.path.dirname(__file__), "results")


N_LAGS = 6   # target lags used by the online forecaster


def _xy(window: np.ndarray, target_idx: int):
    """
    window (L, D) -> compact features for 1-step-ahead forecasting of the target:
    last N_LAGS target values + every sensor at the previous step + window mean.
    y = target at the last step.
    """
    past = window[:-1]
    feats = {f"lag{k}": float(past[-k, target_idx]) for k in range(1, N_LAGS + 1)}
    feats.update({f"s{d}": float(v) for d, v in enumerate(past[-1])})
    feats["mean_target"] = float(past[:, target_idx].mean())
    return feats, float(window[-1, target_idx])


class RollingMAE:
    """MAE over the last n predictions (works on every River version)."""

    def __init__(self, n):
        self.errs = deque(maxlen=n)

    def update(self, y, y_hat):
        self.errs.append(abs(y - y_hat))

    def get(self):
        return float(np.mean(self.errs)) if self.errs else 0.0


def _new_forecaster():
    # small learning rate + gradient clipping keep online SGD stable
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.005), l2=0.001, clip_gradient=5.0)


class WindowComparator:
    def __init__(self, target_idx: int = 0, k: int = 10, mae_window: int = 50,
                 train_windows=None, max_buffer: int = 1000):
        self.t = target_idx
        self.k = k
        self.mae_window = mae_window
        self.max_buffer = max_buffer
        self.real = OrderedDict()            # window_id -> (L, D) array
        self.pending_syn = OrderedDict()     # window_id -> [(generator, array)]
        self.recent = {}                     # generator -> deque of (real, syn) pairs
        self.f_real = _new_forecaster()      # TRTR model
        self.f_syn = {}                      # generator -> TSTR model
        self.mae_real = RollingMAE(mae_window)
        self.mae_syn = {}
        self.real_pred = {}                  # window_id -> prediction of the TRTR model
        self.dcr = None
        if train_windows is not None:
            tw = np.asarray(train_windows)
            # last 10% of training windows act as "fresh real data" baseline
            cut = int(len(tw) * 0.9)
            self.dcr = DCR(tw[:cut], tw[cut:])

    # ------------------------------------------------------------------
    def add_real(self, msg: dict) -> list[dict]:
        wid = msg["window_id"]
        w = np.asarray(msg["data"], float)
        # TRTR: predict this real window BEFORE learning from it (prequential)
        X, y = _xy(w, self.t)
        self.real_pred[wid] = (self.f_real.predict_one(X), y)
        self.mae_real.update(y, self.real_pred[wid][0])
        self.f_real.learn_one(X, y)
        self.real[wid] = w
        self._trim()
        out = []
        for gen, sw in self.pending_syn.pop(wid, []):   # synthetic arrived first
            out.append(self._pair(wid, w, gen, sw))
        return out

    def add_synthetic(self, msg: dict) -> list[dict]:
        wid, gen = msg["window_id"], msg.get("generator", "unknown")
        sw = np.asarray(msg["data"], float)
        if wid in self.real:
            return [self._pair(wid, self.real[wid], gen, sw)]
        self.pending_syn.setdefault(wid, []).append((gen, sw))
        self._trim()
        return []

    # ------------------------------------------------------------------
    def _pair(self, wid, real_w, gen, syn_w) -> dict:
        if real_w.shape != syn_w.shape:
            raise ValueError(f"window {wid}: real {real_w.shape} vs synthetic {syn_w.shape}")
        dq = self.recent.setdefault(gen, deque(maxlen=self.k))
        dq.append((real_w, syn_w))
        R = np.stack([p[0] for p in dq])
        S = np.stack([p[1] for p in dq])
        m = compare_windows(R, S)

        # TSTR: the generator's model predicts the REAL window, then learns from SYNTHETIC
        if gen not in self.f_syn:
            self.f_syn[gen] = _new_forecaster()
            self.mae_syn[gen] = RollingMAE(self.mae_window)
        X_real, y_real = _xy(real_w, self.t)
        self.mae_syn[gen].update(y_real, self.f_syn[gen].predict_one(X_real))
        X_syn, y_syn = _xy(syn_w, self.t)
        self.f_syn[gen].learn_one(X_syn, y_syn)

        out = {"window_id": wid, "generator": gen, "pairs_in_metric": len(dq),
               **{k: round(v, 5) for k, v in m.items()},
               "forecast_mae_real": round(self.mae_real.get(), 5),
               "forecast_mae_syn": round(self.mae_syn[gen].get(), 5)}
        if self.dcr is not None:
            out.update({k: round(v, 5) for k, v in self.dcr.score(syn_w).items()})
        return out

    def _trim(self):
        for d in (self.real, self.pending_syn):
            while len(d) > self.max_buffer:
                d.popitem(last=False)
        while len(self.real_pred) > self.max_buffer:
            self.real_pred.pop(next(iter(self.real_pred)))


# ======================================================================
# Offline demo (UCI)
# ======================================================================
def run_offline(path, stride, train_windows_path):
    df = dl.load_uci(path)
    train_df, test_df = dl.chrono_split(df, 0.7)
    scaler = dl.ZScaler().fit(train_df.values)
    train_w = dl.make_windows(scaler.transform(train_df.values))
    stream_w = dl.make_windows(scaler.transform(test_df.values), stride=stride)
    starts = test_df.index[np.arange(0, len(test_df) - dl.WINDOW_LEN + 1, stride)]
    if train_windows_path and os.path.exists(train_windows_path):
        train_w = np.load(train_windows_path)
    print(f"streaming {len(stream_w)} real windows (stride {stride}) through Pipeline 5 ...")

    # Two PLACEHOLDER generators until Suyoung's service is live:
    #  "placeholder-tracking": follows the live data closely (real + noise) -> what a
    #                          good, drift-adapted generator should look like
    #  "placeholder-stale":    replays random TRAINING windows -> a generator that never
    #                          adapts; its similarity should get worse as the stream drifts
    rng = np.random.default_rng(0)
    comp = WindowComparator(target_idx=dl.UCI_FEATURES.index(dl.UCI_TARGET), train_windows=train_w)
    rows = []
    for wid, (w, ts) in enumerate(zip(stream_w, starts)):
        comp.add_real({"window_id": wid, "data": w.tolist()})
        gens = {"placeholder-tracking": w + rng.normal(0, 0.15, w.shape),
                "placeholder-stale": train_w[rng.integers(len(train_w))]}
        for g, sw in gens.items():
            for r in comp.add_synthetic({"window_id": wid, "generator": g, "data": sw.tolist()}):
                rows.append({"start_ts": ts, **r})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "p5_uci_metrics.csv"), index=False)
    print(res.groupby("generator")[["mmd_score", "wasserstein", "acf_diff", "xcorr_diff",
                                    "forecast_mae_real", "forecast_mae_syn", "dcr_ratio"]]
          .mean().round(4).T.to_string())
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["mmd_score", "wasserstein", "acf_diff", "forecast_mae_syn"]
    fig, ax = plt.subplots(len(cols), 1, figsize=(13, 9), sharex=True)
    for g, d in res.groupby("generator"):
        t = pd.to_datetime(d["start_ts"])
        for a, c in zip(ax, cols):
            a.plot(t, d[c].rolling(10, min_periods=1).mean(), lw=0.9, label=g)
    ax[-1].plot(pd.to_datetime(res["start_ts"]),
                res["forecast_mae_real"].rolling(10, min_periods=1).mean(), "k--", lw=0.9,
                label="forecast_mae_real (TRTR)")
    for a, c in zip(ax, cols):
        a.set_ylabel(c, fontsize=8)
    ax[0].set_title("Pipeline 5: real vs synthetic similarity and utility over time (lower = better)")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(OUT, "p5_uci_plot.png")
    fig.savefig(out, dpi=110)
    print(f"plot saved to {out}")


# ======================================================================
# Kafka mode (Phase 3)
# ======================================================================
def _first_existing(*paths):
    return next((p for p in paths if p and os.path.exists(p)), None)


def load_scaler(dataset: str):
    """
    Fixed training scaler. Looked up in this order:
      $UCI_SCALER / $HAI_SCALER   (in Docker: Faisal's official data/splits/uci_scaler.npz)
      results/<ds>_scaler.npz     (written by batch_tstr_benchmark.py)
    Accepts our key 'features' and Faisal's key 'sensor_names'.
    """
    ds = "hai" if "hai" in str(dataset).lower() else "uci"
    path = _first_existing(os.environ.get(f"{ds.upper()}_SCALER"), os.path.join(OUT, f"{ds}_scaler.npz"))
    if path is None:
        return None
    z = np.load(path, allow_pickle=True)
    return {"mean": z["mean"], "std": z["std"],
            "features": [str(f) for f in z["features" if "features" in z.files else "sensor_names"]]}


def load_train_windows(ds: str):
    """Real training windows for the DCR privacy check: $UCI_TRAIN_WINDOWS (a .npy, or Faisal's
    split .npz -> its X_train) or results/<ds>_train_windows.npy. None if not available."""
    path = _first_existing(os.environ.get(f"{ds.upper()}_TRAIN_WINDOWS"),
                           os.path.join(OUT, f"{ds}_train_windows.npy"))
    if path is None:
        return None
    if path.endswith(".npz"):
        return np.load(path)["X_train"]
    return np.load(path)


_SCALERS = {}


def normalize_window(msg: dict) -> dict:
    """
    Make every window look like {"window_id", "dataset", "data": (L, D) in FIXED z-scale}.

    Accepts:
      * planned contract / Suyoung: "data" already in the fixed scale -> unchanged
      * Faisal's Flink output: "data_raw" + "sensor_names" -> we select the agreed
        features in the agreed order (drops NMHC(GT)) and apply the fixed scaler.
        His "data_normalized" uses a running scaler that keeps changing, so we
        do not use it (it would not match the scale Suyoung trained on).
    """
    if "data" in msg:
        return msg
    msg = dict(msg)
    ds = msg.get("dataset", "uci")
    if ds not in _SCALERS:
        _SCALERS[ds] = load_scaler(ds)
    sc = _SCALERS[ds]
    raw = np.asarray(msg["data_raw"], float)
    names = msg.get("sensor_names")
    if sc is None:
        msg["data"] = msg.get("data_normalized", raw)
        return msg
    if names:
        idx = [names.index(f) for f in sc["features"]]
        raw = raw[:, idx]
    msg["data"] = ((raw - sc["mean"]) / sc["std"]).tolist()
    return msg


def _dataset_key(name) -> str:
    return "hai" if "hai" in str(name or "").lower() else "uci"


def run_kafka(bootstrap, group="synstream-hazem-pipeline5", idle_exit=None):
    """
    Live Pipeline 5. One WindowComparator PER DATASET (UCI windows have 12
    features, HAI windows 15, so they must never be compared with each other).
    idle_exit: stop after this many seconds without messages (used by tests).
    """
    from kafka import KafkaConsumer, KafkaProducer   # pip install kafka-python

    consumer = KafkaConsumer(
        "sensors.windowed.real", "sensors.windowed.synthetic",
        bootstrap_servers=bootstrap, group_id=group,
        auto_offset_reset="earliest", value_deserializer=lambda b: json.loads(b),
        consumer_timeout_ms=int(idle_exit * 1000) if idle_exit else float("inf"))
    producer = KafkaProducer(bootstrap_servers=bootstrap,
                             value_serializer=lambda v: json.dumps(v).encode())
    comps = {}
    window_ts = OrderedDict()            # (dataset, window_id) -> end timestamp of the REAL window
    print(f"[pipeline5] listening on sensors.windowed.real + .synthetic ({bootstrap}, group {group})", flush=True)
    n_out = 0
    try:
        for msg in consumer:
            ds = _dataset_key(msg.value.get("dataset"))
            if ds not in comps:
                tw = load_train_windows(ds)
                target = dl.UCI_FEATURES.index(dl.UCI_TARGET) if ds == "uci" else 0
                comps[ds] = WindowComparator(target_idx=target, train_windows=tw)
                print(f"[pipeline5] new dataset '{ds}' (DCR baseline: {'yes' if tw is not None else 'no'})", flush=True)
            value = normalize_window(msg.value)
            if msg.topic == "sensors.windowed.real":
                window_ts[(ds, value["window_id"])] = value.get("end_ts") or value.get("end_time")
                while len(window_ts) > 5000:
                    window_ts.popitem(last=False)
                results = comps[ds].add_real(value)
            else:
                results = comps[ds].add_synthetic(value)
            for r in results:
                r["dataset"] = msg.value.get("dataset") or ds      # team name, e.g. "UCI_AirQuality"
                # data time of the real window (the synthetic message has no timestamp)
                r["timestamp"] = window_ts.get((ds, r["window_id"]))
                producer.send("analytics.metrics", r)
                n_out += 1
                if n_out % 50 == 1:
                    print(f"[pipeline5] {n_out} metric messages, latest: {r}", flush=True)
    finally:
        producer.flush()
        consumer.close()
        print(f"[pipeline5] stopped after {n_out} metric messages", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uci", help="offline demo: path to AirQualityUCI.csv")
    ap.add_argument("--stride", type=int, default=4, help="window slide (Flink uses 4)")
    ap.add_argument("--train-windows", default=os.path.join(OUT, "uci_train_windows.npy"),
                    help="offline demo: real training windows for the DCR privacy metric")
    ap.add_argument("--kafka", action="store_true")
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BROKER", os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")))
    ap.add_argument("--group", default="synstream-hazem-pipeline5", help="Kafka consumer group")
    ap.add_argument("--idle-exit", type=float, help="stop after N seconds without messages")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.kafka:
        run_kafka(a.bootstrap, a.group, a.idle_exit)
    elif a.uci:
        run_offline(a.uci, a.stride, a.train_windows)
    else:
        ap.error("give --uci (offline demo) or --kafka")


if __name__ == "__main__":
    main()
