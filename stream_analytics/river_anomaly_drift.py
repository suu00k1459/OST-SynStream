"""
PIPELINES 3 & 4 - Online anomaly detection + concept-drift detection (River)
===========================================================================
Every incoming record is processed ONCE, in order (true streaming):

  Pipeline 3  Two online detectors, an alert fires if either triggers:
                * per-sensor z-score (exponentially weighted) > 4  -> level shifts, spikes
                * HalfSpaceTrees score > running 99th percentile   -> unusual combinations
              The alert names the most deviating sensor and which detector fired.
  Pipeline 4  ADWIN (or KSWIN) on block means (daily for UCI, per minute for HAI) of
                (a) each monitored sensor                         -> distribution drift
                (b) the error of an online forecaster              -> behaviour drift
              A DRIFT alert carries trigger_retrain=true for Suyoung's generator.

The same StreamMonitor class runs in two modes:
  offline : reads a CSV and simulates the stream  (use this now)
  --kafka : consumes `sensors.raw`, publishes to `analytics.alerts`
            and per-record scores to `analytics.scores` (use in Phase 3)

Usage
-----
  python river_anomaly_drift.py --dataset uci --path data/AirQualityUCI.csv
  python river_anomaly_drift.py --dataset hai --path "data/hai/test1.csv" --max-rows 100000
  python river_anomaly_drift.py --dataset uci --kafka --bootstrap localhost:9092
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from river import anomaly, drift, linear_model, optim, preprocessing, stats

import data_loading as dl

OUT = os.path.join(os.path.dirname(__file__), "results")
# silence harmless numpy warnings from River on the very first records
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)

# per-dataset defaults (in records): smooth = block size for drift detection,
# warmup = no alerts before this, cooldown = min gap between drift alerts,
# memory = how far back the z-score statistics look,
# z_limit / quantile = anomaly thresholds, adwin_delta = drift sensitivity (lower = fewer alerts)
PRESETS = {
    # hourly data: daily blocks, ~1 month memory
    "uci": dict(smooth=24, warmup=24 * 14, cooldown=24 * 14, hst_window=24 * 14,
                memory=24 * 30, z_limit=4.0, quantile=0.99, adwin_delta=0.001),
    # 1 Hz data. Tuned on HAI 22.04 test1 (see README "Tuning on real HAI"):
    # the process changes operating point every ~30-60 min, so drift uses 10-min
    # blocks and at most one alert per sensor per 2 h; z_limit 6 and the stricter
    # HST quantile cut false-alarm episodes from 78 to 14 per day.
    "hai": dict(smooth=600, warmup=1800, cooldown=7200, hst_window=600,
                memory=1800, z_limit=6.0, quantile=0.999, adwin_delta=0.0002),
}


class StreamMonitor:
    def __init__(self, sensors, target=None, drift_sensors=None, quantile=0.99,
                 smooth=24, warmup=500, cooldown=500, hst_window=250, memory=1000,
                 z_limit=4.0, adwin_delta=0.001, drift_method="adwin", seed=42):
        self.sensors = sensors
        self.last_valid = {}                 # for forward-filling missing readings
        self.target = target or sensors[0]
        self.drift_sensors = drift_sensors or sensors
        self.warmup, self.cooldown = warmup, cooldown
        self.n = 0

        # ---- Pipeline 3: anomaly detection (two detectors) -----------------
        # (a) per-sensor z-score with exponentially weighted mean/variance:
        #     adapts slowly to seasons/drift, barely moves during a short attack
        self.z_limit = z_limit
        self.ew_mean = {s: stats.EWMean(fading_factor=1 / memory) for s in sensors}
        self.ew_var = {s: stats.EWVar(fading_factor=1 / memory) for s in sensors}
        # (b) multivariate HalfSpaceTrees. It needs inputs in [0, 1]; we squash the
        #     z-scores with tanh (an online MinMaxScaler would stretch its range on
        #     the first attack and hide it).
        self.hst = anomaly.HalfSpaceTrees(n_trees=25, height=6,       # height 6: tuned
                                          window_size=hst_window, seed=seed)
        self.threshold = stats.Quantile(quantile)       # dynamic threshold

        # ---- Pipeline 4: drift detection ----------------------------------
        def make_detector():
            if drift_method == "kswin":
                return drift.KSWIN(alpha=0.001, window_size=60, stat_size=20, seed=seed)
            return drift.ADWIN(delta=adwin_delta, clock=4)  # check every 4 block means

        # Drift detectors assume roughly independent inputs, so we feed them
        # NON-overlapping block means (e.g. one daily mean for hourly UCI data)
        # instead of every raw, strongly autocorrelated value.
        self.block = smooth
        self.block_means = {s: stats.Mean() for s in self.drift_sensors}
        self.drift_det = {s: make_detector() for s in self.drift_sensors}
        self.last_drift = {s: -10**9 for s in [*self.drift_sensors, "forecast_error"]}
        self.last_retrain = -10**9      # one retrain trigger per drift event

        # online forecaster: predict target(t) from all sensors at (t-1)
        # small learning rate + gradient clipping: plain SGD diverged on HAI
        self.forecaster = preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(0.005), l2=0.001, intercept_lr=0.01, clip_gradient=5.0)
        self.prev_x = None
        self.err_block = stats.Mean()
        self.err_det = make_detector()

    # ------------------------------------------------------------------
    def process(self, record: dict):
        """
        record = message from `sensors.raw`:
          {"timestamp": "...", "dataset": "...", "features": {...}, "ground_truth_anomaly": 0}
        returns (alerts: list[dict], scores: dict)
        """
        ts = record["timestamp"]
        # missing readings (null / NaN / -200 / absent) -> last valid value of that sensor
        # (causal forward-fill: in a live stream only past values are known)
        x, n_missing = {}, 0
        for s in self.sensors:
            v = record["features"].get(s)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            if v != v or v <= -199:
                n_missing += 1
                v = self.last_valid.get(s, 0.0)
            else:
                self.last_valid[s] = v
            x[s] = v
        self.n += 1
        alerts = []

        # ---- Pipeline 3 ---------------------------------------------------
        # score BEFORE learning, so a point never judges itself (prequential)
        z = {}
        for s in self.sensors:
            # max(..., 0): floating-point error can make the variance of a constant
            # sensor a tiny negative number, and sqrt would crash
            sd = math.sqrt(max(self.ew_var[s].get(), 0.0)) if self.n > 2 else 0.0
            z[s] = (x[s] - self.ew_mean[s].get()) / sd if sd > 1e-9 else 0.0
        max_z = max(abs(v) for v in z.values())
        xs = {s: 0.5 + 0.5 * math.tanh(v / 4) for s, v in z.items()}
        score = self.hst.score_one(xs)
        thr = self.threshold.get()

        ready = self.n > self.warmup
        hst_hit = ready and thr is not None and score > thr
        z_hit = ready and max_z > self.z_limit
        is_anom = hst_hit or z_hit
        if is_anom:
            culprit = max(z, key=lambda s: abs(z[s]))
            alerts.append({"timestamp": ts, "type": "ANOMALY", "sensor": culprit,
                           "score": round(score, 4), "max_abs_z": round(max_z, 2),
                           "detector": "both" if (hst_hit and z_hit) else ("hst" if hst_hit else "zscore"),
                           "trigger_retrain": False})
        # learn
        for s in self.sensors:
            self.ew_mean[s].update(x[s])
            self.ew_var[s].update(x[s])
        self.hst.learn_one(xs)
        # always update: skipping flagged points can freeze the threshold forever
        # after a regime change (every new point looks "anomalous")
        self.threshold.update(score)

        # ---- Pipeline 4a: sensor distribution drift ------------------------
        block_end = self.n % self.block == 0
        for s in self.drift_sensors:
            self.block_means[s].update(x[s])
            if block_end:
                det = self.drift_det[s]
                det.update(self.block_means[s].get())
                self.block_means[s] = stats.Mean()
                if det.drift_detected:
                    a = self._drift_alert(ts, s)
                    if a:
                        alerts.append(a)

        # ---- Pipeline 4b: forecaster-error drift ---------------------------
        abs_err = None
        if self.prev_x is not None:
            y_hat = self.forecaster.predict_one(self.prev_x)
            abs_err = abs(x[self.target] - y_hat)
            self.forecaster.learn_one(self.prev_x, x[self.target])
            self.err_block.update(abs_err)
            if block_end:
                self.err_det.update(self.err_block.get())
                self.err_block = stats.Mean()
                if self.err_det.drift_detected:
                    a = self._drift_alert(ts, "forecast_error")
                    if a:
                        alerts.append(a)
        self.prev_x = x

        scores = {"timestamp": ts, "dataset": record.get("dataset"),
                  "anomaly_score": round(score, 4),
                  "anomaly_threshold": round(thr, 4) if thr is not None else None,
                  "max_abs_z": round(max_z, 3),
                  "missing_sensors": n_missing,
                  "is_anomaly": int(is_anom),
                  "forecast_abs_err": None if abs_err is None else round(abs_err, 4)}
        return alerts, scores

    def _drift_alert(self, ts, sensor):
        # ignore drift during warm-up and suppress bursts of repeated alerts
        if self.n <= self.warmup or self.n - self.last_drift[sensor] < self.cooldown:
            return None
        self.last_drift[sensor] = self.n
        retrain = self.n - self.last_retrain >= self.cooldown
        if retrain:
            self.last_retrain = self.n
        return {"timestamp": ts, "type": "DRIFT", "sensor": sensor,
                "score": 1.0, "trigger_retrain": retrain}


# ======================================================================
# Helpers: turn a DataFrame into the same messages Faisal's producer sends
# ======================================================================
def df_to_records(df: pd.DataFrame, sensors, dataset):
    labels = df["attack"].values if "attack" in df else np.zeros(len(df), int)
    for ts, row, lab in zip(df.index, df[sensors].itertuples(index=False), labels):
        yield {"timestamp": str(ts), "dataset": dataset,
               "features": dict(zip(sensors, row)), "ground_truth_anomaly": int(lab)}


def build_monitor(dataset, sensors, drift_method):
    p = PRESETS[dataset]
    if dataset == "uci":
        target = dl.UCI_TARGET
        # watch a few key sensors only; all 12 drift together with the seasons
        drift_sensors = ["CO(GT)", "NOx(GT)", "NO2(GT)", "T"]
    else:
        target, drift_sensors = sensors[0], sensors[:5]
    return StreamMonitor(sensors, target=target, drift_sensors=drift_sensors,
                         drift_method=drift_method, **p)


# ======================================================================
# Offline mode
# ======================================================================
def run_offline(dataset, path, max_rows, drift_method, sensors_file=None):
    if dataset == "uci":
        df = dl.load_uci(path)
        sensors = dl.UCI_FEATURES
    else:
        df = dl.load_hai(path, max_rows)
        if sensors_file and os.path.exists(sensors_file):
            sensors = json.load(open(sensors_file))
        else:
            sensors = list(dl.HAI_SENSORS)
    if max_rows:
        df = df.iloc[:max_rows]
    print(f"streaming {len(df):,} records, {len(sensors)} sensors ...")

    mon = build_monitor(dataset, sensors, drift_method)
    all_alerts, all_scores = [], []
    for rec in df_to_records(df, sensors, dataset):
        alerts, scores = mon.process(rec)
        scores["ground_truth_anomaly"] = rec["ground_truth_anomaly"]
        all_alerts += alerts
        all_scores.append(scores)

    alerts = pd.DataFrame(all_alerts, columns=["timestamp", "type", "sensor", "score", "max_abs_z",
                                               "detector", "trigger_retrain"])
    scores = pd.DataFrame(all_scores)
    alerts.to_csv(os.path.join(OUT, f"p34_{dataset}_alerts.csv"), index=False)
    scores.to_csv(os.path.join(OUT, f"p34_{dataset}_scores.csv"), index=False)

    n_anom = (alerts["type"] == "ANOMALY").sum()
    drifts = alerts[alerts["type"] == "DRIFT"]
    print(f"ANOMALY alerts: {n_anom:,} ({n_anom / len(df):.2%} of records)  by detector: "
          f"{alerts.loc[alerts['type'] == 'ANOMALY', 'detector'].value_counts().to_dict()}")
    print(f"DRIFT alerts:   {len(drifts)} "
          f"({int(drifts['trigger_retrain'].sum())} with trigger_retrain=true)")
    if len(drifts):
        summary = drifts.groupby("timestamp")["sensor"].apply(lambda x: ", ".join(x))
        print(summary.to_string())
    fe = scores["forecast_abs_err"].dropna()
    print(f"online forecaster MAE ({mon.target}, original units): {fe.mean():.4f}")

    if scores["ground_truth_anomaly"].any():
        evaluate_with_labels(scores)
    plot(df, sensors, mon.target, scores, alerts, dataset)


def evaluate_with_labels(scores: pd.DataFrame):
    """Point-level precision/recall/F1 and event-level detection rate (HAI)."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    y, p, s = scores["ground_truth_anomaly"], scores["is_anomaly"], scores["anomaly_score"]
    print("\n--- evaluation against ground-truth attack labels ---")
    print(f"point-level  AUROC {roc_auc_score(y, s):.3f}  precision {precision_score(y, p, zero_division=0):.3f}"
          f"  recall {recall_score(y, p, zero_division=0):.3f}  F1 {f1_score(y, p, zero_division=0):.3f}")
    # event level: an attack counts as detected if any alert fires during it
    yv = y.values
    starts = np.flatnonzero(np.diff(np.r_[0, yv]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[yv, 0]) == -1)
    # (an alert up to 60 records after the attack ends still counts: detection delay)
    pv = p.values.astype(bool)
    hit = [pv[a:b + 61].any() for a, b in zip(starts, ends)]
    print(f"event-level  detected {sum(hit)}/{len(hit)} attacks  "
          f"[{''.join('X' if h else '.' for h in hit)}]")
    # false-alarm episodes: alerts outside attacks, merged if < 60 records apart
    fa = np.flatnonzero(pv & (yv == 0))
    episodes = 0 if len(fa) == 0 else 1 + int((np.diff(fa) > 60).sum())
    print(f"false alarms {episodes} episodes ({len(fa)} alerted records outside attacks)")


def plot(df, sensors, target, scores, alerts, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.to_datetime(scores["timestamp"])
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    show = target if target in df else sensors[0]
    ax[0].plot(t, df[show].values, lw=0.6, label=show)
    an = scores["is_anomaly"] == 1
    ax[0].scatter(t[an], df[show].values[an.values], s=6, c="red", label="ANOMALY alert", zorder=3)
    if "ground_truth_anomaly" in scores and scores["ground_truth_anomaly"].any():
        ax[0].fill_between(t, df[show].min(), df[show].max(),
                           where=scores["ground_truth_anomaly"] == 1, alpha=0.15, color="orange",
                           label="true attack")
    for i, d in enumerate(alerts[alerts["type"] == "DRIFT"].itertuples()):
        for a in ax:
            a.axvline(pd.to_datetime(d.timestamp), color="purple", ls="--", lw=0.8,
                      label="DRIFT" if (i == 0 and a is ax[0]) else None)
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(f"{dataset.upper()}: Pipeline 3 (anomaly) & Pipeline 4 (drift)")
    ax[1].plot(t, scores["anomaly_score"], lw=0.5, label="HalfSpaceTrees score")
    ax[1].plot(t, scores["anomaly_threshold"], lw=0.8, c="red", label="99% threshold")
    ax[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT, f"p34_{dataset}_plot.png")
    fig.savefig(out, dpi=110)
    print(f"plot saved to {out}")


# ======================================================================
# Kafka mode (Phase 3)
# ======================================================================
def dataset_matches(name, dataset):
    """'UCI_AirQuality', 'uci', 'HAI', 'hai-22.04' ... -> match 'uci' / 'hai'."""
    return dataset in str(name or "").lower()


def normalize_record(rec: dict) -> dict:
    """
    Accept both the planned contract and Faisal's current producer format:
      planned : {"timestamp": ..., "features": {...}, "ground_truth_anomaly": 0}
      Faisal  : {"sequence_id": ..., "event_time": ..., "features": {...}}
    """
    rec = dict(rec)
    rec.setdefault("timestamp", rec.get("event_time") or str(rec.get("sequence_id", "")))
    rec.setdefault("ground_truth_anomaly", 0)
    return rec


def run_kafka(dataset, bootstrap, drift_method, sensors, group=None, idle_exit=None):
    """
    Live Pipelines 3 + 4 for ONE dataset (run the script once per dataset).
    idle_exit: stop after this many seconds without messages (used by tests).
    """
    from kafka import KafkaConsumer, KafkaProducer   # pip install kafka-python

    group = group or f"synstream-hazem-river-{dataset}"
    consumer = KafkaConsumer(
        # one consumer group PER service: two services in one group would each get
        # only part of the partitions (part of the records)
        "sensors.raw", bootstrap_servers=bootstrap, group_id=group,
        auto_offset_reset="earliest", value_deserializer=lambda b: json.loads(b),
        consumer_timeout_ms=int(idle_exit * 1000) if idle_exit else float("inf"))
    producer = KafkaProducer(bootstrap_servers=bootstrap,
                             value_serializer=lambda v: json.dumps(v).encode())
    mon = None
    n_rec = n_alert = 0
    print(f"[river-{dataset}] listening on sensors.raw ({bootstrap}, group {group})", flush=True)
    try:
        for msg in consumer:
            rec = normalize_record(msg.value)
            if not dataset_matches(rec.get("dataset"), dataset):
                continue                                    # one monitor per dataset
            if mon is None:
                use = sensors or [k for k in rec["features"]]
                missing = [s for s in use if s not in rec["features"]]
                if missing:
                    raise KeyError(f"sensors.raw records lack sensors {missing}; check the feature list")
                mon = build_monitor(dataset, use, drift_method)
            alerts, scores = mon.process(rec)
            n_rec += 1
            producer.send("analytics.scores", scores)
            for a in alerts:
                # keep the team's dataset name ("UCI_AirQuality" / "HAI"), as Hadil's InfluxDB tags expect
                a["dataset"] = rec.get("dataset") or dataset
                producer.send("analytics.alerts", a)
                n_alert += 1
                if a["type"] == "DRIFT" or n_alert % 20 == 1:
                    print(f"[river-{dataset}] {a}", flush=True)
            if n_rec % 2000 == 0:
                print(f"[river-{dataset}] {n_rec} records, {n_alert} alerts", flush=True)
    finally:
        producer.flush()
        consumer.close()
        print(f"[river-{dataset}] stopped: {n_rec} records, {n_alert} alerts", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["uci", "hai"], required=True)
    ap.add_argument("--path", help="CSV (offline mode)")
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--drift", choices=["adwin", "kswin"], default="adwin")
    ap.add_argument("--sensors-file", default=os.path.join(OUT, "hai_sensors.json"),
                    help="HAI sensor list written by batch_tstr_benchmark.py")
    ap.add_argument("--kafka", action="store_true")
    ap.add_argument("--bootstrap", default=os.environ.get("KAFKA_BROKER", os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")))
    ap.add_argument("--group", help="Kafka consumer group (default synstream-hazem-river-<dataset>)")
    ap.add_argument("--idle-exit", type=float, help="stop after N seconds without messages")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.kafka:
        sensors = dl.UCI_FEATURES if a.dataset == "uci" else (
            json.load(open(a.sensors_file)) if os.path.exists(a.sensors_file) else None)
        run_kafka(a.dataset, a.bootstrap, a.drift, sensors, a.group, a.idle_exit)
    else:
        if not a.path:
            ap.error("--path is required in offline mode")
        run_offline(a.dataset, a.path, a.max_rows, a.drift, a.sensors_file)


if __name__ == "__main__":
    main()
