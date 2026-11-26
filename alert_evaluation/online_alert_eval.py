"""
Stage 1 (Hadil): ONLINE alert evaluation on HAI.

Reads two Kafka topics at the same time
  sensors.raw         one record per second, with the ground truth (ground_truth_anomaly = 1 during an attack)
  analytics.alerts    Hazem's ANOMALY alerts (River detectors), each with the timestamp of the record

and keeps running metrics WHILE the stream is replayed, using the same definitions as
Hazem's offline script (stream_analytics/evaluate_live_run.py):
  * an attack event  = a run of consecutive attack records
  * an event is detected if at least one ANOMALY alert falls in [start, end + TOLERANCE) records
  * detection delay  = records between the start of the event and its first alert
  * point precision  = alerts (one per record) that fall inside an attack / all alerts

Output  InfluxDB (-> Grafana)
  measurement "alert_evaluation"  tag dataset
      fields  attack_events, attacks_detected, attacks_missed, alerts_total, alerts_during_attacks,
              alerts_unmatched (integers);  event_recall, alert_precision_point,
              median_detection_delay_records, mean_detection_delay_records (floats, only when defined)
  measurement "attack_event"      tags dataset, event_id   (one point when an event is finalised)
      fields  detected (bool), delay_records (float, only if detected), length_records, start_event_time

Run (in Docker): python alert_evaluation/online_alert_eval.py
Check without Kafka:  python alert_evaluation/online_alert_eval.py --demo
"""
import json
import os
import sys
from collections import OrderedDict, deque
from datetime import datetime, timezone
from statistics import mean, median

TOLERANCE = int(os.getenv("TOLERANCE", "60"))           # records after the end of an attack (same as Hazem)
GRACE = int(os.getenv("GRACE", "200"))                  # extra records to wait for late alerts before finalising
ALERT_TYPES = {t.strip().upper() for t in os.getenv("ALERT_TYPES", "ANOMALY").split(",") if t.strip()}
EVAL_DATASETS = [d.strip().upper() for d in os.getenv("EVAL_DATASETS", "HAI").split(",") if d.strip()]
TS_CACHE = int(os.getenv("TS_CACHE", "50000"))          # records remembered to match alerts by timestamp
PENDING_MAX_AGE = int(os.getenv("PENDING_MAX_AGE", "2000"))  # records an alert may wait for its raw record


def norm_ts(value) -> str:
    """Same text for the same moment: '2021-07-10T07:20:00.000' and '2021-07-10 07:20:00' match."""
    text = str(value).strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1]
    if "." in text:
        head, frac = text.split(".", 1)
        if frac.strip("0") == "":
            text = head
    return text


def dataset_key(name):
    """'HAI', 'hai' or 'HAI-22.04' -> 'HAI' (the entry of EVAL_DATASETS it starts with); None if not evaluated."""
    name = str(name or "").upper()
    for d in EVAL_DATASETS:
        if name.startswith(d):
            return d
    return None


def dataset_wanted(name) -> bool:
    return dataset_key(name) is not None


class DatasetState:
    """Running evaluation of ONE dataset."""

    def __init__(self, tolerance=TOLERANCE, grace=GRACE):
        self.tolerance, self.grace = tolerance, grace
        self.idx = -1                       # index of the last raw record
        self.ts_index = OrderedDict()       # norm timestamp -> (idx, ground truth)
        self.pending = deque()              # alerts whose raw record has not been seen yet: (ts, idx_when_received)
        self.events = []                    # all events: dicts (start, end, detected, delay, final, start_time)
        self.active = []                    # events not finalised yet
        self.flagged = OrderedDict()        # idx of records that got an alert (counted once per record)
        self.alerts_total = 0
        self.alerts_in_attack = 0
        self.alerts_unmatched = 0
        self.finalised = []                 # events finalised since the last drain
        self.in_attack = False

    # ---- raw records
    def add_record(self, ts, ground_truth: int):
        self.idx += 1
        i = self.idx
        self.ts_index[norm_ts(ts)] = (i, ground_truth)
        if len(self.ts_index) > TS_CACHE:
            self.ts_index.popitem(last=False)
        if ground_truth and not self.in_attack:
            ev = {"id": len(self.events), "start": i, "end": None, "detected": False,
                  "delay": None, "final": False, "start_time": str(ts)}
            self.events.append(ev)
            self.active.append(ev)
            self.in_attack = True
        elif not ground_truth and self.in_attack:
            for ev in self.active:
                if ev["end"] is None:
                    ev["end"] = i
            self.in_attack = False
        # finalise events whose window (+ grace) is over
        still = []
        for ev in self.active:
            if ev["end"] is not None and i >= ev["end"] + self.tolerance + self.grace:
                ev["final"] = True
                self.finalised.append(ev)
            else:
                still.append(ev)
        self.active = still
        self._retry_pending()

    # ---- alerts
    def add_alert(self, ts):
        hit = self.ts_index.get(norm_ts(ts))
        if hit is None:
            self.pending.append((norm_ts(ts), self.idx))
            return
        self._apply_alert(*hit)

    def _retry_pending(self):
        for _ in range(len(self.pending)):
            ts, seen_at = self.pending.popleft()
            hit = self.ts_index.get(ts)
            if hit is not None:
                self._apply_alert(*hit)
            elif self.idx - seen_at < PENDING_MAX_AGE:
                self.pending.append((ts, seen_at))
            else:
                self.alerts_unmatched += 1

    def _apply_alert(self, i, ground_truth):
        if i not in self.flagged:                       # one alert per record, as in Hazem's script
            self.flagged[i] = True
            if len(self.flagged) > TS_CACHE:
                self.flagged.popitem(last=False)
            self.alerts_total += 1
            self.alerts_in_attack += 1 if ground_truth else 0
        for ev in self.events[-20:]:                    # only recent events can still be hit
            if ev["start"] <= i and (ev["end"] is None or i < ev["end"] + self.tolerance):
                d = i - ev["start"]
                if not ev["detected"] or d < ev["delay"]:      # keep the EARLIEST alert, even if it arrives late
                    ev["detected"], ev["delay"] = True, d

    # ---- results
    def summary(self) -> dict:
        total = len(self.events)
        detected = sum(1 for e in self.events if e["detected"])
        missed = sum(1 for e in self.events if e["final"] and not e["detected"])
        out = {"attack_events": total, "attacks_detected": detected, "attacks_missed": missed,
               "alerts_total": self.alerts_total, "alerts_during_attacks": self.alerts_in_attack,
               "alerts_unmatched": self.alerts_unmatched + len(self.pending)}
        if total:
            out["event_recall"] = round(detected / total, 3)
        if self.alerts_total:
            out["alert_precision_point"] = round(self.alerts_in_attack / self.alerts_total, 3)
        delays = [e["delay"] for e in self.events if e["detected"]]
        if delays:
            out["median_detection_delay_records"] = float(median(delays))
            out["mean_detection_delay_records"] = round(float(mean(delays)), 1)
        return out

    def drain_finalised(self):
        done, self.finalised = self.finalised, []
        return done


class Evaluator:
    """Routes raw records and alerts to the state of their dataset."""

    def __init__(self):
        self.states = {}

    def state(self, dataset) -> DatasetState:
        key = dataset_key(dataset) or str(dataset)
        if key not in self.states:
            self.states[key] = DatasetState()
        return self.states[key]

    @staticmethod
    def _truth(msg) -> int:
        for key in ("ground_truth_anomaly", "attack", "label"):
            if key in msg and msg[key] is not None:
                try:
                    return 1 if float(msg[key]) > 0 else 0
                except (TypeError, ValueError):
                    return 0
        return 0

    def on_raw(self, msg: dict):
        if not dataset_wanted(msg.get("dataset")):
            return
        ts = msg.get("event_time") or msg.get("timestamp")
        if ts in (None, ""):
            return
        self.state(msg.get("dataset")).add_record(ts, self._truth(msg))

    def on_alert(self, msg: dict):
        if not dataset_wanted(msg.get("dataset")):
            return
        if str(msg.get("type", "")).upper() not in ALERT_TYPES or msg.get("timestamp") in (None, ""):
            return
        self.state(msg.get("dataset")).add_alert(msg["timestamp"])


# ----------------------------------------------------------------------------- InfluxDB points


def build_points(evaluator: Evaluator):
    """Returns the InfluxDB points for the current state (and the finalised events)."""
    from influxdb_client import Point
    now = datetime.now(timezone.utc)
    points = []
    for dataset, st in evaluator.states.items():
        fields = st.summary()
        ints = {"attack_events", "attacks_detected", "attacks_missed", "alerts_total",
                "alerts_during_attacks", "alerts_unmatched"}
        p = Point("alert_evaluation").tag("dataset", dataset)
        for k, v in fields.items():
            p.field(k, int(v) if k in ints else float(v))
        points.append(p.time(now))
        for ev in st.drain_finalised():
            q = (Point("attack_event").tag("dataset", dataset).tag("event_id", str(ev["id"]))
                 .field("detected", bool(ev["detected"]))
                 .field("length_records", int((ev["end"] if ev["end"] is not None else st.idx) - ev["start"]))
                 .field("start_event_time", ev["start_time"]))
            if ev["delay"] is not None:
                q.field("delay_records", float(ev["delay"]))
            points.append(q.time(now))
    return points


# ----------------------------------------------------------------------------- service


def run():
    import time
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS
    from kafka import KafkaConsumer

    broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    topic_raw = os.getenv("TOPIC_RAW_SENSORS", "sensors.raw")
    topic_alerts = os.getenv("TOPIC_ANALYTICS_ALERTS", "analytics.alerts")
    bucket = os.getenv("INFLUX_BUCKET", "synstream_bucket")
    org = os.getenv("INFLUX_ORG", "synstream_org")
    every = float(os.getenv("EVAL_INTERVAL_S", "5"))
    client = InfluxDBClient(url=os.getenv("INFLUX_URL", "http://influxdb:8086"),
                            token=os.getenv("INFLUX_TOKEN", "synstream-secret-token"), org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    from kafka.errors import NoBrokersAvailable
    while True:                                   # Kafka may start slowly: wait instead of crashing
        try:
            consumer = KafkaConsumer(
                topic_raw, topic_alerts, bootstrap_servers=[broker],
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest", group_id="synstream-hadil-alert-eval",
                api_version_auto_timeout_ms=30000)
            break
        except NoBrokersAvailable:
            print(f"[alert-eval] Kafka not ready at {broker}, retrying in 5 s")
            time.sleep(5)
    ev = Evaluator()
    print(f"[alert-eval] {topic_raw} + {topic_alerts}, datasets={EVAL_DATASETS}, "
          f"tolerance={TOLERANCE}, alert types={sorted(ALERT_TYPES)}")
    last_write = 0.0
    while True:
        batch = consumer.poll(timeout_ms=1000)
        for records in batch.values():
            for message in records:
                try:
                    if message.topic == topic_alerts:
                        ev.on_alert(message.value)
                    else:
                        ev.on_raw(message.value)
                except Exception as err:   # keep the service alive on a bad message
                    print(f"[alert-eval] error: {err}")
        if ev.states and time.time() - last_write >= every:
            try:
                write_api.write(bucket=bucket, org=org, record=build_points(ev))
                last_write = time.time()
            except Exception as err:
                print(f"[alert-eval] write error: {err}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        import random
        random.seed(1)
        e = Evaluator()
        attack = {(100, 140), (300, 330), (600, 650)}
        for i in range(800):
            gt = int(any(a <= i < b for a, b in attack))
            ts = f"2021-07-10 {i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}"
            e.on_raw({"dataset": "HAI", "event_time": ts, "ground_truth_anomaly": gt})
            if (gt and i in (120, 125, 305)) or i in (50, 450):
                e.on_alert({"dataset": "HAI", "type": "ANOMALY", "timestamp": ts, "sensor": "P1_FT01"})
        for k, v in e.state("HAI").summary().items():
            print(f"{k:34s} {v}")
    else:
        run()
