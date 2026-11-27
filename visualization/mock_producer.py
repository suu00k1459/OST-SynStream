"""
Mock Kafka producer for local development (Issue #3, Hadil).

Simulates the topics that the teammates' services write, with the REAL message format, so that
the stages of Hadil can be tested on a small machine WITHOUT the Flink job, the River services
or the HAI data file.

  sensors.raw            Faisal's replay producer: event_time, dataset, features, ground_truth_anomaly
  sensors.features.real  Faisal's Flink job: one message per window with window_features per sensor
  analytics.alerts       Hazem's River detectors: ANOMALY alerts with the timestamp of the record

The stream is HAI-like (dataset "HAI"): 3000 records, 1 per second of data time, 3 attacks.
Attack 1 is detected quickly, attack 2 is detected twice, attack 3 is missed, and 3 alerts are
false alarms. Expected result of alert-eval at the end:
    attack_events 3, attacks_detected 2, attacks_missed 1, event_recall 0.667, alerts_total 6,
    alert_precision_point 0.5

These are SIMULATED data (not the real HAI file): they show that the pipeline and the dashboards
work, not how good the detectors are on real data.

Usage
  python mock_producer.py --check            check the stream without Kafka (no install needed;
                                             online_alert_eval.py must be in the same folder or in ../alert_evaluation)
  python mock_producer.py                    send to Kafka (KAFKA_BROKER, default kafka:9092), 50 records/s
  python mock_producer.py --hz 20 --loop     slower, repeated

In Docker, with the stack running (kafka, influxdb, alert-eval, feature-summaries), from the repo root:
  docker compose run --rm --no-deps -v "%cd%\\visualization:/mock" alert-eval python /mock/mock_producer.py
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta

DATASET = os.getenv("MOCK_DATASET", "HAI")
# The 15 HAI sensors agreed by the team (stream_analytics/results/hai_sensors.json).
SENSORS = ["P1_FT01", "P1_FT02", "P1_FT03", "P1_LIT01", "P1_PIT01", "P1_PIT02", "P1_TIT02", "P1_TIT03",
           "P2_VIBTR01", "P2_VIBTR02", "P2_VIBTR03", "P2_VIBTR04", "P3_FIT01", "P3_LIT01", "P3_PIT01"]
# Simulated levels (NOT real HAI values): any value works, the mock only checks the pipeline.
BASE = {s: 10.0 + 5.0 * k for k, s in enumerate(SENSORS)}
SCALE = {s: 0.5 + 0.1 * k for k, s in enumerate(SENSORS)}
ATTACKS = [(400, 440), (1000, 1050), (1800, 1830)]          # [start, end) in records
ALERT_AT = {200, 410, 1005, 1030, 1400, 2400}              # records that get an alert: 3 inside attacks, 3 false alarms
N_RECORDS = 3000
WINDOW, SLIDE = 24, 4
START = datetime(2021, 7, 10, 0, 0, 0)

TOPIC_RAW = os.getenv("TOPIC_RAW_SENSORS", "sensors.raw")
TOPIC_FEATURES = os.getenv("TOPIC_FEATURES_REAL", "sensors.features.real")
TOPIC_ALERTS = os.getenv("TOPIC_ANALYTICS_ALERTS", "analytics.alerts")


def in_attack(i):
    return any(a <= i < b for a, b in ATTACKS)


def ts(i):
    return (START + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")


def stream(seed=1):
    """Yields (topic, message) in the order a real system would produce them."""
    rnd = random.Random(seed)
    history = {s: [] for s in SENSORS}
    window_id = 0
    for i in range(N_RECORDS):
        attack = in_attack(i)
        values = {}
        for s in SENSORS:
            shift = 4 * SCALE[s] if attack else 0.0
            values[s] = BASE[s] + shift + rnd.gauss(0, SCALE[s])
            history[s].append(values[s])
        yield TOPIC_RAW, {"sequence_id": i, "event_time": ts(i), "dataset": DATASET,
                          "features": values, "ground_truth_anomaly": int(attack)}
        if i in ALERT_AT:
            yield TOPIC_ALERTS, {"timestamp": ts(i), "type": "ANOMALY", "sensor": "P1_FT01",
                                 "score": 0.9, "max_abs_z": 4.2, "detector": "both",
                                 "trigger_retrain": False, "dataset": DATASET}
        if (i + 1) >= WINDOW and (i + 1 - WINDOW) % SLIDE == 0:
            feats = {}
            for s in SENSORS:
                w = history[s][-WINDOW:]
                m = sum(w) / len(w)
                sd = (sum((x - m) ** 2 for x in w) / len(w)) ** 0.5
                feats[s] = {"mean": m, "std": sd, "min": min(w), "max": max(w), "slope": 0.0,
                            "mean_abs_change": 0.0, "spectral_energy": 0.0, "lag1_autocorr": 0.0}
            yield TOPIC_FEATURES, {"window_id": window_id, "dataset": DATASET,
                                   "end_time": ts(i), "window_features": feats}
            window_id += 1


def check():
    """Runs the stream through Hadil's evaluator and prints the result (no Kafka needed)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.join(here, "..", "alert_evaluation"), "alert_evaluation", "/app/alert_evaluation"):
        if os.path.isfile(os.path.join(p, "online_alert_eval.py")):
            sys.path.insert(0, p)
            break
    import online_alert_eval as ae
    ev = ae.Evaluator()
    counts = {}
    for topic, msg in stream():
        counts[topic] = counts.get(topic, 0) + 1
        if topic == TOPIC_RAW:
            ev.on_raw(msg)
        elif topic == TOPIC_ALERTS:
            ev.on_alert(msg)
    print("messages per topic:", counts)
    for k, v in ev.state(DATASET).summary().items():
        print(f"{k:34s} {v}")


def send(hz, loop):
    from kafka import KafkaProducer
    broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    producer = KafkaProducer(bootstrap_servers=[broker], api_version_auto_timeout_ms=30000,
                             value_serializer=lambda m: json.dumps(m).encode("utf-8"))
    print(f"[mock] sending {N_RECORDS} records at {hz}/s to {broker} (dataset {DATASET})")
    while True:
        for topic, msg in stream():
            producer.send(topic, msg)
            if topic == TOPIC_RAW:
                time.sleep(1.0 / hz)
        producer.flush()
        print("[mock] done")
        if not loop:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="no Kafka: run the stream through the evaluator")
    ap.add_argument("--hz", type=float, default=50.0, help="raw records per second")
    ap.add_argument("--loop", action="store_true", help="repeat the stream")
    a = ap.parse_args()
    check() if a.check else send(a.hz, a.loop)
