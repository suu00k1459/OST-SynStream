"""
Handlers for Hazem's analytics topics -> InfluxDB points (schema v5, docs/influx_schema.md).

  analytics.scores   -> measurement "anomaly_scores"     (one point per sensor record)
  analytics.alerts   -> measurement "alerts"             (ANOMALY and DRIFT alerts)
  analytics.metrics  -> measurement "similarity_metrics" (real vs synthetic windows)

Rules:
  * The point time is the ARRIVAL time. Hazem's "timestamp" is the original dataset
    time (2004-2005, and the replay loops), so it is kept as a text field "event_time".
  * Numeric fields are always written as float (ints for window_id and pairs_in_metric)
    because InfluxDB refuses to change the type of a field after the first write.
  * None, NaN and infinite values are skipped instead of crashing the sink.

Use from kafka_to_influx.py:
    from handlers_analytics import handle_message
    point = handle_message(topic, payload)
    if point is not None:
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

Quick check without Kafka:  python handlers_analytics.py   (needs: pip install influxdb-client)
"""
import math
import os
from datetime import datetime, timezone

from influxdb_client import Point

TOPIC_ANALYTICS_SCORES = os.getenv("TOPIC_ANALYTICS_SCORES", "analytics.scores")
TOPIC_ANALYTICS_ALERTS = os.getenv("TOPIC_ANALYTICS_ALERTS", "analytics.alerts")
TOPIC_ANALYTICS_METRICS = os.getenv("TOPIC_ANALYTICS_METRICS", "analytics.metrics")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numbers(payload: dict, names, cast) -> dict:
    """Keeps only the names that hold a usable number, converted with cast (float or int)."""
    return {n: cast(payload[n]) for n in names if _is_number(payload.get(n))}


def _build(measurement: str, tags: dict, fields: dict):
    """Creates the point, or returns None when there is nothing to write."""
    if not fields:
        return None
    point = Point(measurement)
    for key, value in tags.items():
        if value is not None and str(value) != "":
            point.tag(key, str(value))
    for key, value in fields.items():
        point.field(key, value)
    return point.time(datetime.now(timezone.utc))


def _event_time(payload: dict) -> dict:
    ts = payload.get("timestamp")
    return {"event_time": str(ts)} if ts not in (None, "") else {}


def handle_scores(payload: dict):
    """analytics.scores: one message per sensor record (anomaly score over time)."""
    fields = _numbers(payload, ["anomaly_score", "anomaly_threshold", "max_abs_z",
                                "missing_sensors", "forecast_abs_err"], float)
    if payload.get("is_anomaly") is not None:
        fields["is_anomaly"] = bool(payload["is_anomaly"])
    fields.update(_event_time(payload))
    tags = {"dataset": payload.get("dataset")}
    return _build("anomaly_scores", tags, fields)


def handle_alerts(payload: dict):
    """analytics.alerts: ANOMALY alerts (with detector and max_abs_z) and DRIFT alerts."""
    fields = _numbers(payload, ["score", "max_abs_z"], float)
    if payload.get("trigger_retrain") is not None:
        fields["trigger_retrain"] = bool(payload["trigger_retrain"])
    fields.update(_event_time(payload))
    tags = {"dataset": payload.get("dataset"), "type": payload.get("type"),
            "sensor": payload.get("sensor"), "detector": payload.get("detector")}
    return _build("alerts", tags, fields)


def handle_metrics(payload: dict):
    """analytics.metrics: similarity, forecasting error and privacy metrics per generator."""
    fields = _numbers(payload, ["mmd_score", "wasserstein", "acf_diff", "xcorr_diff",
                                "forecast_mae_real", "forecast_mae_syn", "dcr", "dcr_ratio"], float)
    fields.update(_numbers(payload, ["pairs_in_metric", "window_id"], int))
    fields.update(_event_time(payload))
    tags = {"dataset": payload.get("dataset"), "generator": payload.get("generator")}
    return _build("similarity_metrics", tags, fields)


HANDLERS = {
    TOPIC_ANALYTICS_SCORES: handle_scores,
    TOPIC_ANALYTICS_ALERTS: handle_alerts,
    TOPIC_ANALYTICS_METRICS: handle_metrics,
}


def handle_message(topic: str, payload: dict):
    """Returns the InfluxDB point for a message of one of the analytics topics, or None."""
    handler = HANDLERS.get(topic)
    return handler(payload) if handler else None


if __name__ == "__main__":
    samples = [
        (TOPIC_ANALYTICS_SCORES, {"timestamp": "2004-03-12 18:00:00", "dataset": "UCI_AirQuality",
                                  "anomaly_score": 0.4123, "anomaly_threshold": None, "max_abs_z": 1.2,
                                  "missing_sensors": 0, "is_anomaly": 0, "forecast_abs_err": None}),
        (TOPIC_ANALYTICS_ALERTS, {"timestamp": "2004-03-20 04:00:00", "type": "ANOMALY", "sensor": "CO(GT)",
                                  "score": 0.9712, "max_abs_z": 5.31, "detector": "both",
                                  "trigger_retrain": False, "dataset": "UCI_AirQuality"}),
        (TOPIC_ANALYTICS_ALERTS, {"timestamp": "2004-05-02 00:00:00", "type": "DRIFT", "sensor": "T",
                                  "score": 1.0, "trigger_retrain": True, "dataset": "UCI_AirQuality"}),
        (TOPIC_ANALYTICS_METRICS, {"window_id": 104, "generator": "TimeVAE", "pairs_in_metric": 10,
                                   "mmd_score": 0.00712, "wasserstein": 0.12, "acf_diff": 0.084,
                                   "xcorr_diff": 0.033, "forecast_mae_real": 0.381, "forecast_mae_syn": 0.405,
                                   "dataset": "UCI_AirQuality", "timestamp": "2004-12-21 07:00:00"}),
        ("sensors.raw", {"dataset": "UCI_AirQuality"}),   # not an analytics topic -> None
    ]
    for topic, payload in samples:
        point = handle_message(topic, payload)
        print(f"{topic:20s} ->", point.to_line_protocol() if point is not None else None)
