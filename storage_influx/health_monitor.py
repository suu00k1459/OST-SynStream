"""
Stream health monitor for all Kafka topics -> InfluxDB (measurement: stream_health).
Tracks per-topic: throughput (msg/s), consumer_lag, missing_values, and late_timestamps.
"""
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from influxdb_client import Point

HEALTH_INTERVAL_S = float(os.getenv("HEALTH_INTERVAL_S", "10.0"))


def _parse_event_ts(payload: dict) -> float | None:
    """Attempts to parse an event timestamp into epoch seconds for late-arrival detection."""
    raw_ts = payload.get("timestamp") or payload.get("end_time") or payload.get("time")
    if raw_ts is None:
        return None
    if isinstance(raw_ts, (int, float)) and math.isfinite(raw_ts):
        return float(raw_ts)
    if isinstance(raw_ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
            try:
                return datetime.strptime(raw_ts[:26], fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _count_missing(payload: dict) -> int:
    """Counts None, NaN, or infinite values inside features/values or top-level payload."""
    missing = 0
    target = payload.get("features") or payload.get("values") or payload
    if isinstance(target, dict):
        for v in target.values():
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                missing += 1
    return missing


class StreamHealthMonitor:
    def __init__(self, interval_s: float = HEALTH_INTERVAL_S):
        self.interval_s = interval_s
        self.last_flush = time.monotonic()
        self.msg_counts: dict[str, int] = defaultdict(int)
        self.missing_counts: dict[str, int] = defaultdict(int)
        self.late_counts: dict[str, int] = defaultdict(int)
        self.max_event_ts: dict[str, float] = {}

    def record(self, topic: str, payload: dict) -> None:
        """Records metrics for a single incoming Kafka message."""
        self.msg_counts[topic] += 1
        self.missing_counts[topic] += _count_missing(payload)

        event_ts = _parse_event_ts(payload)
        if event_ts is not None:
            prev_max = self.max_event_ts.get(topic)
            if prev_max is not None and event_ts < prev_max:
                self.late_counts[topic] += 1
            else:
                self.max_event_ts[topic] = event_ts

    def _compute_lag_by_topic(self, consumer) -> dict[str, int]:
        """Computes total consumer lag per topic across assigned partitions."""
        lag_by_topic: dict[str, int] = defaultdict(int)
        try:
            assigned = consumer.assignment()
            if not assigned:
                return lag_by_topic
            end_offsets = consumer.end_offsets(assigned)
            for tp in assigned:
                end = end_offsets.get(tp, 0)
                pos = consumer.position(tp)
                lag_by_topic[tp.topic] += max(0, int(end - pos))
        except Exception:
            pass
        return lag_by_topic

    def flush_if_ready(self, consumer, write_api, bucket: str, org: str) -> int:
        """Writes stream_health points to InfluxDB if the interval has elapsed."""
        now_mono = time.monotonic()
        elapsed = now_mono - self.last_flush
        if elapsed < self.interval_s:
            return 0

        self.last_flush = now_mono
        lag_by_topic = self._compute_lag_by_topic(consumer)
        active_topics = set(self.msg_counts.keys()) | set(lag_by_topic.keys())
        if not active_topics:
            return 0

        ts = datetime.now(timezone.utc)
        points = []
        for topic in active_topics:
            count = self.msg_counts.get(topic, 0)
            throughput = float(count / max(elapsed, 0.001))
            p = (
                Point("stream_health")
                .tag("topic", topic)
                .field("throughput", throughput)
                .field("consumer_lag", int(lag_by_topic.get(topic, 0)))
                .field("missing_values", int(self.missing_counts.get(topic, 0)))
                .field("late_timestamps", int(self.late_counts.get(topic, 0)))
                .field("msg_count", int(count))
                .time(ts)
            )
            points.append(p)

        self.msg_counts.clear()
        self.missing_counts.clear()
        self.late_counts.clear()

        if points:
            write_api.write(bucket=bucket, org=org, record=points)
        return len(points)