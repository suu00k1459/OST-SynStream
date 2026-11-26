"""
Stage 2 (Hadil): streaming feature summaries with a t-digest.

Input   Kafka topic sensors.features.real (Faisal's Flink job): one message per window,
        {"window_id", "dataset", "end_time", "window_features": {sensor: {"mean", "std", ...}}}
Output  InfluxDB measurement "feature_bands" -> Grafana
          tags    dataset, sensor, feature
          fields  value (current), n, p05, p25, p50, p75, p95, outside_band, window_id, event_time

For every (dataset, sensor, feature) a t-digest keeps a small, fixed-size summary of ALL the values
seen so far, so percentile bands (p05..p95) are available at any time without storing the stream.
Each new value is compared with the band computed from the PAST values only, then added to the digest.
The first MIN_COUNT values only build the digest (no band yet).

Reference: Dunning, T., & Ertl, O. (2019). Computing extremely accurate quantiles using t-digests.
arXiv:1902.04023.

Run (in Docker): python stream_summaries/feature_summaries.py
Quick check without Kafka: python stream_summaries/feature_summaries.py --demo   (needs influxdb-client)
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

# ----------------------------------------------------------------------------- t-digest


class TDigest:
    """Merging t-digest (Dunning & Ertl, 2019) with the k1 scale function (asin)."""

    def __init__(self, compression: float = 100.0):
        self.delta = float(compression)
        self.centroids = []          # sorted list of [mean, weight]
        self.buffer = []             # values waiting to be merged
        self.count = 0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, x: float):
        if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
            return
        self.buffer.append((float(x), 1.0))
        self.count += 1
        self.minimum = min(self.minimum, x)
        self.maximum = max(self.maximum, x)
        if len(self.buffer) >= 5 * self.delta:
            self._compress()

    def _k(self, q: float) -> float:
        return self.delta / (2 * math.pi) * math.asin(2 * min(max(q, 0.0), 1.0) - 1)

    def _q(self, k: float) -> float:
        k = min(k, self.delta / 4)
        return (math.sin(k * 2 * math.pi / self.delta) + 1) / 2

    def _compress(self):
        items = sorted([tuple(c) for c in self.centroids] + self.buffer)
        self.buffer = []
        if not items:
            return
        total = sum(w for _, w in items)
        merged = []
        cur_mean, cur_w = items[0]
        done = 0.0                                   # weight of the centroids already closed
        q_limit = self._q(self._k(0.0) + 1)
        for mean, w in items[1:]:
            if (done + cur_w + w) / total <= q_limit:
                new_w = cur_w + w                    # merge into the current centroid
                cur_mean += (mean - cur_mean) * w / new_w
                cur_w = new_w
            else:
                merged.append([cur_mean, cur_w])
                done += cur_w
                q_limit = self._q(self._k(done / total) + 1)
                cur_mean, cur_w = mean, w
        merged.append([cur_mean, cur_w])
        self.centroids = merged

    def quantile(self, q: float):
        if self.buffer:
            self._compress()
        if not self.centroids:
            return None
        total = sum(w for _, w in self.centroids)
        target = min(max(q, 0.0), 1.0) * total
        if len(self.centroids) == 1:
            return self.centroids[0][0]
        centers, cum = [], 0.0
        for mean, w in self.centroids:
            centers.append(cum + w / 2)
            cum += w
        if target <= centers[0]:                     # left tail: between the minimum and the first centroid
            span = centers[0] or 1.0
            return self.minimum + (self.centroids[0][0] - self.minimum) * target / span
        if target >= centers[-1]:                    # right tail
            span = (total - centers[-1]) or 1.0
            return self.centroids[-1][0] + (self.maximum - self.centroids[-1][0]) * (target - centers[-1]) / span
        for i in range(len(centers) - 1):
            if target <= centers[i + 1]:
                frac = (target - centers[i]) / (centers[i + 1] - centers[i])
                return self.centroids[i][0] + (self.centroids[i + 1][0] - self.centroids[i][0]) * frac
        return self.centroids[-1][0]


# ----------------------------------------------------------------------------- summarizer

BANDS = {"p05": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}
FEATURES = [f.strip() for f in os.getenv("FEATURES", "mean,std").split(",") if f.strip()]
MIN_COUNT = int(os.getenv("MIN_COUNT", "30"))
COMPRESSION = float(os.getenv("COMPRESSION", "100"))


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


class Summarizer:
    def __init__(self, features=None, min_count=MIN_COUNT, compression=COMPRESSION):
        self.features = list(features or FEATURES)
        self.min_count = min_count
        self.compression = compression
        self.digests = {}

    def process(self, msg: dict) -> list:
        """Returns the InfluxDB points for one sensors.features.real message."""
        from influxdb_client import Point
        dataset = msg.get("dataset")
        window_id = msg.get("window_id")
        end_time = msg.get("end_time")
        now = datetime.now(timezone.utc)
        points = []
        for sensor, feats in (msg.get("window_features") or {}).items():
            for feature in self.features:
                value = _num((feats or {}).get(feature))
                if value is None:
                    continue
                digest = self.digests.setdefault((dataset, sensor, feature), TDigest(self.compression))
                fields = {"value": value, "n": digest.count}
                if digest.count >= self.min_count:
                    bands = {name: digest.quantile(q) for name, q in BANDS.items()}
                    fields.update(bands)
                    fields["outside_band"] = bool(value < bands["p05"] or value > bands["p95"])
                digest.add(value)
                if isinstance(window_id, int) and not isinstance(window_id, bool):
                    fields["window_id"] = window_id
                if end_time not in (None, ""):
                    fields["event_time"] = str(end_time)
                point = Point("feature_bands").tag("sensor", str(sensor)).tag("feature", feature)
                if dataset:
                    point.tag("dataset", str(dataset))
                for key, val in fields.items():
                    point.field(key, val)
                points.append(point.time(now))
        return points


# ----------------------------------------------------------------------------- service


def run():
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS
    from kafka import KafkaConsumer

    broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    topic = os.getenv("TOPIC_FEATURES_REAL", "sensors.features.real")
    bucket = os.getenv("INFLUX_BUCKET", "synstream_bucket")
    org = os.getenv("INFLUX_ORG", "synstream_org")
    client = InfluxDBClient(url=os.getenv("INFLUX_URL", "http://influxdb:8086"),
                            token=os.getenv("INFLUX_TOKEN", "synstream-secret-token"), org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    from kafka.errors import NoBrokersAvailable
    import time
    while True:                                   # Kafka may start slowly: wait instead of crashing
        try:
            consumer = KafkaConsumer(
                topic, bootstrap_servers=[broker],
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest", group_id="synstream-hadil-feature-summaries",
                api_version_auto_timeout_ms=30000)
            break
        except NoBrokersAvailable:
            print(f"[feature-summaries] Kafka not ready at {broker}, retrying in 5 s")
            time.sleep(5)
    summarizer = Summarizer()
    print(f"[feature-summaries] listening to {topic}, features={summarizer.features}, min_count={summarizer.min_count}")
    for message in consumer:
        try:
            points = summarizer.process(message.value)
            if points:
                write_api.write(bucket=bucket, org=org, record=points)
        except Exception as err:  # keep the service alive on a bad message
            print(f"[feature-summaries] error: {err}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        import random
        s = Summarizer(min_count=5)
        for i in range(8):
            m = {"window_id": i, "dataset": "UCI_AirQuality", "end_time": f"2004-03-10 {i:02d}:00:00",
                 "window_features": {"CO(GT)": {"mean": 2 + random.random(), "std": 0.3}}}
            for p in s.process(m)[:1]:
                print(p.to_line_protocol())
    else:
        run()
