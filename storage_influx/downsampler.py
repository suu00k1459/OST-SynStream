"""
Stream downsampling for sensors.raw -> InfluxDB.
Writes:
  - sensor_downsampled_1m: per-minute mean, min, max, and count per (dataset, sensor).
  - sensor_lttb: visually downsampled points using Steinarsson (2013) LTTB.
"""
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from influxdb_client import Point

DOWNSAMPLE_INTERVAL_S = float(os.getenv("DOWNSAMPLE_INTERVAL_S", "60.0"))
LTTB_THRESHOLD = int(os.getenv("LTTB_THRESHOLD", "15"))


def _is_valid_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def lttb_downsample(data: list[tuple[float, float]], threshold: int) -> list[tuple[float, float]]:
    """Largest-Triangle-Three-Buckets (Steinarsson 2013) for (x, y) series."""
    n = len(data)
    if threshold >= n or threshold <= 2:
        return list(data)

    sampled = [data[0]]
    bucket_size = (n - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        avg_range_start = int(math.floor((i + 1) * bucket_size)) + 1
        avg_range_end = min(int(math.floor((i + 2) * bucket_size)) + 1, n)
        avg_len = max(avg_range_end - avg_range_start, 1)
        avg_x = sum(data[j][0] for j in range(avg_range_start, avg_range_end)) / avg_len
        avg_y = sum(data[j][1] for j in range(avg_range_start, avg_range_end)) / avg_len

        range_offs = int(math.floor(i * bucket_size)) + 1
        range_to = int(math.floor((i + 1) * bucket_size)) + 1

        point_ax, point_ay = data[a]
        max_area = -1.0
        next_a = range_offs

        for j in range(range_offs, min(range_to, n)):
            area = abs(
                (point_ax - avg_x) * (data[j][1] - point_ay)
                - (point_ax - data[j][0]) * (avg_y - point_ay)
            ) * 0.5
            if area > max_area:
                max_area = area
                next_a = j

        sampled.append(data[next_a])
        a = next_a

    sampled.append(data[-1])
    return sampled


class StreamDownsampler:
    def __init__(self, interval_s: float = DOWNSAMPLE_INTERVAL_S, lttb_threshold: int = LTTB_THRESHOLD):
        self.interval_s = interval_s
        self.lttb_threshold = lttb_threshold
        self.last_flush = time.monotonic()
        # key: (dataset, sensor_name) -> list of (epoch_seconds, float_value)
        self.buffer: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    def add(self, payload: dict) -> None:
        """Extracts numeric sensor readings with arrival timestamp from a sensors.raw payload."""
        dataset = str(payload.get("dataset", "unknown"))
        features = payload.get("features") or payload.get("values")

        if isinstance(features, dict):
            source = features
        else:
            source = {k: v for k, v in payload.items() if k not in ("sequence_id", "window_id")}

        now_epoch = time.time()
        for sensor, val in source.items():
            if _is_valid_number(val):
                self.buffer[(dataset, str(sensor))].append((now_epoch, float(val)))

    def flush_if_ready(self, write_api, bucket: str, org: str) -> int:
        """Writes per-minute mean/min/max and LTTB-reduced points if the interval has elapsed."""
        now_mono = time.monotonic()
        if (now_mono - self.last_flush) < self.interval_s:
            return 0

        self.last_flush = now_mono
        if not self.buffer:
            return 0

        ts = datetime.now(timezone.utc)
        points = []
        for (dataset, sensor), series in self.buffer.items():
            if not series:
                continue

            # 1. Per-minute aggregates (mean / min / max / count)
            values = [v for _, v in series]
            p_agg = (
                Point("sensor_downsampled_1m")
                .tag("dataset", dataset)
                .tag("sensor", sensor)
                .field("mean", float(sum(values) / len(values)))
                .field("min", float(min(values)))
                .field("max", float(max(values)))
                .field("count", int(len(values)))
                .time(ts)
            )
            points.append(p_agg)

            # 2. Visual downsampling with LTTB
            reduced = lttb_downsample(series, self.lttb_threshold)
            for pt_ts, pt_val in reduced:
                p_lttb = (
                    Point("sensor_lttb")
                    .tag("dataset", dataset)
                    .tag("sensor", sensor)
                    .field("value", float(pt_val))
                    .time(datetime.fromtimestamp(pt_ts, tz=timezone.utc))
                )
                points.append(p_lttb)

        self.buffer.clear()
        if points:
            write_api.write(bucket=bucket, org=org, record=points)
        return len(points)