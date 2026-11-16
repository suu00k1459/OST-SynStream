"""
Handler for the window topics -> InfluxDB points (schema v5, docs/influx_schema.md).

  sensors.windowed.real       (Faisal)   -> stream_type "real",      generator "none"
  sensors.windowed.synthetic  (Suyoung)  -> stream_type "synthetic", generator from the message

Each message holds a window of 24 time steps x N sensors (data_raw and data_normalized).
Windows overlap (slide 4), so only the LAST `window_slide` rows of each window are written,
otherwise every time step would be stored about 6 times.

One point per time step and sensor:
  measurement  sensor_window
  tags         dataset, stream_type, generator, sensor
  fields       value (normalized), value_raw, window_id, event_time (end_time of the window)

Point times are ARRIVAL times. The rows of one window arrive together, so they are spread
backwards by the replay interval (1 / REPLAY_HZ) to give each row its own timestamp
(identical tags and time would overwrite each other in InfluxDB).
Limit: a synthetic window arrives a little after the real window with the same window_id,
so on a time axis the synthetic line is slightly shifted.

Use from kafka_to_influx.py:
    from handlers_windows import handle_window_message, WINDOW_TOPICS
    ...
    elif topic in WINDOW_TOPICS:
        points = handle_window_message(topic, payload)
        if points:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)

Quick check without Kafka:  python handlers_windows.py   (needs: pip install influxdb-client)
"""
import math
import os
from datetime import datetime, timedelta, timezone

from influxdb_client import Point

TOPIC_WINDOWED_REAL = os.getenv("TOPIC_WINDOWED_REAL", "sensors.windowed.real")
TOPIC_WINDOWED_SYN = os.getenv("TOPIC_WINDOWED_SYN", "sensors.windowed.synthetic")
WINDOW_TOPICS = (TOPIC_WINDOWED_REAL, TOPIC_WINDOWED_SYN)

REPLAY_HZ = float(os.getenv("REPLAY_HZ", "5.0"))
ROW_INTERVAL_S = 1.0 / max(REPLAY_HZ, 0.1)


def _number(value):
    """float(value) if it is a usable number, else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _cell(matrix, i, j):
    try:
        return _number(matrix[i][j])
    except (TypeError, IndexError):
        return None


def handle_window_message(topic: str, payload: dict) -> list:
    """Returns the list of InfluxDB points for one window message ([] if nothing to write)."""
    if topic not in WINDOW_TOPICS:
        return []
    names = payload.get("sensor_names") or []
    norm = payload.get("data_normalized")
    raw = payload.get("data_raw")
    matrix = norm if norm is not None else raw
    if not names or not matrix:
        return []

    synthetic = topic == TOPIC_WINDOWED_SYN
    stream_type = payload.get("stream_type") or ("synthetic" if synthetic else "real")
    generator = payload.get("generator") or ("unknown" if synthetic else "none")
    dataset = payload.get("dataset")
    window_id = payload.get("window_id")
    end_time = payload.get("end_time")
    try:
        slide = int(payload.get("window_slide") or 4)
    except (TypeError, ValueError):
        slide = 4

    n_rows = len(matrix)
    now = datetime.now(timezone.utc)
    points = []
    for i in range(max(n_rows - slide, 0), n_rows):
        row_time = now - timedelta(seconds=(n_rows - 1 - i) * ROW_INTERVAL_S)
        for j, sensor in enumerate(names):
            fields = {}
            value = _cell(norm, i, j)
            value_raw = _cell(raw, i, j)
            if value is not None:
                fields["value"] = value
            if value_raw is not None:
                fields["value_raw"] = value_raw
            if not fields:
                continue
            if isinstance(window_id, int) and not isinstance(window_id, bool):
                fields["window_id"] = window_id
            if end_time not in (None, ""):
                fields["event_time"] = str(end_time)
            point = Point("sensor_window").tag("stream_type", stream_type).tag("generator", str(generator))
            if dataset:
                point.tag("dataset", str(dataset))
            point.tag("sensor", str(sensor))
            for key, val in fields.items():
                point.field(key, val)
            points.append(point.time(row_time))
    return points


if __name__ == "__main__":
    names = ["CO(GT)", "T", "RH"]
    rows = [[float(r + c) for c in range(3)] for r in range(24)]
    real = {"window_id": 7, "dataset": "UCI_AirQuality", "window_slide": 4, "end_time": "2004-03-12 18:00:00",
            "sensor_names": names, "data_raw": rows, "data_normalized": rows}
    synthetic = dict(real, stream_type="synthetic", generator="TimeVAE")
    for topic, msg in [(TOPIC_WINDOWED_REAL, real), (TOPIC_WINDOWED_SYN, synthetic), ("sensors.raw", real)]:
        pts = handle_window_message(topic, msg)
        print(f"{topic}: {len(pts)} points")
        for p in pts[:2]:
            print("   ", p.to_line_protocol())
