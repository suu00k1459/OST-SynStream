"""
Standalone loader for the HAI dataset -> InfluxDB to compare Original vs. LTTB in Grafana.
"""
import csv
import math
import time
from datetime import datetime, timedelta, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import os
from pathlib import Path
import gzip

from downsampler import lttb_downsample

# --- CONFIGURATION ---
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "hai" / "hai-21.03" / "train1.csv.gz"
CSV_PATH = os.getenv("HAI_CSV_PATH", str(DEFAULT_CSV))
MAX_ROWS = 10000                   # Number of rows to load for the test
LTTB_POINTS = 300                  # Target points after LTTB reduction (e.g., 97% reduction)
SENSORS = []                       # E.g. ["P1_B2004", "P1_B3004"], or empty [] to auto-pick the first 2 numeric columns

INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "synstream-secret-token"
INFLUX_ORG = "synstream_org"
INFLUX_BUCKET = "synstream_bucket"


def load_hai_columns(csv_path: str, target_sensors: list[str], max_rows: int) -> dict[str, list[float]]:
    opener = gzip.open if str(csv_path).endswith(".gz") else open
    with opener(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file has no headers.")

        first_row = next(reader)
        if not target_sensors:
            # Automatically select the first 2 numeric columns (ignoring timestamp/attack columns)
            for col, val in first_row.items():
                if "time" in col.lower() or "attack" in col.lower():
                    continue
                try:
                    if math.isfinite(float(val)):
                        target_sensors.append(col)
                        if len(target_sensors) == 2:
                            break
                except (TypeError, ValueError):
                    continue

        series = {s: [] for s in target_sensors}
        rows = [first_row, *reader]

        for i, row in enumerate(rows):
            if i >= max_rows:
                break
            for s in target_sensors:
                try:
                    val = float(row[s])
                    if math.isfinite(val):
                        series[s].append(val)
                except (TypeError, ValueError, KeyError):
                    continue

    return series


def main():
    print(f"Reading {CSV_PATH} (max {MAX_ROWS} rows)...")
    series_map = load_hai_columns(CSV_PATH, list(SENSORS), MAX_ROWS)

    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Anchor the end of the series to the current UTC time (1 point = 1 second backwards)
    end_time = datetime.now(timezone.utc)

    for sensor, values in series_map.items():
        n = len(values)
        if n == 0:
            continue

        start_time = end_time - timedelta(seconds=n)
        indexed = [(float(i), v) for i, v in enumerate(values)]

        t0 = time.perf_counter()
        reduced = lttb_downsample(indexed, LTTB_POINTS)
        dt_ms = (time.perf_counter() - t0) * 1000

        # 1. Write original points in batches of 2000
        orig_points = [
            Point("hai_original")
            .tag("sensor", sensor)
            .field("value", float(v))
            .time(start_time + timedelta(seconds=i))
            for i, v in enumerate(values)
        ]
        for k in range(0, len(orig_points), 2000):
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=orig_points[k:k + 2000])

        # 2. Write LTTB-downsampled points
        lttb_points = [
            Point("hai_lttb")
            .tag("sensor", sensor)
            .field("value", float(v))
            .time(start_time + timedelta(seconds=int(idx)))
            for idx, v in reduced
        ]
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=lttb_points)

        reduction = (1 - len(reduced) / n) * 100
        print(f"Sensor '{sensor}': {n:,} pts -> {len(reduced):,} LTTB pts ({reduction:.1f}% reduction) in {dt_ms:.1f} ms")

    client.close()
    print("Done. You can now plot 'hai_original' and 'hai_lttb' in Grafana.")


if __name__ == "__main__":
    main()