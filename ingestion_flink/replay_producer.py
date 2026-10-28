import os
import time
import json
import zipfile
import requests
import pandas as pd
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from common.config import (
    KAFKA_BROKER,
    TOPIC_RAW_SENSORS,
    DATASET_NAME,
    REPLAY_HZ,
)

DATA_DIR = "/app/data"
UCI_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
UCI_CSV_PATH = os.path.join(DATA_DIR, "AirQualityUCI.csv")

SENSOR_COLS = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH"
]


def load_or_download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(UCI_CSV_PATH):
        print(f"[Replay Producer] Downloading UCI Air Quality dataset from {UCI_URL}...")
        zip_path = os.path.join(DATA_DIR, "air_quality.zip")
        resp = requests.get(UCI_URL, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        print("[Replay Producer] Download and extraction complete.")

    # Read European CSV format (semicolon separator, comma decimal)
    df = pd.read_csv(UCI_CSV_PATH, sep=";", decimal=",")
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # Parse actual historical recording timestamp from Date (DD/MM/YYYY) and Time (HH.MM.SS)
    if "Date" in df.columns and "Time" in df.columns:
        raw_dt_str = (
            df["Date"].astype(str).str.strip()
            + " "
            + df["Time"].astype(str).str.strip().str.replace(".", ":", regex=False)
        )
        # Convert to ISO-8601 format with UTC 'Z' suffix (e.g., "2004-03-10T18:00:00Z")
        parsed_dt = pd.to_datetime(raw_dt_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        df["event_time"] = parsed_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        # Drop any trailing empty rows where Date/Time was missing
        df = df.dropna(subset=["event_time"]).reset_index(drop=True)
    else:
        # Fallback if Date/Time columns are missing
        df["event_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure all 13 sensor columns are numeric and forward-fill -200 sentinels
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-200.0, pd.NA)

    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().bfill().fillna(0.0)
    print(
        f"[Replay Producer] Loaded {len(df)} rows | "
        f"Range: {df['event_time'].iloc[0]} -> {df['event_time'].iloc[-1]}"
    )
    return df, SENSOR_COLS


def connect_kafka():
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print(f"[Replay Producer] Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except NoBrokersAvailable:
            print(f"[Replay Producer] Waiting for Kafka broker at {KAFKA_BROKER}...")
            time.sleep(2)


def run_replay():
    df, cols = load_or_download_dataset()
    producer = connect_kafka()
    sleep_interval = 1.0 / max(REPLAY_HZ, 0.1)

    seq_id = 0
    while True:
        for _, row in df.iterrows():
            # Use the actual historical recording timestamp as event_time
            actual_recording_time = str(row["event_time"])

            payload = {
                "sequence_id": seq_id,
                "event_time": actual_recording_time,
                "ingest_time": datetime.now(timezone.utc).isoformat(),  # Optional: keeps wall-clock time for latency checks
                "dataset": DATASET_NAME,
                "features": {col: float(row[col]) for col in cols},
            }
            producer.send(TOPIC_RAW_SENSORS, value=payload)

            if seq_id % 10 == 0:
                print(
                    f"Sent seq_id={seq_id} | "
                    f"event_time={actual_recording_time} | "
                    f"CO(GT)={payload['features']['CO(GT)']}"
                )

            seq_id += 1
            time.sleep(sleep_interval)


if __name__ == "__main__":
    run_replay()