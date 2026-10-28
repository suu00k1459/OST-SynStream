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
    ACTIVE_SENSORS,
)

DATA_DIR = "/app/data"
UCI_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
UCI_CSV_PATH = os.path.join(DATA_DIR, "AirQualityUCI.csv")


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
        parsed_dt = pd.to_datetime(raw_dt_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        df["event_time"] = parsed_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df.dropna(subset=["event_time"]).reset_index(drop=True)
    else:
        df["event_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Convert the 12 selected sensor columns to float (keep -200.0 for PyFlink causal imputation)
    for col in ACTIVE_SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-200.0)

    print(
        f"[Replay Producer] Loaded {len(df)} rows with {len(ACTIVE_SENSORS)} clean features | "
        f"Range: {df['event_time'].iloc[0]} -> {df['event_time'].iloc[-1]}"
    )
    return df, ACTIVE_SENSORS


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
            actual_recording_time = str(row["event_time"])

            payload = {
                "sequence_id": seq_id,
                "event_time": actual_recording_time,
                "ingest_time": datetime.now(timezone.utc).isoformat(),
                "dataset": DATASET_NAME,
                "features": {col: float(row[col]) for col in cols},
            }
            producer.send(
                TOPIC_RAW_SENSORS,
                key=DATASET_NAME.encode("utf-8"),
                value=payload,
            )

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