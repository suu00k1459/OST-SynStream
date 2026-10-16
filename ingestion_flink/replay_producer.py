# ingestion_flink/replay_producer.py
import io
import json
import os
import time
import zipfile
from datetime import datetime, timezone
import pandas as pd
import requests
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from common.config import KAFKA_BROKER, TOPIC_RAW_SENSORS, REPLAY_HZ, DATASET_NAME

DATA_PATH = "/app/data/AirQualityUCI.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"

SENSOR_COLUMNS = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH"
]

def load_or_download_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"[SynStream] Downloading UCI Air Quality dataset to {DATA_PATH}...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        resp = requests.get(UCI_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open("AirQualityUCI.csv") as f_in, open(DATA_PATH, "wb") as f_out:
                f_out.write(f_in.read())

    df = pd.read_csv(DATA_PATH, sep=";", decimal=",")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    # Clean -200 sentinel values used by UCI Air Quality for missing observations
    available_cols = [c for c in SENSOR_COLUMNS if c in df.columns]
    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-200.0, pd.NA)

    df[available_cols] = df[available_cols].ffill().bfill().fillna(0.0)
    return df, available_cols

def connect_kafka() -> KafkaProducer:
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            print(f"[SynStream] Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except NoBrokersAvailable:
            print("[SynStream] Waiting for Kafka broker to start...")
            time.sleep(3)

def run_replay():
    df, cols = load_or_download_dataset()
    producer = connect_kafka()
    sleep_interval = 1.0 / max(REPLAY_HZ, 0.1)
    seq_id = 0

    print(f"[SynStream] Starting stream replay ({REPLAY_HZ} msg/sec) to '{TOPIC_RAW_SENSORS}'...")
    while True:
        for _, row in df.iterrows():
            payload = {
                "sequence_id": seq_id,
                "event_time": datetime.now(timezone.utc).isoformat(),
                "dataset": DATASET_NAME,
                "features": {col: round(float(row[col]), 4) for col in cols}
            }
            producer.send(TOPIC_RAW_SENSORS, value=payload)
            if seq_id % 10 == 0:
                print(f"Sent seq_id={seq_id} | CO(GT)={payload['features'].get('CO(GT)')}")
            seq_id += 1
            time.sleep(sleep_interval)

if __name__ == "__main__":
    run_replay()