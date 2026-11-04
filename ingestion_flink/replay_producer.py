import time
import json
from datetime import datetime, timezone
from kafka import KafkaProducer

from common.config import (
    KAFKA_BROKER,
    TOPIC_RAW_SENSORS,
    DATASET_NAME,
    REPLAY_HZ,
)
from common.datasets import load_or_download_dataset


def connect_kafka():
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print(f"[Replay Producer] Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except Exception as err:
            print(f"[Replay Producer] Waiting for Kafka broker at {KAFKA_BROKER} ({err})...")
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
