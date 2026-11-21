import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_generator_service import (
    load_generator, build_message, check_sensors, connect, retrain_vae,
    KAFKA_BROKER, TOPIC_WINDOWED_REAL, TOPIC_WINDOWED_SYN,
    TOPIC_ANALYTICS_ALERTS, GROUP_SYNTHETIC_GEN,
)

GENERATOR = os.getenv("GENERATOR", "TimeVAE")
WAIT_WINDOWS = int(os.getenv("RETRAIN_WAIT_WINDOWS", "60"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "15"))



def run_live_v2(generator):
    from kafka import KafkaConsumer, KafkaProducer

    generate_fn, ckpt, model = load_generator(generator)
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    std = np.asarray(ckpt["scaler_std"], dtype=np.float32)
    clip = ckpt["config"]["clip"]
    post_drift = None

    consumer = connect(lambda: KafkaConsumer(
        TOPIC_WINDOWED_REAL, TOPIC_ANALYTICS_ALERTS,
        bootstrap_servers=KAFKA_BROKER,
        group_id=f"{GROUP_SYNTHETIC_GEN}-{generator.lower()}-v2",
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    ), "consumer")
    producer = connect(lambda: KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    ), "producer")
    print(f"{generator} v2 running: retrain after {WAIT_WINDOWS} post-drift windows, "
          f"{RETRAIN_EPOCHS} epochs")

    
    for record in consumer:
        if record.topic == TOPIC_ANALYTICS_ALERTS:
            alert = record.value
            if (alert.get("type") == "DRIFT" and alert.get("trigger_retrain")
                    and generator == "TimeVAE"):
                if post_drift is None:
                    post_drift = []
                    print(f"Drift alert on {alert.get('sensor')} -> "
                          f"collecting {WAIT_WINDOWS} new windows before retraining")
                else:
                    print(f"Drift alert on {alert.get('sensor')} ignored (already collecting)")
            continue


        
        real = record.value
        check_sensors(real, ckpt)
        if post_drift is not None:
            z = (np.asarray(real["data_raw"], dtype=np.float32) - mean) / std
            post_drift.append(np.clip(z, -clip, clip))
            if len(post_drift) >= WAIT_WINDOWS:
                retrain_vae(model, post_drift, ckpt["config"], epochs=RETRAIN_EPOCHS)
                post_drift = None

        msg = build_message(real, generate_fn(1)[0], ckpt, generator)
        producer.send(TOPIC_WINDOWED_SYN, key=msg["dataset"].encode("utf-8"), value=msg)
        print(f"{datetime.now(timezone.utc):%H:%M:%S} real window {real['window_id']} "
              f"-> synthetic {generator}")


if __name__ == "__main__":
    run_live_v2(GENERATOR)