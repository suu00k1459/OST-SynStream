import os
import json
from datetime import datetime, timezone
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from handlers_analytics import handle_message as handle_analytics, HANDLERS as ANALYTICS_HANDLERS
from handlers_windows import handle_window_message, WINDOW_TOPICS
from downsampler import StreamDownsampler
from health_monitor import StreamHealthMonitor

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "synstream-secret-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "synstream_org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "synstream_bucket")

TOPIC_RAW_SENSORS = os.getenv("TOPIC_RAW_SENSORS", "sensors.raw")
TOPIC_WINDOWED_REAL = os.getenv("TOPIC_WINDOWED_REAL", "sensors.windowed.real")
TOPIC_FEATURES_REAL = os.getenv("TOPIC_FEATURES_REAL", "sensors.features.real")

TOPICS = [
    TOPIC_RAW_SENSORS,
    *WINDOW_TOPICS,              
    *ANALYTICS_HANDLERS.keys(),  
    # TOPIC_FEATURES_REAL,
]

def handle_raw_sensors(payload: dict) -> Point:
    point = Point("sensor_readings").tag("data_source", "real")

    for tag_key in ["dataset", "sensor_id"]:
        if tag_key in payload:
            point.tag(tag_key, str(payload.pop(tag_key)))

    ts = payload.pop("timestamp", payload.pop("time", None))
    point.time(ts if ts else datetime.now(timezone.utc))

    features_dict = payload.pop("features", payload.pop("values", {}))
    if isinstance(features_dict, dict):
        for k, v in features_dict.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                point.field(k, float(v))

    for k, v in payload.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            point.field(k, float(v))

    return point

def process_message(topic: str, payload: dict, write_api):
    if topic == TOPIC_RAW_SENSORS:
        point = handle_raw_sensors(payload)
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

    elif topic in WINDOW_TOPICS:
        points = handle_window_message(topic, payload)
        if points:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)

    elif topic in ANALYTICS_HANDLERS:
        point = handle_analytics(topic, payload)
        if point is not None:
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

    elif topic == TOPIC_FEATURES_REAL:
        pass

def main():
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="influx-sink-group"
    )

    downsampler = StreamDownsampler()
    health_monitor = StreamHealthMonitor()

    print(f"InfluxDB Sink running. Listening to topics: {TOPICS}")
    for msg in consumer:
        try:
            payload = msg.value
            health_monitor.record(msg.topic, payload)
            if msg.topic == TOPIC_RAW_SENSORS:
                downsampler.add(payload)
            process_message(msg.topic, payload, write_api)
            health_monitor.flush_if_ready(consumer, write_api, INFLUX_BUCKET, INFLUX_ORG)
            downsampler.flush_if_ready(write_api, INFLUX_BUCKET, INFLUX_ORG)
        except Exception as e:
            print(f"Error processing message from {msg.topic}: {e}")

if __name__ == "__main__":
    main()