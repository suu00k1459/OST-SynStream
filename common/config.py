import os

# Kafka Settings
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_RAW_SENSORS = os.getenv("TOPIC_RAW_SENSORS", "sensors.raw")
TOPIC_WINDOWED_REAL = os.getenv("TOPIC_WINDOWED_REAL", "sensors.windowed.real")
TOPIC_FEATURES_REAL = os.getenv("TOPIC_FEATURES_REAL", "sensors.features.real")
TOPIC_WINDOWED_SYN = os.getenv("TOPIC_WINDOWED_SYN", "sensors.windowed.synthetic")

# Replay & Windowing Settings (Pipelines 1 & 2)
DATASET_NAME = os.getenv("DATASET_NAME", "UCI_AirQuality")
REPLAY_HZ = float(os.getenv("REPLAY_HZ", "5.0"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "24"))
WINDOW_SLIDE = int(os.getenv("WINDOW_SLIDE", "4"))

GROUP_FLINK_WINDOW = "synstream-flink-window-group"       # Faisal (Pipeline 2)
GROUP_SYNTHETIC_GEN = "synstream-suyoung-generator-group" # Suyoung (Online Generator)
GROUP_RIVER_ANALYTICS = "synstream-hazem-river-group"     # Hazem (Pipelines 3, 4, 5)
GROUP_INFLUX_SINK = "synstream-hadil-influx-sink-group"   # Hadil (Visualization Sink)