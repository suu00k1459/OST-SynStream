# common/config.py
import os

# Kafka Settings
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_RAW_SENSORS = os.getenv("TOPIC_RAW_SENSORS", "sensors.raw")
TOPIC_WINDOWED_REAL = os.getenv("TOPIC_WINDOWED_REAL", "sensors.windowed.real")
TOPIC_WINDOWED_SYN = os.getenv("TOPIC_WINDOWED_SYN", "sensors.windowed.synthetic")

# Replay Settings (Pipeline 1)
DATASET_NAME = os.getenv("DATASET_NAME", "UCI_AirQuality")
REPLAY_HZ = float(os.getenv("REPLAY_HZ", "2.0"))  # Events per second (e.g., 2.0 = 1 row every 0.5s)
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "24"))
#testing
