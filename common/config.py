import os

# Kafka Broker
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")

# Kafka Topics (Shared Contract across all 5 Team Members)
TOPIC_RAW_SENSORS = os.getenv("TOPIC_RAW_SENSORS", "sensors.raw")
TOPIC_WINDOWED_REAL = os.getenv("TOPIC_WINDOWED_REAL", "sensors.windowed.real")
TOPIC_FEATURES_REAL = os.getenv("TOPIC_FEATURES_REAL", "sensors.features.real")
TOPIC_WINDOWED_SYN = os.getenv("TOPIC_WINDOWED_SYN", "sensors.windowed.synthetic")
TOPIC_ANALYTICS_ALERTS = os.getenv("TOPIC_ANALYTICS_ALERTS", "analytics.alerts")
TOPIC_ANALYTICS_METRICS = os.getenv("TOPIC_ANALYTICS_METRICS", "analytics.metrics")

# Kafka Consumer Group IDs
GROUP_FLINK_WINDOW = "synstream-flink-window-group"        # Faisal (Pipeline 2)
GROUP_SYNTHETIC_GEN = "synstream-suyoung-generator-group"  # Suyoung (Pipeline 5 Generator)
GROUP_RIVER_ANALYTICS = "synstream-hazem-river-group"      # Hazem (Pipelines 3, 4, 5)
GROUP_INFLUX_SINK = "synstream-hadil-influx-sink-group"    # Hadil (InfluxDB/Grafana Sink)

# Replay & Windowing Parameters
DATASET_NAME = os.getenv("DATASET_NAME", "UCI_AirQuality")  # "UCI_AirQuality" or "HAI"
REPLAY_HZ = float(os.getenv("REPLAY_HZ", "5.0"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "24"))
WINDOW_SLIDE = int(os.getenv("WINDOW_SLIDE", "4"))

# Dataset Registry (NMHC(GT) removed after EDA due to 90.2% missing values)
DATASET_CONFIGS = {
    "UCI_AirQuality": {
        "sensors": [
            "CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)",
            "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
            "PT08.S5(O3)", "T", "RH", "AH"
        ],
        "label_col": None,
    },
    "HAI": {
        "sensors": [
            "P1_B2004", "P1_B2016", "P1_B3004", "P1_B3005", "P1_B4002",
            "P1_FT01", "P1_FT02", "P1_PIT01", "P1_TIT01", "P2_SIT01"
        ],
        "label_col": "Attack",
    },
}

ACTIVE_SENSORS = DATASET_CONFIGS[DATASET_NAME]["sensors"]