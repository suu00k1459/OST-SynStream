import json
import time
from collections import deque
import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from pyflink.common import WatermarkStrategy, Types
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.datastream.functions import KeyedProcessFunction, RuntimeContext

from common.config import (
    KAFKA_BROKER,
    TOPIC_RAW_SENSORS,
    TOPIC_WINDOWED_REAL,
    TOPIC_FEATURES_REAL,
    WINDOW_SIZE,
    WINDOW_SLIDE,
)

SENSOR_ORDER = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH"
]


class SlidingWindowFeatureExtractor(KeyedProcessFunction):
    """
    Stateful PyFlink operator that:
    1. Performs causal streaming imputation (replaces -200 sentinels or NaNs with last valid reading).
    2. Maintains online Welford running mean/variance per sensor for real-time Z-score scaling.
    3. Clips normalized Z-scores to [-4.0, +4.0] to prevent exploding gradients in TimeGAN/TimeVAE.
    4. Buffers a sliding window of size WINDOW_SIZE (24) and emits every WINDOW_SLIDE (4) events.
    5. Publishes normalized 2D window tensors to `sensors.windowed.real` (for Suyoung's generator).
    6. Publishes statistical, temporal, and frequency-domain (FFT) features to `sensors.features.real` (for Hazem).
    """

    def open(self, runtime_context: RuntimeContext):
        self.raw_buffer = deque(maxlen=WINDOW_SIZE)
        self.norm_buffer = deque(maxlen=WINDOW_SIZE)
        self.time_buffer = deque(maxlen=WINDOW_SIZE)
        self.event_count = 0
        self.window_id = 0

        # UPGRADE 1: Track last valid reading per sensor for true streaming causal forward-fill
        self.last_valid = np.zeros(len(SENSOR_ORDER), dtype=np.float64)

        # Online Welford statistics for Z-score normalization
        self.n = 0
        self.mean = np.zeros(len(SENSOR_ORDER), dtype=np.float64)
        self.m2 = np.zeros(len(SENSOR_ORDER), dtype=np.float64)

        # Initialize Kafka Producer with retry loop
        self.producer = None
        while self.producer is None:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=[KAFKA_BROKER],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
            except NoBrokersAvailable:
                time.sleep(2)

    def _update_and_normalize(self, x: np.ndarray) -> np.ndarray:
        """Updates running mean/variance using Welford's online algorithm and returns Z-scores."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

        if self.n < 2:
            return np.zeros_like(x)
        std = np.sqrt(self.m2 / (self.n - 1))
        std = np.where(std < 1e-6, 1.0, std)
        return (x - self.mean) / std

    def process_element(self, value: str, ctx: KeyedProcessFunction.Context):
        record = json.loads(value)
        feats = record.get("features", {})
        dataset = record.get("dataset", "UCI_AirQuality")
        event_time = record.get("event_time", "")

        # UPGRADE 2: Causal Streaming Imputation (replaces -200 sentinel or NaN with last valid reading)
        raw_vals = []
        for idx, col in enumerate(SENSOR_ORDER):
            val = float(feats.get(col, -200.0))
            if np.isnan(val) or val <= -199.0:  # Catch UCI -200 sentinel or NaN in real time!
                val = self.last_valid[idx]      # Causal forward-fill from Flink state
            else:
                self.last_valid[idx] = val      # Update state with valid reading
            raw_vals.append(val)
        x = np.array(raw_vals, dtype=np.float64)

        # UPGRADE 3: Online Z-score normalization + +/- 4.0 sigma clipping for GAN stability
        x_norm = self._update_and_normalize(x)
        x_norm_clipped = np.clip(x_norm, -4.0, 4.0)

        # Append unclipped raw values (for anomaly detection) and clipped normalized values (for GANs)
        self.raw_buffer.append(x)
        self.norm_buffer.append(x_norm_clipped)
        self.time_buffer.append(event_time)
        self.event_count += 1

        # Trigger sliding window emission once buffer reaches WINDOW_SIZE and every WINDOW_SLIDE steps
        if len(self.raw_buffer) == WINDOW_SIZE and (self.event_count % WINDOW_SLIDE == 0):
            raw_mat = np.array(self.raw_buffer)    # shape: (WINDOW_SIZE, 13)
            norm_mat = np.array(self.norm_buffer)  # shape: (WINDOW_SIZE, 13)

            # 1. Payload for `sensors.windowed.real` (Suyoung & Hazem Pipeline 5)
            window_payload = {
                "window_id": self.window_id,
                "dataset": dataset,
                "start_time": self.time_buffer[0],
                "end_time": self.time_buffer[-1],
                "window_size": WINDOW_SIZE,
                "window_slide": WINDOW_SLIDE,
                "sensor_names": SENSOR_ORDER,
                "data_raw": np.round(raw_mat, 4).tolist(),
                "data_normalized": np.round(norm_mat, 4).tolist(),
                "scaler_stats": {
                    "mean": np.round(self.mean, 4).tolist(),
                    "std": np.round(np.sqrt(np.maximum(self.m2 / max(self.n - 1, 1), 1e-6)), 4).tolist(),
                },
            }
            self.producer.send(TOPIC_WINDOWED_REAL, value=window_payload)

            # 2. Extract temporal, statistical & spectral features per sensor for `sensors.features.real`
            t_idx = np.arange(WINDOW_SIZE, dtype=np.float64)
            t_centered = t_idx - t_idx.mean()
            denom = np.sum(t_centered ** 2)

            feature_summary = {}
            for idx, col in enumerate(SENSOR_ORDER):
                series = raw_mat[:, idx]
                slope = float(np.sum(t_centered * (series - series.mean())) / denom)
                mean_abs_diff = float(np.mean(np.abs(np.diff(series))))

                # UPGRADE 4: Add FFT Spectral Energy & Lag-1 Autocorrelation
                fft_vals = np.fft.rfft(series - np.mean(series))
                spectral_energy = float(np.sum(np.abs(fft_vals) ** 2) / len(series))

                # Lag-1 autocorrelation (safe against zero variance when sensor is flat)
                if np.std(series[:-1]) > 1e-6 and np.std(series[1:]) > 1e-6:
                    lag1_autocorr = float(np.corrcoef(series[:-1], series[1:])[0, 1])
                else:
                    lag1_autocorr = 0.0

                feature_summary[col] = {
                    "mean": round(float(np.mean(series)), 4),
                    "std": round(float(np.std(series)), 4),
                    "min": round(float(np.min(series)), 4),
                    "max": round(float(np.max(series)), 4),
                    "slope": round(slope, 5),
                    "mean_abs_change": round(mean_abs_diff, 4),
                    "spectral_energy": round(spectral_energy, 4),
                    "lag1_autocorr": round(lag1_autocorr, 4),
                }

            features_payload = {
                "window_id": self.window_id,
                "dataset": dataset,
                "end_time": self.time_buffer[-1],
                "window_features": feature_summary,
            }
            self.producer.send(TOPIC_FEATURES_REAL, value=features_payload)
            self.producer.flush()

            print(
                f"[PyFlink Pipeline 2] Emitted window_id={self.window_id} | "
                f"Shape=({WINDOW_SIZE}, {len(SENSOR_ORDER)}) | "
                f"CO(GT) mean={feature_summary['CO(GT)']['mean']} | "
                f"CO(GT) spec_energy={feature_summary['CO(GT)']['spectral_energy']}"
            )
            self.window_id += 1
            yield f"window_id={self.window_id - 1}"


def run_flink_pipeline():
    print(f"[PyFlink Pipeline 2] Initializing StreamExecutionEnvironment...")
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    env.add_jars("file:///app/jars/flink-sql-connector-kafka.jar")

    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(KAFKA_BROKER)
        .set_topics(TOPIC_RAW_SENSORS)
        .set_group_id("synstream-flink-window-group")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    stream = env.from_source(
        kafka_source,
        WatermarkStrategy.no_watermarks(),
        "Kafka Raw Sensor Source",
    )

    # Key by dataset name and apply stateful sliding window & feature extraction
    (
        stream.key_by(lambda msg: json.loads(msg).get("dataset", "UCI_AirQuality"))
        .process(SlidingWindowFeatureExtractor(), output_type=Types.STRING())
    )

    print(
        f"[PyFlink Pipeline 2] Listening to '{TOPIC_RAW_SENSORS}' "
        f"(Window={WINDOW_SIZE}, Slide={WINDOW_SLIDE})..."
    )
    env.execute("SynStream-Pipeline2-Window-Processor")


if __name__ == "__main__":
    run_flink_pipeline()