import ast
from collections import deque
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from common import config, datasets
from ingestion_flink import replay_producer as replay

SENSORS = config.DATASET_CONFIGS["HAI"]["sensors"]


class HAITests(unittest.TestCase):
    def fixture(self):
        rows = [{**{s: float(i + 1) for s in SENSORS},
                 "timestamp": f"2022-08-09 00:00:{i:02d}", "Attack": i % 2}
                for i in range(28)]
        return pd.DataFrame(rows)

    def load(self, folder, frame):
        path = Path(folder) / "hai-22.04"
        path.mkdir(exist_ok=True)
        frame.to_csv(path / "test1.csv", index=False)
        with patch.object(datasets, "DATASET_NAME", "HAI"), patch.object(datasets, "ACTIVE_SENSORS", SENSORS):
            return datasets.load_or_download_dataset(folder)

    def test_sensor_contract_matches_hazem(self):
        path = Path(__file__).resolve().parents[1] / "stream_analytics/results/hai_sensors.json"
        self.assertEqual(SENSORS, json.loads(path.read_text()))
        self.assertEqual(len(SENSORS), 15)

    def test_hai_loading_and_one_pass_messages(self):
        with tempfile.TemporaryDirectory() as folder:
            frame, sensors = self.load(folder, self.fixture())
        self.assertEqual(frame.event_time.tolist(), self.fixture().timestamp.tolist())
        producer = Mock()
        with patch.object(replay, "load_or_download_dataset", return_value=(frame, sensors)), \
             patch.object(replay, "connect_kafka", return_value=producer), \
             patch.object(replay, "DATASET_NAME", "HAI"), \
             patch.object(replay, "REPLAY_LOOP", False), \
             patch.object(replay.time, "sleep"), patch("builtins.print"):
            replay.run_replay()
        messages = [c.kwargs["value"] for c in producer.send.call_args_list]
        self.assertEqual(len(messages), 28)
        self.assertEqual([m["ground_truth_anomaly"] for m in messages], self.fixture().Attack.tolist())
        self.assertTrue(all(m["dataset"] == "HAI" and list(m["features"]) == SENSORS for m in messages))
        producer.flush.assert_called_once()
        producer.close.assert_called_once()

    def test_missing_file_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(datasets, "DATASET_NAME", "HAI"):
                with self.assertRaises(FileNotFoundError):
                    datasets.load_or_download_dataset(folder)
            for kind in ["sensor", "label", "time"]:
                frame = self.fixture()
                if kind == "sensor": frame = frame.drop(columns=SENSORS[0])
                if kind == "label": frame.loc[0, "Attack"] = 2
                if kind == "time": frame.loc[0, "timestamp"] = "invalid"
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    self.load(folder, frame)

    def test_actual_flink_operator_for_both_sensor_counts(self):
        # Execute the real operator class with a fake Kafka producer, without a JVM.
        path = Path(__file__).resolve().parents[1] / "ingestion_flink/flink_window_processor.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        for name in ["UCI_AirQuality", "HAI"]:
            sensors = config.DATASET_CONFIGS[name]["sensors"]
            producer = Mock()
            base = type("KeyedProcessFunction", (), {"Context": object})
            ns = dict(np=np, json=json, deque=deque, KeyedProcessFunction=base,
                      RuntimeContext=object, KafkaProducer=lambda **kw: producer,
                      SENSOR_ORDER=sensors, WINDOW_SIZE=24, WINDOW_SLIDE=4,
                      KAFKA_BROKER="unused", TOPIC_WINDOWED_REAL="windows",
                      TOPIC_FEATURES_REAL="features")
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            operator = ns[node.name]()
            operator.open(None)
            with patch("builtins.print"):
                for i in range(28):
                    record = dict(dataset=name, event_time=str(i), features={s: i + 1 for s in sensors})
                    list(operator.process_element(json.dumps(record), None))
            windows = [c.kwargs["value"] for c in producer.send.call_args_list if c.args[0] == "windows"]
            self.assertEqual([w["window_id"] for w in windows], [0, 1])
            self.assertEqual([w["end_time"] for w in windows], ["23", "27"])
            for w in windows:
                self.assertEqual(np.asarray(w["data_raw"]).shape, (24, len(sensors)))
                self.assertTrue(np.isfinite(w["data_normalized"]).all())
                self.assertLessEqual(np.max(np.abs(w["data_normalized"])), 4)
