import io
import os
import runpy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from common import datasets
from ingestion_flink import replay_producer as replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "event_time": ["2004-03-10T18:00:00Z", "2004-03-10T19:00:00Z"],
            "CO(GT)": [2.6, -200.0],
        })
        self.producer = Mock()
        self.enterContext(patch.object(replay, "load_or_download_dataset",
                                      return_value=(self.frame, ["CO(GT)"])))
        self.enterContext(patch.object(replay, "connect_kafka", return_value=self.producer))
        self.sleep = self.enterContext(patch.object(replay.time, "sleep"))
        self.enterContext(patch("builtins.print"))

    def test_one_pass_sends_each_record_once_and_closes(self):
        with patch.object(replay, "REPLAY_LOOP", False):
            replay.run_replay()
        messages = [c.kwargs["value"] for c in self.producer.send.call_args_list]
        self.assertEqual([m["sequence_id"] for m in messages], [0, 1])
        self.assertEqual([m["event_time"] for m in messages], self.frame.event_time.tolist())
        self.assertEqual([m["features"]["CO(GT)"] for m in messages], [2.6, -200.0])
        self.producer.flush.assert_called_once_with()
        self.producer.close.assert_called_once_with()

    def test_loop_restarts_dataset_with_continuous_sequence_ids(self):
        self.sleep.side_effect = [None, None, KeyboardInterrupt]
        with patch.object(replay, "REPLAY_LOOP", True):
            with self.assertRaises(KeyboardInterrupt):
                replay.run_replay()
        messages = [c.kwargs["value"] for c in self.producer.send.call_args_list]
        self.assertEqual([m["sequence_id"] for m in messages], [0, 1, 2])
        self.assertEqual(messages[2]["event_time"], messages[0]["event_time"])
        self.producer.flush.assert_called_once_with()
        self.producer.close.assert_called_once_with()

    def test_connection_closes_even_if_flush_fails(self):
        self.producer.flush.side_effect = RuntimeError("flush failed")
        with patch.object(replay, "REPLAY_LOOP", False):
            with self.assertRaisesRegex(RuntimeError, "flush failed"):
                replay.run_replay()
        self.producer.close.assert_called_once_with()


class ConfigurationAndDownloadTests(unittest.TestCase):
    def test_replay_loop_environment(self):
        config_path = Path(__file__).resolve().parents[1] / "common" / "config.py"
        with patch.dict(os.environ):
            os.environ.pop("REPLAY_LOOP", None)
            self.assertTrue(runpy.run_path(str(config_path))["REPLAY_LOOP"])
            for value, expected in [("false", False), ("False", False), ("0", False),
                                    ("true", True), (" TRUE ", True)]:
                with self.subTest(value=value):
                    os.environ["REPLAY_LOOP"] = value
                    self.assertEqual(runpy.run_path(str(config_path))["REPLAY_LOOP"], expected)

    def test_download_path_when_csv_is_missing(self):
        row = {name: 1.0 for name in datasets.ACTIVE_SENSORS}
        row.update(Date="10/03/2004", Time="18.00.00")
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("AirQualityUCI.csv", pd.DataFrame([row]).to_csv(
                index=False, sep=";", decimal=","))
        response = Mock(content=content.getvalue())
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(datasets.requests, "get", return_value=response) as get:
                frame, sensors = datasets.load_or_download_dataset(folder)
            get.assert_called_once_with(
                "https://archive.ics.uci.edu/static/public/360/air+quality.zip", timeout=60)
            response.raise_for_status.assert_called_once_with()
            self.assertTrue((Path(folder) / "AirQualityUCI.csv").exists())
            self.assertEqual(frame.event_time.tolist(), ["2004-03-10T18:00:00Z"])
            self.assertEqual(sensors, datasets.ACTIVE_SENSORS)


if __name__ == "__main__":
    unittest.main()
