import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch


SINK_PATH = Path(__file__).resolve().parents[1] / "storage_influx/kafka_to_influx.py"
# Exercise the actual retry function without requiring Docker-only dependencies.
node = next(n for n in ast.parse(SINK_PATH.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "connect_consumer")
sink = types.ModuleType("sink_under_test")
sink.NoBrokersAvailable = type("NoBrokersAvailable", (Exception,), {})
sink.KafkaConsumer = Mock()
sink.time = types.SimpleNamespace(sleep=Mock())
sink.KAFKA_BROKER = "test-broker:9092"
sink.TOPICS = ["sensors.raw", "analytics.metrics"]
sink.json = __import__("json")
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SINK_PATH), "exec"), sink.__dict__)


class SinkStartupTests(unittest.TestCase):
    def test_retries_broker_unavailability_then_connects(self):
        consumer = Mock()
        with patch.object(sink, "KafkaConsumer", side_effect=[
            sink.NoBrokersAvailable(), sink.NoBrokersAvailable(), consumer
        ]) as create, patch.object(sink.time, "sleep") as sleep, patch("builtins.print"):
            self.assertIs(sink.connect_consumer(), consumer)
        self.assertEqual(create.call_count, 3)
        self.assertEqual([c.args for c in sleep.call_args_list], [(5,), (5,)])
        self.assertEqual(create.call_args.args, tuple(sink.TOPICS))
        self.assertEqual(create.call_args.kwargs["group_id"], "influx-sink-group")
        self.assertEqual(create.call_args.kwargs["auto_offset_reset"], "latest")

    def test_configuration_errors_are_not_retried(self):
        with patch.object(sink, "KafkaConsumer", side_effect=ValueError("bad config")), \
             patch.object(sink.time, "sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "bad config"):
                sink.connect_consumer()
        sleep.assert_not_called()

    def test_shutdown_interrupts_retry(self):
        with patch.object(sink, "KafkaConsumer", side_effect=sink.NoBrokersAvailable()), \
             patch.object(sink.time, "sleep", side_effect=KeyboardInterrupt), patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                sink.connect_consumer()
