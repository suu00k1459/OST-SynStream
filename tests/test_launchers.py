import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if not BASH and Path("C:/Program Files/Git/bin/bash.exe").exists():
    BASH = "C:/Program Files/Git/bin/bash.exe"


@unittest.skipUnless(BASH, "Bash is required for launcher tests")
class LauncherTests(unittest.TestCase):
    def run_launcher(self, dataset, fail=False, missing=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["run_pipeline.sh", "run_uci.sh", "run_hai.sh"]:
                shutil.copy(ROOT / name, root / name)
            models = root / "synthetic_gen/models"
            models.mkdir(parents=True)
            for name in ["timevae_uci.pt", "timevae_hai.pt"]:
                (models / name).touch()
            data = root / "data/hai-22.04"
            data.mkdir(parents=True)
            if not missing:
                (data / "test1.csv").touch()
            env = dict(os.environ, MOCK_LOG=str(root / "calls.log").replace("\\", "/"),
                       FAIL_CHECK="1" if fail else "0", MODEL_FILE="wrong.pt")
            env.pop("REPLAY_HZ", None)
            shell = r'''
function docker() {
    printf "%s|%s|%s|%s|%s\n" "$DATASET_NAME" "$MODEL_FILE" "$REPLAY_HZ" "$REPLAY_LOOP" "$*" >> "$MOCK_LOG"
    if [[ "$*" == "compose ps -a -q "* ]]; then echo test-id; fi
    if [[ "$1" == inspect ]]; then
        if [[ "$FAIL_CHECK" == 1 ]]; then echo exited; else echo running; fi
    fi
    return 0
}
function sleep() { return 0; }
export -f docker sleep
bash "$1" start
'''
            result = subprocess.run([BASH, "-c", shell, "test", f"./run_{dataset}.sh"],
                                    cwd=root, env=env, capture_output=True, text=True)
            log = (root / "calls.log").read_text() if (root / "calls.log").exists() else ""
            return result, log

    def test_dataset_selection_and_consumer_order(self):
        for ds, label, rate in [("uci", "UCI_AirQuality", "5"), ("hai", "HAI", "100")]:
            with self.subTest(dataset=ds):
                result, log = self.run_launcher(ds)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = log.splitlines()
                self.assertTrue(all(c.startswith(f"{label}|timevae_{ds}.pt|{rate}|false|") for c in calls))
                consumer = next(i for i, c in enumerate(calls) if "compose up -d --no-deps --force-recreate flink-processor" in c)
                replay = next(i for i, c in enumerate(calls) if c.endswith("compose up -d --no-deps --force-recreate replay-producer"))
                self.assertLess(consumer, replay)
                self.assertIn(f"river-{ds}", calls[consumer])
                self.assertEqual("alert-eval" in calls[consumer], ds == "hai")

    def test_failed_consumer_prevents_replay(self):
        result, log = self.run_launcher("uci", fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("compose up -d --no-deps --force-recreate replay-producer", log)

    def test_missing_hai_file_fails_before_docker(self):
        result, log = self.run_launcher("hai", missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing local HAI file", result.stderr)
        self.assertEqual(log, "")
