import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent           # .../synthetic_gen
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))                    # train_*.py (model classes)
sys.path.insert(0, str(REPO))                    # common/config.py

from common.config import (                      # noqa: E402  (shared team contract)
    KAFKA_BROKER, TOPIC_WINDOWED_REAL, TOPIC_WINDOWED_SYN,
    GROUP_SYNTHETIC_GEN, DATASET_NAME, WINDOW_SIZE, WINDOW_SLIDE,
)

MODEL_FILES = {
    "TimeVAE": ROOT / "models" / "timevae_uci.pt",
    "TimeGAN": ROOT / "models" / "timegan_uci.pt",
    "Diffusion": ROOT / "models" / "diffusion_uci.pt",
}


# ---------------------------------------------------------------------------
# 1. Load a trained generator -> a function that returns n windows (z-scale)
# ---------------------------------------------------------------------------
def load_generator(name):
    """Rebuilds the model from its .pt file and returns (generate_fn, checkpoint).
    generate_fn(n) -> numpy array (n, seq_len, n_features), same z-scale as X_train."""
    path = MODEL_FILES[name]
    if not path.exists():
        sys.exit(f"Model file not found: {path}\nTrain it first with the matching train_*.py script.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg, seq_len, n_feat = ckpt["config"], ckpt["seq_len"], ckpt["n_features"]
    clip = cfg["clip"]

    if name == "TimeVAE":
        from train_timevae import TimeVAE
        model = TimeVAE(n_feat, seq_len, cfg["hidden_dim"], cfg["latent_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        raw_fn = lambda n: model.generate(n).numpy()

    elif name == "TimeGAN":
        from train_timegan import TimeGAN, from_unit
        model = TimeGAN(n_feat, cfg["hidden_dim"], cfg["num_layers"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        raw_fn = lambda n: from_unit(model.generate(n, seq_len).numpy(), clip)

    elif name == "Diffusion":
        from train_timediffusion import Diffusion, Denoiser
        model = Denoiser(n_feat, cfg["channels"], cfg["n_blocks"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        diff = Diffusion(cfg["T"])
        raw_fn = lambda n: diff.sample(model, n, seq_len, n_feat, clip).numpy()

    else:
        sys.exit(f"Unknown generator: {name}")

    def generate_fn(n=1):
        with torch.no_grad():
            return np.clip(raw_fn(n), -clip, clip).astype(np.float32)

    return generate_fn, ckpt


# ---------------------------------------------------------------------------
# 2. Wrap one synthetic window in the agreed message format
# ---------------------------------------------------------------------------
def build_message(real_msg, synth_z, ckpt, generator):
    """Same fields as Faisal's real window + stream_type + generator."""
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    std = np.asarray(ckpt["scaler_std"], dtype=np.float64)
    synth_raw = synth_z * std + mean                     # back to original units
    return {
        "window_id": real_msg["window_id"],              # copied from the real window
        "dataset": real_msg.get("dataset", DATASET_NAME),
        "start_time": real_msg.get("start_time"),        # copied: same time axis as real
        "end_time": real_msg.get("end_time"),
        "window_size": int(synth_z.shape[0]),
        "window_slide": real_msg.get("window_slide", WINDOW_SLIDE),
        "sensor_names": list(ckpt["sensor_names"]),
        "data_raw": np.round(synth_raw, 4).tolist(),
        "data_normalized": np.round(synth_z, 4).tolist(),
        "scaler_stats": {"mean": np.round(mean, 4).tolist(), "std": np.round(std, 4).tolist()},
        "stream_type": "synthetic",
        "generator": generator,
    }


def check_sensors(real_msg, ckpt):
    """Stop early if Flink and the model disagree on the sensors."""
    real_names = real_msg.get("sensor_names")
    if real_names is not None and list(real_names) != list(ckpt["sensor_names"]):
        sys.exit("Sensor mismatch between the real window and the model.\n"
                 f"  real : {real_names}\n  model: {list(ckpt['sensor_names'])}")


# ---------------------------------------------------------------------------
# 3a. Dry run: no Kafka
# ---------------------------------------------------------------------------
def fake_real_messages(n, sensor_names):
    """Stand-in real window messages built from X_val (labels only matter here)."""
    from train_timevae import load_split
    split = load_split()
    start = datetime(2004, 12, 20, 8, 0, 0)
    for i in range(n):
        s = start + timedelta(hours=i * WINDOW_SLIDE)
        yield {
            "window_id": i,
            "dataset": DATASET_NAME,
            "start_time": s.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": (s + timedelta(hours=WINDOW_SIZE - 1)).strftime("%Y-%m-%d %H:%M:%S"),
            "window_size": WINDOW_SIZE,
            "window_slide": WINDOW_SLIDE,
            "sensor_names": sensor_names,
            "data_normalized": split["X_val"][i % len(split["X_val"])].tolist(),
        }


def run_dry(generator, every, n_real=10):
    generate_fn, ckpt = load_generator(generator)
    out, t0 = [], time.time()
    for k, real in enumerate(fake_real_messages(n_real, list(ckpt["sensor_names"]))):
        check_sensors(real, ckpt)
        if k % every:
            continue
        msg = build_message(real, generate_fn(1)[0], ckpt, generator)
        out.append(msg)
        print(f"[dry-run] real window {real['window_id']:>3} -> synthetic {generator} "
              f"{np.array(msg['data_raw']).shape}, CO(GT) first hour = {msg['data_raw'][0][0]}")
    secs = (time.time() - t0) / max(len(out), 1)
    path = ROOT / "outputs" / f"sample_synthetic_messages_{generator.lower()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out[:3], indent=2))
    print(f"\n{len(out)} messages built, {secs:.2f}s per window. First 3 saved -> {path}")


# ---------------------------------------------------------------------------
# 3b. Live: Kafka in, Kafka out
# ---------------------------------------------------------------------------
def connect(make, what):
    from kafka.errors import KafkaError        # base class, same name in every kafka-python version
    while True:
        try:
            return make()
        except KafkaError:
            print(f"Waiting for Kafka at {KAFKA_BROKER} ({what}) ...")
            time.sleep(2)


def run_live(generator, every):
    from kafka import KafkaConsumer, KafkaProducer

    generate_fn, ckpt = load_generator(generator)
    consumer = connect(lambda: KafkaConsumer(
        TOPIC_WINDOWED_REAL,
        bootstrap_servers=KAFKA_BROKER,
        group_id=f"{GROUP_SYNTHETIC_GEN}-{generator.lower()}",   # one group per generator
        auto_offset_reset="latest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    ), "consumer")
    producer = connect(lambda: KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    ), "producer")
    print(f"{generator} service running: {TOPIC_WINDOWED_REAL} -> {TOPIC_WINDOWED_SYN} "
          f"(every {every} window(s))")

    for k, record in enumerate(consumer):
        real = record.value
        check_sensors(real, ckpt)
        if k % every:
            continue
        msg = build_message(real, generate_fn(1)[0], ckpt, generator)
        producer.send(TOPIC_WINDOWED_SYN, key=msg["dataset"].encode("utf-8"), value=msg)
        print(f"{datetime.now(timezone.utc):%H:%M:%S} real window {real['window_id']} "
              f"-> synthetic {generator}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live synthetic window generator")
    p.add_argument("--generator", default=os.getenv("GENERATOR", "TimeVAE"), choices=list(MODEL_FILES))
    p.add_argument("--every", type=int, default=int(os.getenv("GENERATE_EVERY", "1")))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    (run_dry if a.dry_run else run_live)(a.generator, max(a.every, 1))
