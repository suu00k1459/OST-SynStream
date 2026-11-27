# OST-SynStream

### Bash launchers: one dataset at a time

```bash
bash run_uci.sh start --build  # build and start UCI, TimeVAE, analytics and storage
bash run_hai.sh start --build  # switch to HAI and its TimeVAE checkpoint
```

After building, use `start` without `--build`. Both launchers support `build`,
`test`, `logs`, `status`, and `stop`; for example `bash run_hai.sh logs`.
They share `run_pipeline.sh`. UCI defaults to 5 Hz, HAI to 100 Hz, and both
force one-pass replay. Override the rate for one invocation with
`REPLAY_HZ=200 bash run_hai.sh start`.

Starting either dataset stops the shared application services and the other
River service before recreating the selected pipeline. They are not simultaneous
launchers. HAI requires the local `data/hai-22.04/test1.csv` and
`synthetic_gen/models/timevae_hai.pt`; UCI selects `timevae_uci.pt`.
Both start feature summaries; HAI additionally starts alert evaluation.

The launchers check Kafka, allow consumers 30 seconds to initialize, and check
that their containers are running before starting replay. Container state is
not a strict Flink readiness guarantee. Existing Kafka offsets/history and
database records are retained. `stop` shuts down the shared infrastructure
without deleting data. `test` runs the suite against a read-only bind mount
of this checkout in the replay image; build that image first.

Verify stored measurements and dashboards separately after startup. A successful
launcher does not establish that the whole experiment or replay has completed.

The InfluxDB sink waits and retries every five seconds if Kafka is unavailable
at startup. Compose also restarts the sink after an unexpected exit. Rebuild
and recreate it after source changes:

```bash
docker compose build influx-sink
docker compose up -d --no-deps --force-recreate influx-sink
docker compose logs --tail 30 influx-sink
```

The sink resumes its existing Kafka consumer group offsets, so retained data may
take time to drain before new points appear. This startup fix does not provide
exactly-once delivery or change the existing handling of database write errors.
Synthetic Time-Series Analytics and Real-Time Monitoring

### Group members: <br>

| Name | Neptun code |
|---|---|
| Sebastian Florez Rojas | G9X2XY |
| Suyoung Kim | W3K59J |
| Hashlamon Hazem Wael Abdalmohsen | RFTP59 |
| Inayat Muhammad Faisal | EJLU11 |
| Attafi Hadil | CB1RLC |

## Project Overview <br>
SynStream will be developed by Team EastWest (EW) as an open-source, containerized platform to generate, monitor, and evaluate synthetic multivariate time-series data from continuously
arriving sensor observations. The system will replay historical telemetry as a live stream, process
observations using sliding windows, periodically generate synthetic time-series data, and evaluate
temporal similarity, concept drift, anomalies, and downstream analytical utility in real time.

## Contents
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Flow & Viewer](#flow-&-viewer)
- [Docker Deployment Guide](#docker-deployment-guide)
- [Models & Evaluation](#models-&-evaluation)
- [Evaluation](#evaluation)
- [How to run](#how-to-run)
- [References](#references)

---
## Datasets 

Two public multivariate time-series datasets will be used to evaluate the platform:

• UCI Air Quality Dataset: Hourly environmental and chemical sensor measurements (CO,
NOx, NO2, C6H6, temperature, and humidity) will be utilized for synthetic model prototyping, forecasting benchmarks, and seasonal concept-drift analysis.

• HAI / SWaT Industrial Control Dataset: High-frequency cyber-physical telemetry from
industrial control systems with labeled anomalies will be used to test streaming throughput
and evaluate real-time anomaly detectio

---

## Architecture 

<img src="images/SynStream Architecture.png" alt="SynStream architecture" width="800">

---

## Flow & Viewer

---

## Docker Deployment Guide

---

## Models & Evaluation
###Models

###Evaluation

###Results

---

## How to run

---

## References
### Replay mode

Replay loops by default (`REPLAY_LOOP=true`). To replay the dataset once and stop
without jumping back to 2004, run from PowerShell:

```powershell
$env:REPLAY_LOOP = "false"
docker compose up -d --build replay-producer
```

The producer flushes pending Kafka messages and closes its connection before
exiting. Set `REPLAY_LOOP` to `true` and rerun the command to restore looping.
The option also works when running the producer directly as a Python module:
`python -m ingestion_flink.replay_producer`.

Existing Kafka messages remain available; coordinate consumer offsets with the
analytics team when starting a fresh experiment.

### HAI 22.04 replay

Place the original HAI 22.04 `test1.csv` in `data/hai-22.04/` locally
(`/app/data/hai-22.04/test1.csv` inside the replay container). Data files are not
committed. Missing files, required columns, invalid labels, and invalid or
out-of-order timestamps fail with an explanatory error before Kafka connects.

HAI replay preserves the source timestamp text as `event_time`, sets `dataset`
to `HAI`, and publishes the agreed 15 sensors in `features`. Every raw message
also contains `ground_truth_anomaly`: HAI's `Attack` (0/1), or 0 for UCI.
Flink emits 24-by-15 HAI windows using the existing window/feature schemas.
Windows count observations: 24 HAI observations represent 24 seconds, whereas
24 hourly UCI observations represent 24 hours.

Run the regression tests locally:

```powershell
python -m unittest discover -s tests -v
```

Configure a one-pass HAI run in the same PowerShell session:

```powershell
$env:DATASET_NAME = "HAI"
$env:REPLAY_HZ = "100"
$env:REPLAY_LOOP = "false"
docker compose config --quiet
docker compose build replay-producer flink-processor river-hai
docker compose stop replay-producer flink-processor river-uci
docker compose up -d kafka kafka-ui
```

Wait until the following Kafka command succeeds, then start the consumers before
replay. `--no-deps` prevents Flink's dependency from starting replay prematurely.

```powershell
docker exec synstream-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --list
docker compose up -d --no-deps --force-recreate flink-processor river-hai
docker compose logs --tail 30 flink-processor river-hai
```

Allow Flink and River to initialize and check their logs before starting replay:

```powershell
docker compose up -d --no-deps --force-recreate replay-producer
docker compose logs --tail 20 replay-producer flink-processor river-hai
docker compose ps -a
```

Flink still uses latest starting offsets, so this sequence reduces startup loss
but is not a strict readiness guarantee. Existing Kafka history and saved River
consumer offsets must be coordinated separately for a clean experiment.

Check `sensors.raw` for HAI timestamps, 15 features and 0/1 labels;
`sensors.windowed.real` for 24-by-15 windows; `sensors.features.real` for 15
sensor summaries; and `analytics.scores` / `analytics.alerts` for River outputs.
Raw missing values may use -200; Flink performs causal imputation.

For 86,400 input rows, one pass takes at least 14.4 minutes at 100 Hz or 7.2
minutes at 200 Hz, plus overhead. Successful completion logs the actual row
count and exits with code 0. These are pacing targets, not measured throughput.
Use the original data first; artificial drift injection is not enabled.

To restore UCI settings, set `DATASET_NAME=UCI_AirQuality` and `REPLAY_HZ=5`
in PowerShell, then recreate replay and Flink and select `river-uci`.
