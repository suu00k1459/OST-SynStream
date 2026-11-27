# Visualization & Storage

Grafana dashboards, the Streamlit report app and the two Document 49 stream stages of SynStream. Owner of this folder: Hadil Attafi.

## What is here, and who owns what

| Part | Where | Owner |
|---|---|---|
| Kafka-to-InfluxDB sink (consumer loop, downsampling, health monitor) | `storage_influx/kafka_to_influx.py` | Sebastian |
| Write handlers for window and analytics topics | `storage_influx/handlers_windows.py`, `storage_influx/handlers_analytics.py` | Hadil |
| InfluxDB schema | `docs/influx_schema.md` | Hadil |
| Grafana provisioning and dashboards | `visualization/grafana/provisioning/` | Hadil (LTTB dashboards: Sebastian) |
| Streamlit app (Batch 2, Batch 3, PCA / t-SNE) | `visualization/streamlit_app/app.py` | Hadil |
| Stage 1: online alert evaluation on HAI | `alert_evaluation/online_alert_eval.py` | Hadil |
| Stage 2: t-digest feature summaries | `stream_summaries/feature_summaries.py` | Hadil |
| Mock producer (test tool, not part of the real pipeline) | `visualization/mock_producer.py` | Hadil |

## Dashboards

Folder `SynStream` in Grafana (http://localhost:3000, user `admin`, password `synstream_password`, development values only):

- **Live monitoring** and **Live monitoring v2** (generator selector, DRIFT markers)
- **HAI attack detection (online evaluation)**: event recall, point precision, detection delay per attack
- **Feature bands (t-digest)**: value of a window feature against its p05 to p95 band
- two LTTB dashboards (Sebastian). They stay empty until `storage_influx/load_hai_lttb.py` is run by hand (`HAI_CSV_PATH` set, `influxdb-client` installed).

The `Sensor` dropdown does not follow the `Dataset` dropdown: pick a sensor of the chosen dataset, otherwise a panel shows "No data".

## Stage 1 and stage 2

Both are Kafka consumers started by `docker compose` (`alert-eval`, `feature-summaries`) and write to InfluxDB (`alert_evaluation`, `attack_event`, `feature_bands`). Details and field types: `docs/influx_schema.md`.

- Stage 1 reads `sensors.raw` (ground truth) and `analytics.alerts`. It needs a HAI run (`test1.csv`, which contains the attacks).
- Stage 2 reads `sensors.features.real`.

## Try the stages without the full pipeline

```
docker compose up -d --build kafka influxdb grafana alert-eval feature-summaries
docker compose run --rm --no-deps -v "%cd%\visualization:/mock" alert-eval python /mock/mock_producer.py --hz 20
```

Then open the dashboards and set the time range to the last few hours. The mock sends a simulated HAI stream with 3 attacks. Expected on the HAI dashboard: event recall 66.7 %, point precision 50 %, 2 of 3 attacks detected. `python visualization/mock_producer.py --check` runs the stage 1 logic without Kafka.

The mock follows the message format of `ingestion_flink/replay_producer.py` and `ingestion_flink/flink_window_processor.py` (compared in the code, not by running Flink).

## Real HAI run

Put the HAI 22.04 `test1.csv` in `data/hai-22.04/` (data files are not in the repository), then start the stack with Faisal's launcher:

```
bash run_hai.sh start --build
```

On Windows this needs "Enable integration with my default WSL distro" in Docker Desktop (Settings, Resources, WSL integration). If you ran the mock producer before, delete the old Kafka topics and the `dataset=HAI` points in InfluxDB first, otherwise `river-hai` stops on the simulated records (they lack sensors). Never use `docker compose down -v` for this.

Result of one run (86,400 records): 7 attack events, 3 detected by the ANOMALY alerts of `river-hai` (event recall 42.9 %), median detection delay 65 records, point precision 11.9 %. The numbers describe the detector; stage 1 measures it.

## Streamlit app

With Docker (from the repository root), then open http://localhost:8501:

```
docker compose -f visualization/streamlit_app/docker-compose.yml up --build -d
```

Without Docker:

```
pip install -r visualization/streamlit_app/requirements.txt
streamlit run visualization/streamlit_app/app.py
```

The result files are read from `stream_analytics/results/` and `synthetic_gen/outputs/`. Tabs: Batch 2 (similarity), Batch 3 (forecasting), PCA / t-SNE of real against synthetic windows. These are static results, not live data.

## Status

- Tested end to end on the real streams, from a fresh download with a clean Docker setup: UCI (full stack) and HAI 22.04 `test1.csv` (full stack with `run_hai.sh`). Both stages and all Grafana dashboards of this folder showed data on HAI.
- Earlier tests used the mock producer (simulated data).
- Not verified: the LTTB dashboards without the manual loader (Sebastian), and the fine-tuning recovery of the generator (measured by Hazem, mixed result).
- Streamlit: all three tabs checked on the real result files (PCA and t-SNE on HAI and UCI), also inside the Docker container. The service is not in the main `docker-compose.yaml` (separate compose file).
- Task list: GitHub Issues.

## Local development setup (Issue #2)

Separate small stack for dashboard prototyping (own InfluxDB with org and bucket `synstream`, development token only):

```
docker compose -f visualization/docker-compose.viz-dev.yml up -d
```

- InfluxDB: http://localhost:8086
- Grafana: http://localhost:3000 (data source: InfluxDB, Flux)

Do not run it at the same time as the team stack: both use ports 8086 and 3000.
