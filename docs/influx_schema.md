# InfluxDB Schema (Draft v6)

Org / bucket / token: as in the team docker-compose (`synstream_org`, `synstream_bucket`).

## Kafka topics read by the sink

| Topic | Producer | Content |
|---|---|---|
| `sensors.raw` | Faisal | `sequence_id`, `event_time`, `ingest_time`, `dataset`, `features` (dict of sensors) |
| `sensors.windowed.real` | Faisal | `window_id`, `dataset`, `start_time`, `end_time`, `sensor_names`, `data_raw`, `data_normalized` (24 x 12) |
| `sensors.windowed.synthetic` | Suyoung | same as real + `stream_type`, `generator` |
| `analytics.scores` | Hazem | per record: `anomaly_score`, `anomaly_threshold`, `max_abs_z`, `missing_sensors`, `is_anomaly`, `forecast_abs_err`, `dataset` |
| `analytics.alerts` | Hazem | `timestamp`, `type`, `sensor`, `score`, `max_abs_z`, `detector`, `trigger_retrain`, `dataset` |
| `analytics.metrics` | Hazem | `window_id`, `generator`, `mmd_score`, `wasserstein`, `acf_diff`, `xcorr_diff`, `forecast_mae_real`, `forecast_mae_syn`, `dcr`, `dcr_ratio`, `pairs_in_metric`, `dataset`, `timestamp` |

## Kafka topics read by the Document 49 services (v6)

These topics are read by two separate services, not by `influx-sink`.

| Topic | Producer | Read by | Content |
|---|---|---|---|
| `sensors.features.real` | Faisal | `feature-summaries` | `window_id`, `dataset`, `end_time`, `window_features` (per sensor: `mean`, `std`, `min`, `max`, `slope`, ...) |
| `sensors.raw` | Faisal | `alert-eval` | ground truth `ground_truth_anomaly` (1 during an attack), `event_time`, `dataset` |
| `analytics.alerts` | Hazem | `alert-eval` | ANOMALY alerts (see above) |

## Timestamps

All source timestamps are the original dataset times (2004-2005) and the replay loops. Proposal: the InfluxDB point time is the arrival time, and the original time is kept as a text field `event_time`.

## Measurements

| Measurement | Tags | Fields | Source topic |
|---|---|---|---|
| `sensor_readings` (Sebastian) | `dataset`, `data_source` | one field per sensor | `sensors.raw` |
| `sensor_window` | `dataset`, `stream_type`, `generator`, `sensor` | `value` (normalized), `value_raw`, `window_id`, `event_time` | `sensors.windowed.real`, `sensors.windowed.synthetic` |
| `anomaly_scores` | `dataset` | `anomaly_score`, `anomaly_threshold`, `max_abs_z`, `missing_sensors`, `is_anomaly`, `forecast_abs_err` | `analytics.scores` |
| `alerts` | `dataset`, `type`, `sensor`, `detector` | `score`, `max_abs_z`, `trigger_retrain`, `event_time` | `analytics.alerts` |
| `similarity_metrics` | `dataset`, `generator` | `mmd_score`, `wasserstein`, `acf_diff`, `xcorr_diff`, `forecast_mae_real`, `forecast_mae_syn`, `dcr`, `dcr_ratio`, `pairs_in_metric`, `window_id`, `event_time` | `analytics.metrics` |
| `feature_bands` (v6, `feature-summaries`) | `dataset`, `sensor`, `feature` | `value`, `n`, `p05`, `p25`, `p50`, `p75`, `p95`, `outside_band`, `window_id`, `event_time` | `sensors.features.real` |
| `alert_evaluation` (v6, `alert-eval`) | `dataset` | `attack_events`, `attacks_detected`, `attacks_missed`, `alerts_total`, `alerts_during_attacks`, `alerts_unmatched`, `event_recall`, `alert_precision_point`, `median_detection_delay_records`, `mean_detection_delay_records` | `sensors.raw`, `analytics.alerts` |
| `attack_event` (v6, `alert-eval`) | `dataset`, `event_id` | `detected`, `length_records`, `start_event_time`, `delay_records` | `sensors.raw`, `analytics.alerts` |

`detector` is only present on ANOMALY alerts (`hst`, `zscore` or `both`).

### v6 measurements: field types

Field types stay fixed so writes never conflict (integers for counts, floats for ratios).

- `feature_bands`: `value`, `p05`, `p25`, `p50`, `p75`, `p95` float; `n`, `window_id` integer; `outside_band` boolean; `event_time` string. The percentile fields are present only after 30 values (`MIN_COUNT`). Each band is computed from past values only, before the new value is added to the t-digest (Dunning and Ertl, 2019, arXiv:1902.04023).
- `alert_evaluation`: the six counts are integers; `event_recall`, `alert_precision_point`, `median_detection_delay_records`, `mean_detection_delay_records` are floats and are written only when defined. One point every 5 seconds per dataset.
- `attack_event`: `detected` boolean; `length_records` integer; `start_event_time` string; `delay_records` float (only if the attack was detected). One point per finished attack.
- Definitions follow `stream_analytics/evaluate_live_run.py`: an attack event is a run of consecutive attack records; it is detected if an ANOMALY alert falls in [start, end + 60 records); detection delay = records from the start of the event to its first alert; point precision = alerts (counted once per record) inside an attack / all alerts.

## Tag values

| Tag | Values |
|---|---|
| `dataset` | `UCI_AirQuality`, `HAI` |
| `data_source` / `stream_type` | `real`, `synthetic` |
| `generator` | `TimeGAN`, `TimeVAE`, `Diffusion`, `none` (real data) |
| `type` | `ANOMALY`, `DRIFT` |
| `feature` (v6) | window feature name, default `mean`, `std` (set by the `FEATURES` variable of `feature-summaries`) |
| `event_id` (v6) | running number of the attack event within a run |
| `sensor` (UCI, 12) | `CO(GT)`, `PT08.S1(CO)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH` |
| `sensor` (HAI, 15) | `P1_FT01`, `P1_FT02`, `P1_FT03`, `P1_LIT01`, `P1_PIT01`, `P1_PIT02`, `P1_TIT02`, `P1_TIT03`, `P2_VIBTR01`, `P2_VIBTR02`, `P2_VIBTR03`, `P2_VIBTR04`, `P3_FIT01`, `P3_LIT01`, `P3_PIT01` (from `stream_analytics/results/hai_sensors.json`) |
| `sensor` (DRIFT alerts) | a sensor name, or `forecast_error` |

## Batch results (not stored in InfluxDB)

Batch 2 and Batch 3 results are static tables (`evaluation_results.csv`, `batch3_uci_forecasting.csv`). They are shown in the Streamlit app, not in Grafana.

## Notes

- Windows overlap (size 24, slide 4): only the last 4 rows of each window should be written.
- Sensor names contain parentheses: always quote them in Flux queries.
- `dcr` and `dcr_ratio` only appear when training windows are available to Hazem's service. Checked in the real HAI run (HAI 22.04 test1.csv): both are written for `HAI`; for `UCI_AirQuality` only a few points exist.
- In the real HAI run only the generator `TimeVAE` is streamed (`generator` tag).
- Sensor names depend on the dataset: a Grafana `sensor` variable that still holds a UCI sensor (e.g. `AH`) shows no data on `dataset=HAI`.
- `alert_evaluation` and `attack_event` need the HAI replay (`DATASET_NAME=HAI`, `river-hai`, `data/hai-22.04/test1.csv`). Training files of HAI contain no attacks.
- Dashboards for the v6 measurements: `synstream-hai-eval.json` and `synstream-feature-bands.json` in `visualization/grafana/provisioning/dashboards/`.

## Open questions

- [ ] Sebastian: do the sensor values inside `features` reach `sensor_readings`? Naming: `sensor_readings`/`data_source` vs `sensor_window`/`stream_type`.
- [ ] Hazem: add `analytics.scores` to `common/config.py`.
- [x] Is the HAI replay producer planned? (Faisal): the replay producer reads `DATASET_NAME=HAI` from `data/hai-22.04/test1.csv`.
- [ ] End-to-end check of `feature_bands` with the full stack (needs Flink output on `sensors.features.real`).
