# InfluxDB Schema (Draft v5)

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

## Timestamps

All source timestamps are the original dataset times (2004-2005) and the replay loops. Proposal: the InfluxDB point time is the arrival time, and the original time is kept as a text field `event_time`.

## Measurements

| Measurement | Tags | Fields | Source topic |
|---|---|---|---|
| `sensor_readings` (Sebastian) | `dataset`, `data_source` | one field per sensor | `sensors.raw` |
| `sensor_window` | `dataset`, `stream_type`, `generator`, `sensor` | `value`, `window_id`, `event_time` | `sensors.windowed.real`, `sensors.windowed.synthetic` |
| `anomaly_scores` | `dataset` | `anomaly_score`, `anomaly_threshold`, `max_abs_z`, `missing_sensors`, `is_anomaly`, `forecast_abs_err` | `analytics.scores` |
| `alerts` | `dataset`, `type`, `sensor`, `detector` | `score`, `max_abs_z`, `trigger_retrain`, `event_time` | `analytics.alerts` |
| `similarity_metrics` | `dataset`, `generator` | `mmd_score`, `wasserstein`, `acf_diff`, `xcorr_diff`, `forecast_mae_real`, `forecast_mae_syn`, `dcr`, `dcr_ratio`, `pairs_in_metric`, `window_id`, `event_time` | `analytics.metrics` |

`detector` is only present on ANOMALY alerts (`hst`, `zscore` or `both`).

## Tag values

| Tag | Values |
|---|---|
| `dataset` | `UCI_AirQuality`, `HAI` |
| `data_source` / `stream_type` | `real`, `synthetic` |
| `generator` | `TimeGAN`, `TimeVAE`, `Diffusion`, `none` (real data) |
| `type` | `ANOMALY`, `DRIFT` |
| `sensor` (UCI, 12) | `CO(GT)`, `PT08.S1(CO)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH` |
| `sensor` (HAI) | `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_FT01`, `P1_FT02`, `P1_PIT01`, `P1_TIT01`, `P2_SIT01` |
| `sensor` (DRIFT alerts) | a sensor name, or `forecast_error` |

## Batch results (not stored in InfluxDB)

Batch 2 and Batch 3 results are static tables (`evaluation_results.csv`, `batch3_uci_forecasting.csv`). They are shown in the Streamlit app, not in Grafana.

## Notes

- Windows overlap (size 24, slide 4): only the last 4 rows of each window should be written.
- Sensor names contain parentheses: always quote them in Flux queries.
- `dcr` and `dcr_ratio` only appear when training windows are available to Hazem's service.

## Open questions

- [ ] Sebastian: do the sensor values inside `features` reach `sensor_readings`? Naming: `sensor_readings`/`data_source` vs `sensor_window`/`stream_type`.
- [ ] Hazem: add `analytics.scores` to `common/config.py`.
- [ ] Is the HAI replay producer planned? (Faisal)
