# InfluxDB Schema (Draft v5)

Bucket: `synstream_bucket`, org `synstream_org` (InfluxDB v2.7, values from the team docker-compose).
Based on the real message formats of Faisal (windows), Suyoung (synthetic windows), Hazem (alerts, metrics, scores) and Sebastian (sink).

## Timestamps

Points are written with their **arrival time**, so that live dashboards work. The replay loops over the 2004-2005 dataset, so original times repeat. The original time is kept in the string field `event_time` when the message has one.

## Measurements

| Measurement | Source topic | Tags | Fields |
|---|---|---|---|
| `sensor_readings` | `sensors.raw` | `dataset`, `data_source` | one field per sensor (the values are nested in `features` in the message), `sequence_id` |
| `sensor_window` | `sensors.windowed.real`, `sensors.windowed.synthetic` | `dataset`, `data_source`, `generator`, `sensor` | `value` (normalized), `value_raw`, `window_id` |
| `similarity_metrics` | `analytics.metrics` | `dataset`, `generator` | `mmd_score`, `wasserstein`, `acf_diff`, `xcorr_diff`, `dcr`, `dcr_ratio`, `forecast_mae_real`, `forecast_mae_syn`, `pairs_in_metric`, `window_id` |
| `alerts` | `analytics.alerts` | `dataset`, `type`, `sensor`, `detector` | `score`, `max_abs_z`, `trigger_retrain` |
| `anomaly_scores` | `analytics.scores` | `dataset` | `anomaly_score`, `anomaly_threshold`, `max_abs_z`, `is_anomaly`, `forecast_abs_err`, `missing_sensors` |

## Tag values

| Tag | Values |
|---|---|
| `dataset` | `UCI_AirQuality`, `HAI` |
| `data_source` | `real`, `synthetic` |
| `generator` | `TimeGAN`, `TimeVAE`, `Diffusion`, `none` (real data) |
| `type` | `ANOMALY`, `DRIFT` |
| `detector` | `hst`, `zscore`, `both` (only for ANOMALY alerts) |
| `sensor` (UCI, 12) | `CO(GT)`, `PT08.S1(CO)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH` |
| `sensor` (HAI, 10) | `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_FT01`, `P1_FT02`, `P1_PIT01`, `P1_TIT01`, `P2_SIT01` |

For `alerts`, `sensor` is the most deviating sensor (ANOMALY) or the monitored sensor. A DRIFT alert can also have `sensor = forecast_error` (drift of the online forecaster error).

## Batch results (not stored in InfluxDB)

Batch 2 (`evaluation_results.csv`, Suyoung) and Batch 3 (`batch3_uci_forecasting.csv`, Hazem) are static tables shown in the Streamlit app, not in Grafana.

## Notes

- Windows overlap (size 24, slide 4): only the last 4 rows of each window are written, to avoid duplicates.
- Real and synthetic windows are linked by `window_id`.
- `analytics.scores` sends one message per record (about 5 per second): it is written as is, and can be downsampled if the bucket grows too much.
- Sensor names contain parentheses: always quote them in Flux queries.

## Open questions

- [ ] Sebastian: sensor values are nested in `features`: are they written in `sensor_readings`? Agree on the names `sensor_readings` and `data_source`.
- [ ] Suyoung: will the online service publish `sensors.windowed.synthetic` in the same format as the sample JSON, for TimeGAN, TimeVAE and Diffusion?
- [ ] Hazem: confirm the field names above (alerts, metrics, scores). Is `analytics.scores` needed at one message per record?
- [ ] Faisal: will the HAI replay also send the `Attack` label?
