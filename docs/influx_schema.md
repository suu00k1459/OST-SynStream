# InfluxDB Schema (Draft v3)

Bucket: `synstream` (InfluxDB v2.7). To be agreed with the team.

## Source message: `sensors.windowed.real`

Fields: `window_id`, `dataset`, `start_time`, `end_time`, `sensor_names`, `data_raw` (24 x 12), `data_normalized` (24 x 12).

## Measurements

| Measurement | Tags | Fields | Source topic |
|---|---|---|---|
| `sensor_window` | `dataset`, `stream_type`, `generator`, `sensor` | `value`, `window_id` | `sensors.windowed.real`, `sensors.windowed.synthetic` |
| `similarity_metrics` | `dataset`, `generator` | `mmd_score`, `wasserstein`, `acf_diff`, `dcr` | `analytics.metrics` |
| `alerts` | `dataset`, `type`, `sensor` | `score` | `analytics.alerts` |

## Tag values

| Tag | Values |
|---|---|
| `dataset` | `UCI_AirQuality`, `HAI` |
| `stream_type` | `real`, `synthetic` |
| `generator` | `TimeGAN`, `TimeVAE`, `none` (real data) |
| `type` | `ANOMALY`, `DRIFT` |
| `sensor` (UCI, 12) | `CO(GT)`, `PT08.S1(CO)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH` |
| `sensor` (HAI, 10) | `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_FT01`, `P1_FT02`, `P1_PIT01`, `P1_TIT01`, `P2_SIT01` |

## Notes

- `start_time` and `end_time` are the original recording times of the dataset (2004-2005), not the current time. The replay loops, so these times repeat.
- Windows overlap (size 24, slide 4): to avoid duplicates, only the last 4 rows of each window should be written.
- Sensor names contain parentheses: always quote them in Flux queries.

## Open questions

- [ ] Which timestamp do we use in InfluxDB: arrival time or original time? (Faisal)
- [ ] Format of `sensors.windowed.synthetic` (Suyoung)
- [ ] Format of `analytics.metrics` and `analytics.alerts`, and the field name `mmd_score` (Hazem)
- [ ] Forecasting error fields: where do they come from? (Hazem)
