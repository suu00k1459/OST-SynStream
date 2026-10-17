# InfluxDB Schema (Draft v2)

Bucket: `synstream` (InfluxDB v2.7). To be agreed with the team.

## Source message (Pipeline 1, topic `sensors.raw`)

```json
{"sequence_id": 0, "event_time": "...", "dataset": "UCI_AirQuality",
 "features": {"CO(GT)": 2.6, "T": 13.6, "RH": 48.9, "...": 0.0}}
```

## Measurements

| Measurement | Tags | Fields | Source topic |
|---|---|---|---|
| `sensor_window` | `dataset`, `stream_type`, `generator`, `sensor` | `value`, `window_id` | `sensors.windowed.real`, `sensors.windowed.synthetic` |
| `similarity_metrics` | `dataset`, `generator` | `mmd`, `wasserstein`, `acf_diff`, `dcr` | `analytics.metrics` |
| `forecast_perf` | `dataset`, `generator` | `mae_real`, `mae_syn` | `analytics.metrics` |
| `alerts` | `dataset`, `type`, `sensor` | `score` | `analytics.alerts` |

## Allowed tag values

| Tag | Values |
|---|---|
| `dataset` | `UCI_AirQuality`, `HAI` |
| `stream_type` | `real`, `synthetic` |
| `generator` | `TimeGAN`, `TimeVAE`, `none` (real data) |
| `type` | `ANOMALY`, `DRIFT` |
| `sensor` | UCI: `CO(GT)`, `PT08.S1(CO)`, `NMHC(GT)`, `C6H6(GT)`, `PT08.S2(NMHC)`, `NOx(GT)`, `PT08.S3(NOx)`, `NO2(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH`. HAI: to be defined |

## Design rules

- Low-cardinality values are **tags**; numeric values that change over time are **fields**.
- Real and synthetic windows are linked by `window_id`.
- Window matrices (time steps x sensors) are flattened into one point per time step and sensor.
- Sensor names contain parentheses: always quote them in Flux queries.

## Open questions

- [ ] `event_time` is currently the replay (wall-clock) time, not the original dataset timestamp. Intended? (Faisal)
- [ ] Add `ground_truth_anomaly` for HAI (needed for the anomaly dashboard) (Faisal)
- [ ] Output format of `sensors.windowed.real` (Faisal)
- [ ] Exact field names in `analytics.metrics` (Hazem)
- [ ] Do synthetic windows keep the same `window_id` as real ones? (Suyoung)
- [ ] HAI sensor names
