# Visualization & Storage

This module covers the storage and visualization layer of SynStream.

## Components
- **Kafka-to-InfluxDB sink** (`influx_sink_service.py`): consumes the Kafka topics and writes them to InfluxDB.
- **Grafana dashboards**: real-time monitoring of real vs. synthetic data, alerts, forecasting and similarity metrics.
- **Streamlit app** (`streamlit_app/app.py`): offline reports (PCA/t-SNE, Batch 2/3 results) and demo controls.

## Owner
Hadil Attafi

## Status
In progress. See the GitHub Issues for the task list.

## Local development setup (Issue #2)

Requirements: Docker Desktop.

```
docker compose -f docker-compose.viz-dev.yml up -d
```

- InfluxDB: http://localhost:8086 (org and bucket: `synstream`)
- Grafana: http://localhost:3000 (data source: InfluxDB, Flux)

Note: this is a local development setup. The dashboards built with it are prototypes on mock data and will be rewired to the real Kafka topics once the sink service is ready.
