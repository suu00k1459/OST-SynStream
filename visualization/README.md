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
