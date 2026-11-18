# OST-SynStream
Synthetic Time-Series Analytics and Real-Time Monitoring

### Group members: <br>

Sebastian Florez Rojas G9X2XY <br>
Suyoung Kim W3K59J <br>
Hashlamon Hazem Wael Abdalmohsen RFTP59 <br>
Inayat Muhammad Faisal EJLU11 <br>
Attafi hadil CB1RLC<br>

### 1 Project Overview <br>
SynStream will be developed by Team EastWest (EW) as an open-source, containerized platform to generate, monitor, and evaluate synthetic multivariate time-series data from continuously
arriving sensor observations. The system will replay historical telemetry as a live stream, process
observations using sliding windows, periodically generate synthetic time-series data, and evaluate
temporal similarity, concept drift, anomalies, and downstream analytical utility in real time.

### 2 Selected Datasets <br>
Two public multivariate time-series datasets will be used to evaluate the platform:
• UCI Air Quality Dataset: Hourly environmental and chemical sensor measurements (CO,
NOx, NO2, C6H6, temperature, and humidity) will be utilized for synthetic model prototyping, forecasting benchmarks, and seasonal concept-drift analysis.
• HAI / SWaT Industrial Control Dataset: High-frequency cyber-physical telemetry from
industrial control systems with labeled anomalies will be used to test streaming throughput
and evaluate real-time anomaly detectio

### 3. Architecture <br>
<img src="images/SynStream Architecture.png" alt="SynStream architecture" width="800">

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
