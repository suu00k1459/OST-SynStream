# OST-SynStream
Synthetic Time-Series Analytics and Real-Time Monitoring

### Group members: <br>

Sebastian Florez Rojas G9X2XY <br>
Suyoung Kim W3K59J <br>
Hashlamon Hazem Wael Abdalmohsen RFTP59 <br>
Inayat Muhammad Faisal EJLU11 <br>
Attafi hadil CB1RLC<br>

### 1 Project Overview <br>
SynStream will be developed by Team EastWest (EW) as an open-source, containerized platform to generate, monitor, and evaluate synthetic multivariate time-series data from continuously
arriving sensor observations. The system will replay historical telemetry as a live stream, process
observations using sliding windows, periodically generate synthetic time-series data, and evaluate
temporal similarity, concept drift, anomalies, and downstream analytical utility in real time.

### 2 Selected Datasets <br>
Two public multivariate time-series datasets will be used to evaluate the platform:
• UCI Air Quality Dataset: Hourly environmental and chemical sensor measurements (CO,
NOx, NO2, C6H6, temperature, and humidity) will be utilized for synthetic model prototyping, forecasting benchmarks, and seasonal concept-drift analysis.
• HAI / SWaT Industrial Control Dataset: High-frequency cyber-physical telemetry from
industrial control systems with labeled anomalies will be used to test streaming throughput
and evaluate real-time anomaly detectio

### 3. Architecture <br>
<img src="images/SynStream Architecture.png" alt="SynStream architecture" width="800">
