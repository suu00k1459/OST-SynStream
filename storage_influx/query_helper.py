import os
import pandas as pd
from influxdb_client import InfluxDBClient

def fetch_dataframe(measurement: str = "sensor_readings", data_source: str = "real", lookback: str = "-1h") -> pd.DataFrame:
    client = InfluxDBClient(
        url=os.getenv("INFLUX_URL", "http://localhost:8086"),
        token=os.getenv("INFLUX_TOKEN", "synstream-secret-token"),
        org=os.getenv("INFLUX_ORG", "synstream_org")
    )
    query = f'''
    from(bucket: "{os.getenv("INFLUX_BUCKET", "synstream_bucket")}")
      |> range(start: {lookback})
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["data_source"] == "{data_source}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    df = client.query_api().query_data_frame(query)
    if not df.empty and "_time" in df.columns:
        df = df.set_index("_time").drop(columns=["result", "table", "_start", "_stop", "_measurement"], errors="ignore")
    return df