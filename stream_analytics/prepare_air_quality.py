
from pathlib import Path

import pandas as pd

# All paths are relative to this Python file.
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "AirQuality.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "real_windows.csv"

FEATURES = [
    "CO(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "C6H6(GT)",
    "T",
    "RH",
]


def main():
    # 1. Check that the original dataset exists.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Place AirQualityUCI.csv beside this Python file."
        )

    # 2. Read the original UCI Air Quality CSV.
    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        decimal=",",
        na_values=["", " "],
    )

    # 3. Clean column names and remove empty trailing columns.
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all")

    print("Columns found:")
    print(df.columns.tolist())

    # 4. Check that required columns exist.
    required_columns = ["Date", "Time"] + FEATURES
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "Check the CSV format."
        )

    # 5. Convert Date and Time into timestamps.
    date_text = df["Date"].astype(str).str.strip()
    time_text = df["Time"].astype(str).str.strip()

    df["timestamp"] = pd.to_datetime(
        date_text + " " + time_text,
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )

    # 6. Keep the timestamp and selected sensor features.
    df = df[["timestamp"] + FEATURES].copy()

    # Convert sensor readings to numeric values.
    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    # 7. Sort observations chronologically.
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # 8. Replace the dataset's missing-value marker.
    df[FEATURES] = df[FEATURES].replace(-200, float("nan"))

    # 9. Fill short gaps using past observations only.
    df[FEATURES] = df[FEATURES].ffill(limit=3)

    # 10. Remove rows with remaining missing sensor values.
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid rows remain. Check the CSV date format "
            "and sensor values."
        )

    # 11. Create the output folder and save the cleaned data.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # 12. Print a summary for verification.
    print("\nData preparation completed successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Valid observations: {len(df)}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Missing sensor values: {int(df[FEATURES].isna().sum().sum())}")

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
