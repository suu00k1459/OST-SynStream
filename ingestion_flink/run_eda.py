import os
import numpy as np
import pandas as pd

DATA_PATH = "/app/data/AirQualityUCI.csv"

ALL_SENSORS = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH"
]

def max_consecutive_nans(series: pd.Series) -> int:
    """Calculates the longest consecutive streak of missing (-200) hours."""
    is_na = series.isna().astype(int)
    streaks = is_na.groupby((~ series.isna()).cumsum()).sum()
    return int(streaks.max()) if len(streaks) > 0 else 0

def run_eda():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Run replay_producer.py once first.")
        return

    df = pd.read_csv(DATA_PATH, sep=";", decimal=",")
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # Convert to numeric and replace -200 sentinel with NaN
    for col in ALL_SENSORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-200.0, np.nan)

    n_rows = len(df)
    print("=" * 88)
    print(f"SYNSTREAM EDA REPORT: UCI AIR QUALITY (Total Rows = {n_rows})")
    print("=" * 88)
    print(f"{'Feature':<15} | {'Missing (-200)':<15} | {'Max Gap (hrs)':<14} | {'Mean':<9} | {'Std':<9} | {'Recommendation'}")
    print("-" * 88)

    keep_features = []
    drop_features = []

    for col in ALL_SENSORS:
        n_miss = int(df[col].isna().sum())
        pct_miss = (n_miss / n_rows) * 100.0
        max_gap = max_consecutive_nans(df[col])
        mean_val = df[col].mean()
        std_val = df[col].std()

        # Criterion 1: Drop if > 30% missing or consecutive gap > 500 hours
        if pct_miss > 30.0 or max_gap > 500:
            rec = "DROP (>30% missing)"
            drop_features.append((col, f"{pct_miss:.1f}% missing, {max_gap}h gap"))
        else:
            rec = "KEEP"
            keep_features.append(col)

        print(
            f"{col:<15} | {n_miss:>5} ({pct_miss:>5.1f}%) | "
            f"{max_gap:>10} hrs | {mean_val:>9.2f} | {std_val:>9.2f} | {rec}"
        )

    # Criterion 2: Check High Multicollinearity (|r| > 0.90) on clean features
    print("\n" + "=" * 88)
    print("HIGH CORRELATION PAIRS (|Pearson r| > 0.85)")
    print("=" * 88)
    corr_matrix = df[keep_features].corr()
    for i in range(len(keep_features)):
        for j in range(i + 1, len(keep_features)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.85:
                f1, f2 = keep_features[i], keep_features[j]
                flag = "[Near-Redundant (>0.95)]" if abs(r) > 0.95 else "[Coupled Physics]"
                print(f"  {f1:<14} <-> {f2:<14} : r = {r:+.4f}  {flag}")

    print("\n" + "=" * 88)
    print(f"FINAL RECOMMENDED SENSOR_ORDER ({len(keep_features)} features):")
    print(keep_features)
    print("=" * 88)

if __name__ == "__main__":
    run_eda()