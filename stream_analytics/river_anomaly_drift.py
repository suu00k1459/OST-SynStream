
"""
Online anomaly detection for SynStream using River.

Input:  Prepared Air Quality CSV
Output: CSV containing anomaly scores and flags
"""

import argparse
from pathlib import Path

import pandas as pd
from river import anomaly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Online anomaly detection with River HalfSpaceTrees"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the prepared Air Quality CSV",
    )
    parser.add_argument(
        "--output",
        default="results/airquality_anomalies.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional sensor column names",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Anomaly score threshold (tune for your data)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=250,
        help="HalfSpaceTrees reference window size",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path.resolve()}"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    # Preserve timestamps and other non-feature columns.
    timestamp_candidates = [
        c for c in df.columns
        if c.strip().lower() in {"timestamp", "datetime", "date", "time"}
    ]
    timestamp_col = timestamp_candidates[0] if timestamp_candidates else None

    excluded = set(timestamp_candidates)
    excluded.update(
        c for c in df.columns
        if c.strip().lower() in {"label", "target", "is_anomaly", "anomaly"}
    )

    if args.features:
        missing = set(args.features) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {sorted(missing)}")
        feature_cols = args.features
    else:
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in excluded
        ]

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found. Check your CSV and --features."
        )

    # Convert invalid numeric values to missing values.
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.replace([float("inf"), float("-inf")], float("nan"))

    # Remove rows with missing sensor values to keep the online model simple.
    valid_mask = features.notna().all(axis=1)
    skipped_rows = int((~valid_mask).sum())

    df = df.loc[valid_mask].copy()
    features = features.loc[valid_mask].copy()

    if features.empty:
        raise ValueError("No complete numeric observations remain.")

    # HalfSpaceTrees works best when features have known bounded ranges.
    # Min-max scaling uses the dataset's range for this offline test.
    # For a real streaming deployment, fit scaling on training/reference
    # data only and reuse those bounds for incoming observations.
    mins = features.min()
    ranges = (features.max() - mins).replace(0, 1)
    scaled = (features - mins) / ranges

    # HalfSpaceTrees assumes bounded input features, normally in [0, 1].
    model = anomaly.HalfSpaceTrees(
        n_trees=25,
        height=8,
        window_size=args.window_size,
        seed=42,
    )

    scores = []
    flags = []

    for row in scaled.to_dict(orient="records"):
        # Score first, before updating the model with this observation.
        score = model.score_one(row)
        is_anomaly = score >= args.threshold

        scores.append(score)
        flags.append(is_anomaly)

        # Update the model incrementally.
        # Note: anomalous observations are also learned here.
        model.learn_one(row)

    df["anomaly_score"] = scores
    df["is_anomaly"] = flags

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Input rows: {len(valid_mask)}")
    print(f"Skipped rows with missing/invalid features: {skipped_rows}")
    print(f"Features: {feature_cols}")
    print(f"Anomalies detected: {sum(flags)}")
    print(f"Anomaly rate: {sum(flags) / len(flags):.2%}")
    print(f"Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
