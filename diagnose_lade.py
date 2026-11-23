
import pandas as pd
from pathlib import Path

for file in Path(".").glob("delivery_*.parquet"):
    df = pd.read_parquet(file, columns=["city", "accept_time", "ds"])

    timestamps = pd.to_datetime(
        "2022-" + df["accept_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    print("\n", file.name)
    print("Timestamp range:", timestamps.min(), "to", timestamps.max())
    print("Unique timestamp dates:", timestamps.dt.date.nunique())
    print("ds range:", df["ds"].min(), "to", df["ds"].max())
    print("Invalid timestamps:", timestamps.isna().sum())

    print("\nEarliest dates:")
    print(timestamps.dt.date.value_counts().sort_index().head())

    print("\nLatest dates:")
    print(timestamps.dt.date.value_counts().sort_index().tail())
