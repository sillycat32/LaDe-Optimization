
from pathlib import Path
import pandas as pd
import numpy as np
import glob

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = sorted(
    glob.glob(str(BASE_DIR / "delivery_*.parquet"))
)

YEAR = 2022

# Documented dataset collection period
START_DATE = pd.Timestamp("2022-05-01 00:00:00")
END_DATE = pd.Timestamp("2022-10-31 23:00:00")

REQUIRED_COLUMNS = [
    "city",
    "region_id",
    "accept_time"
]

if not FILES:
    raise FileNotFoundError(
        "No delivery Parquet files found in the dataset directory."
    )

# ==========================================================
# STEP 1: LOAD AND AGGREGATE RAW DELIVERY RECORDS
# ==========================================================

city_hourly_data = []

print("=" * 70)
print("LADE DATASET PREPROCESSING")
print("=" * 70)

for file in FILES:

    print("\n" + "=" * 65)
    print("Processing:", Path(file).name)

    df = pd.read_parquet(
        file,
        columns=REQUIRED_COLUMNS
    )

    print("Raw records:", len(df))

    # ------------------------------------------------------
    # Reconstruct timestamps.
    # accept_time contains month-day-time without a year.
    # Dataset collection year is assumed to be 2022.
    # ------------------------------------------------------

    timestamp_strings = (
        str(YEAR) + "-" + df["accept_time"].astype(str)
    )

    df["timestamp"] = pd.to_datetime(
        timestamp_strings,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    invalid = df["timestamp"].isna().sum()

    print("Invalid timestamps:", invalid)

    # Remove invalid timestamps and missing region identifiers.
    df = df.dropna(
        subset=["timestamp", "region_id", "city"]
    ).copy()

    # ------------------------------------------------------
    # Restrict data to the documented collection period.
    # This removes the two anomalous Chongqing records.
    # ------------------------------------------------------

    valid_period = (
        (df["timestamp"] >= START_DATE) &
        (df["timestamp"] <= END_DATE)
    )

    excluded = (~valid_period).sum()

    if excluded:
        print(
            "Records outside collection period:",
            excluded
        )

    df = df.loc[valid_period].copy()

    # Floor timestamps to hourly intervals.
    df["timestamp"] = df["timestamp"].dt.floor("h")

    # Keep identifiers separate between cities.
    df["city"] = df["city"].astype(str)
    df["region_id"] = df["region_id"].astype(str)

    # ------------------------------------------------------
    # Aggregate orders by city, region and hour.
    # ------------------------------------------------------

    hourly = (
        df.groupby(
            ["city", "region_id", "timestamp"],
            observed=True
        )
        .size()
        .reset_index(name="demand")
    )

    print("Aggregated hourly observations:", len(hourly))

    if not hourly.empty:
        print("City:", hourly["city"].iloc[0])
        print("Regions:", hourly["region_id"].nunique())

    city_hourly_data.append(hourly)

    del df

# ==========================================================
# STEP 2: COMBINE CITIES
# ==========================================================

hourly_data = pd.concat(
    city_hourly_data,
    ignore_index=True
)

hourly_data = hourly_data.sort_values(
    ["city", "region_id", "timestamp"]
).reset_index(drop=True)

print("\n" + "=" * 65)
print("COMBINED HOURLY DATA")
print("=" * 65)

print("Combined hourly observations:", len(hourly_data))

print("\nObserved records by city:")
print(
    hourly_data.groupby("city").size()
)

# Save observed, non-zero hourly aggregation.
hourly_data.to_parquet(
    OUTPUT_DIR / "hourly_demand_observed.parquet",
    index=False
)

# ==========================================================
# STEP 3: CREATE CONTINUOUS REGION-HOUR SERIES
# ==========================================================

print("\n" + "=" * 65)
print("CREATING CONTINUOUS REGION-HOUR SERIES")
print("=" * 65)

continuous_data = []

# Same 184-day hourly timeline for every city.
timestamps = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="h"
)

for city in hourly_data["city"].unique():

    city_df = hourly_data[
        hourly_data["city"] == city
    ].copy()

    # Remove observations outside the collection period.
    valid_period = (
        (city_df["timestamp"] >= START_DATE) &
        (city_df["timestamp"] <= END_DATE)
    )

    excluded = (~valid_period).sum()

    if excluded:
        print(
            f"{city}: Excluding {excluded} hourly observations"
        )

    city_df = city_df.loc[valid_period].copy()

    regions = sorted(
        city_df["region_id"].unique()
    )

    # Every region × every hour.
    full_index = pd.MultiIndex.from_product(
        [regions, timestamps],
        names=["region_id", "timestamp"]
    )

    city_df = city_df.set_index(
        ["region_id", "timestamp"]
    )

    city_df = city_df.reindex(full_index)

    # Missing region-hour combinations mean zero recorded orders.
    city_df["demand"] = city_df["demand"].fillna(0)

    city_df = city_df.reset_index()

    city_df["city"] = city

    city_df["demand"] = city_df["demand"].astype("int32")

    continuous_data.append(city_df)

    print(
        f"{city}: "
        f"{len(regions)} regions, "
        f"{len(timestamps)} hours, "
        f"{len(city_df)} rows"
    )

# Combine all cities.
data = pd.concat(
    continuous_data,
    ignore_index=True
)

data = data.sort_values(
    ["city", "region_id", "timestamp"]
).reset_index(drop=True)

print("\nTotal continuous observations:", len(data))

# ==========================================================
# STEP 4: CALENDAR FEATURES
# ==========================================================

print("\nGenerating calendar features...")

data["hour"] = data["timestamp"].dt.hour

data["day_of_week"] = (
    data["timestamp"].dt.day_of_week
)

data["day_of_month"] = (
    data["timestamp"].dt.day
)

data["month"] = (
    data["timestamp"].dt.month
)

data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype("int8")

# Cyclical encoding of hour.
data["hour_sin"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["hour_cos"] = np.cos(
    2 * np.pi * data["hour"] / 24
)

# Cyclical encoding of weekday.
data["dow_sin"] = np.sin(
    2 * np.pi * data["day_of_week"] / 7
)

data["dow_cos"] = np.cos(
    2 * np.pi * data["day_of_week"] / 7
)

# ==========================================================
# STEP 5: LAG FEATURES
# ==========================================================

print("Generating lag features...")

group_columns = ["city", "region_id"]

grouped = data.groupby(
    group_columns,
    sort=False
)["demand"]

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]

for lag in LAGS:

    data[f"lag_{lag}"] = grouped.shift(lag)

# ==========================================================
# STEP 6: ROLLING FEATURES
# ==========================================================

print("Generating rolling features...")

# Shift demand by one hour before calculating rolling means.
# This prevents current-hour target leakage.

WINDOWS = [3, 6, 12, 24, 168]

for window in WINDOWS:

    data[f"rolling_mean_{window}h"] = (
        data.groupby(
            group_columns,
            sort=False
        )["demand"]
        .transform(
            lambda s: (
                s.shift(1)
                .rolling(
                    window=window,
                    min_periods=window
                )
                .mean()
            )
        )
    )

# ==========================================================
# STEP 7: REMOVE ROWS WITHOUT SUFFICIENT HISTORY
# ==========================================================

print("Filtering rows without sufficient history...")

required_history = [
    "lag_168",
    "rolling_mean_168h"
]

before = len(data)

data = data.dropna(
    subset=required_history
).reset_index(drop=True)

removed = before - len(data)

print("\nRows before history filtering:", before)
print("Rows after history filtering:", len(data))
print("Rows removed:", removed)

# ==========================================================
# STEP 8: DATA VALIDATION
# ==========================================================

print("\nRunning validation checks...")

# Check for missing values.
missing_values = data.isna().sum()

print("\nColumns with missing values:")

print(
    missing_values[missing_values > 0]
    if missing_values.sum() > 0
    else "None"
)

# Check for negative demand.
negative_demand = (
    data["demand"] < 0
).sum()

print("\nNegative demand values:", negative_demand)

# Check timestamp range.
print(
    "\nFinal timestamp range:",
    data["timestamp"].min(),
    "to",
    data["timestamp"].max()
)

# Check rows per city.
print("\nFinal records by city:")
print(data.groupby("city").size())

# Check demand distribution.
print("\nDemand statistics:")
print(data["demand"].describe())

# ==========================================================
# STEP 9: SAVE FINAL DATASET
# ==========================================================

print("\nSaving processed datasets...")

data.to_parquet(
    OUTPUT_DIR / "lade_hourly_features.parquet",
    index=False
)

# Save a small CSV sample for inspection.
data.head(1000).to_csv(
    OUTPUT_DIR / "lade_hourly_features_sample.csv",
    index=False
)

# ==========================================================
# STEP 10: FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nFinal dataset shape:", data.shape)

print("\nRecords by city:")
print(data.groupby("city").size())

print("\nDemand statistics:")
print(data["demand"].describe())

print("\nFinal columns:")
print(data.columns.tolist())

print("\nOutput files:")

print(
    "1.",
    OUTPUT_DIR / "hourly_demand_observed.parquet"
)

print(
    "2.",
    OUTPUT_DIR / "lade_hourly_features.parquet"
)

print(
    "3.",
    OUTPUT_DIR / "lade_hourly_features_sample.csv"
)

print("\nDone.")
