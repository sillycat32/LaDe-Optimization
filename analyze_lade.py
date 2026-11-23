
import pandas as pd
import glob
from pathlib import Path

files = glob.glob("*.parquet")
output = Path("analysis_outputs")
output.mkdir(exist_ok=True)

for file in files:
    print("\n" + "=" * 70)
    print("CITY FILE:", file)

    df = pd.read_parquet(
        file,
        columns=[
            "ds", "city", "region_id", "aoi_id",
            "aoi_type", "lng", "lat", "accept_time",
            "delivery_time", "courier_id"
        ]
    )

    city = df["city"].iloc[0]
    print("Total records:", len(df))
    print("Date range:", df["ds"].min(), "to", df["ds"].max())

    # Daily demand
    daily = df.groupby("ds").size().rename("orders")
    print("\nDaily demand statistics:")
    print(daily.describe())

    # Hourly demand
    df["hour"] = pd.to_datetime(
        df["accept_time"], format="%m-%d %H:%M:%S"
    ).dt.hour

    hourly = df.groupby("hour").size().rename("orders")
    print("\nHourly demand:")
    print(hourly.to_string())

    # Spatial demand
    print("\nUnique regions:", df["region_id"].nunique())
    print("Unique AOIs:", df["aoi_id"].nunique())

    print("\nTop 10 regions:")
    print(df["region_id"].value_counts().head(10).to_string())

    print("\nTop 10 AOIs:")
    print(df["aoi_id"].value_counts().head(10).to_string())

    # AOI categories
    print("\nAOI types:")
    print(df["aoi_type"].value_counts().head(15).to_string())

    # Geographic extent
    print("\nCoordinate bounds:")
    print("Longitude:", df["lng"].min(), "to", df["lng"].max())
    print("Latitude:", df["lat"].min(), "to", df["lat"].max())

    # Save hourly and daily summaries
    daily.to_csv(output / f"{city}_daily.csv")
    hourly.to_csv(output / f"{city}_hourly.csv")

print("\nAnalysis complete. Outputs saved in analysis_outputs/")
