from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from .fleet_optimizer import allocation_metrics


ROOT_DIR = Path(__file__).resolve().parents[1]


def forecast_metrics(predictions):
    valid = predictions.loc[
        predictions["actual_demand"].notna()
    ]
    if valid.empty:
        return {
            "observations": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE (%)": np.nan,
        }
    actual = valid["actual_demand"].to_numpy(dtype=float)
    predicted = valid["predicted_demand"].to_numpy(dtype=float)
    total = float(np.sum(np.abs(actual)))
    return {
        "observations": len(valid),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE (%)": (
            float(np.sum(np.abs(actual - predicted)) / total * 100)
            if total
            else np.nan
        ),
    }


def hotspot_metrics(hotspots, high_demand_threshold=48.0):
    valid = hotspots.loc[hotspots["actual_demand"].notna()]
    if valid.empty:
        return {
            "observations": 0,
            "precision": np.nan,
            "recall": np.nan,
            "F1": np.nan,
            "label_note": "Historical actual demand is unavailable.",
        }
    actual_hotspot = valid["actual_demand"].to_numpy() >= high_demand_threshold
    predicted_hotspot = valid["is_hotspot"].to_numpy(dtype=bool)
    return {
        "observations": len(valid),
        "precision": float(
            precision_score(actual_hotspot, predicted_hotspot, zero_division=0)
        ),
        "recall": float(
            recall_score(actual_hotspot, predicted_hotspot, zero_division=0)
        ),
        "F1": float(f1_score(actual_hotspot, predicted_hotspot, zero_division=0)),
        "label_note": (
            "Proxy evaluation: identified spatial hotspots vs. same-hour actual "
            "demand >=48; not future hotspot-emergence validation."
        ),
    }


def fleet_comparison(allocations, vehicle_capacity, actual_demand=None):
    rows = []
    for strategy, allocation in allocations.items():
        evaluation_demand = None
        if actual_demand is not None:
            evaluation_demand = actual_demand[strategy]
        metrics = allocation_metrics(allocation, evaluation_demand)
        metrics["vehicle_capacity"] = int(vehicle_capacity)
        rows.append(metrics)
    return pd.DataFrame(rows)


def route_comparison(route_summaries):
    rows = []
    for strategy, summary in route_summaries.items():
        rows.append(
            {
                "Strategy": strategy,
                "Routes": summary["route_count"],
                "Total route distance (km)": summary["total_travel_distance_km"],
                "Average route distance (km)": summary["average_route_distance_km"],
                "Average estimated travel time (min)": summary[
                    "average_estimated_route_minutes"
                ],
                "Demand coverage (%)": summary["demand_coverage_percent"],
                "Unserved demand": summary["unserved_demand"],
                "Distance basis": summary["distance_method"],
            }
        )
    return pd.DataFrame(rows)


def load_existing_model_performance(root_dir=ROOT_DIR):
    root = Path(root_dir)
    path = root / "models" / "calibration_experiment" / "model_comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Validated model comparison file not found: {path}")
    table = pd.read_csv(path)
    required = {"Period", "Model", "MAE", "RMSE", "WAPE (%)"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Saved model metrics are missing columns: {missing}")
    return table
