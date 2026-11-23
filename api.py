import json
import math
import numpy as np
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.forecasting import (
    DEFAULT_PROBABILITY_THRESHOLD,
    forecast_horizon,
    load_feature_data,
    load_models,
)
from src.hotspot_detection import (
    build_spatial_graph,
    load_region_coordinates,
    score_hotspots,
)
from src.fleet_optimizer import STRATEGIES, compare_strategies
from src.route_optimizer import optimize_routes
from src.evaluation import (
    forecast_metrics,
    hotspot_metrics,
    fleet_comparison,
    route_comparison,
    load_existing_model_performance,
)

ROOT = Path(__file__).resolve().parent

def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value

app = FastAPI(
    title="LaDe Delivery Demand API",
    version="1.0.0",
    description="Demand forecasting, hotspot detection, fleet and route optimization",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_resources():
    features = load_feature_data(
        ROOT / "processed" / "lade_hourly_features.parquet"
    )
    models = load_models(ROOT)
    coordinates = load_region_coordinates(ROOT)
    graph, graph_summary = build_spatial_graph(
        coordinates, neighbors=5
    )

    return features, models, coordinates, graph, graph_summary


def records(frame):
    return frame.to_dict(orient="records")

def get_snapshot(
    as_of=None,
    horizon=1,
    planning_hour=1,
    fleet_size=40,
    vehicle_capacity=20,
    hotspot_bonus=0.5,
    probability_threshold=DEFAULT_PROBABILITY_THRESHOLD,
):
    try:
        features, models, coordinates, graph, graph_summary = get_resources()

        anchor = (
            pd.Timestamp(as_of)
            if as_of
            else pd.Timestamp(features["timestamp"].max())
        )

        forecasts = forecast_horizon(
            features,
            models,
            as_of_timestamp=anchor,
            horizon=horizon,
            probability_threshold=probability_threshold,
        )

        hotspots = score_hotspots(forecasts, coordinates, graph)

        selected = hotspots.loc[
            hotspots["forecast_hour"] == planning_hour
        ].copy()

        if selected.empty:
            raise ValueError("No forecast rows for the selected planning hour.")

        allocations = compare_strategies(
            selected,
            fleet_size,
            vehicle_capacity,
            hotspot_bonus=hotspot_bonus,
        )

        route_results = {}
        route_summaries = {}

        for strategy in STRATEGIES:
            routes, route_rows, summary = optimize_routes(
                selected,
                allocations[strategy],
                vehicle_capacity,
            )
            route_results[strategy] = {
                "routes": routes,
                "rows": route_rows,
                "summary": summary,
            }
            route_summaries[strategy] = summary

        fleet_table = fleet_comparison(
            allocations,
            vehicle_capacity,
            actual_demand={
                strategy: selected["actual_demand"].to_numpy(dtype=float)
                for strategy in STRATEGIES
            },
        )

        return {
            "anchor": anchor,
            "target": anchor + pd.Timedelta(hours=planning_hour),
            "forecasts": forecasts,
            "hotspots": selected,
            "all_hotspots": hotspots,
            "allocations": allocations,
            "routes": route_results,
            "fleet_table": fleet_table,
            "route_table": route_comparison(route_summaries),
            "forecast_metrics": forecast_metrics(
                forecasts.loc[forecasts["forecast_hour"] == planning_hour]
            ),
            "hotspot_metrics": hotspot_metrics(selected),
            "graph_summary": graph_summary,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def home():
    return {
        "message": "LaDe Delivery Demand API is running",
        "docs": "/docs",
        "status": "online",
    }



@app.get("/api/overview")
def overview():
    data = get_snapshot()
    selected = data["hotspots"]

    return clean_json({
        "cities": int(selected["city"].nunique()),
        "regions": int(
            selected[["city", "region_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "predicted_demand": float(
            selected["predicted_demand"].sum()
        ),
        "hotspots": int(selected["is_hotspot"].sum()),
        "forecast_target": data["target"].isoformat(),
        "fleet_size": int(
            data["fleet_table"]["Vehicles allocated"].max()
        ),
        "graph": data["graph_summary"],
    })




@app.get("/api/model-performance")
def model_performance():
    try:
        table = load_existing_model_performance(ROOT)
        return {"data": records(table)}
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@app.get("/api/health")
def health():
    return {"status": "healthy"}

# ---------------------------------------------------------
# ADDITIONAL API ENDPOINTS
# ---------------------------------------------------------

@app.get("/api/forecast")
def api_forecast(
    horizon: int = 6,
    planning_hour: int = 1,
    city: str = "All cities",
):
    try:
        data = get_snapshot(
            horizon=horizon,
            planning_hour=planning_hour,
        )

        forecasts = data["forecasts"].copy()

        if city != "All cities":
            forecasts = forecasts[forecasts["city"] == city]

        return clean_json({
            "data": records(forecasts),
            "forecasts": records(forecasts),
            "metrics": data["forecast_metrics"],
            "target": data["target"],
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/hotspots")
def api_hotspots(city: str = "All cities"):
    try:
        data = get_snapshot()
        hotspots = data["hotspots"].copy()

        if city != "All cities":
            hotspots = hotspots[hotspots["city"] == city]

        return clean_json(records(hotspots))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/demand")
def api_demand(city: str = "All cities"):
    try:
        data = get_snapshot()
        forecasts = data["forecasts"].copy()

        if city != "All cities":
            forecasts = forecasts[forecasts["city"] == city]

        demand = (
            forecasts.groupby("timestamp", as_index=False)
            .agg(
                predicted_demand=("predicted_demand", "sum"),
                actual_demand=("actual_demand", "sum"),
            )
            .sort_values("timestamp")
        )

        return clean_json({
            "data": records(demand),
            "city": city,
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/fleet")
def api_fleet(
    fleet_size: int = 40,
    vehicle_capacity: int = 20,
):
    try:
        data = get_snapshot(
            fleet_size=fleet_size,
            vehicle_capacity=vehicle_capacity,
        )

        strategies = {}

        for name, allocation in data["allocations"].items():
            strategies[name] = (
                records(allocation)
                if isinstance(allocation, pd.DataFrame)
                else allocation
            )

        return clean_json({
            "strategies": strategies,
            "comparison": records(data["fleet_table"]),
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/routes")
def api_routes(
    strategy: str = "Proactive",
    fleet_size: int = 40,
    vehicle_capacity: int = 20,
):
    try:
        data = get_snapshot(
            fleet_size=fleet_size,
            vehicle_capacity=vehicle_capacity,
        )

        if strategy not in data["routes"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {strategy}. "
                       f"Choose from {list(data['routes'].keys())}",
            )

        route_data = data["routes"][strategy]

        return clean_json({
            "strategy": strategy,
            "strategies": {
                name: result["summary"]
                for name, result in data["routes"].items()
            },
            "routes": route_data["routes"],
            "rows": records(route_data["rows"])
                if isinstance(route_data["rows"], pd.DataFrame)
                else route_data["rows"],
            "summary": route_data["summary"],
            "comparison": records(data["route_table"]),
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
