import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hotspot_detection import haversine_km


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CRUISE_SPEED_KMH = 20.0


def optimize_routes(
    demand_frame,
    allocation,
    vehicle_capacity,
    cruise_speed_kmh=DEFAULT_CRUISE_SPEED_KMH,
):
    """Build capacity-feasible nearest-neighbor routes using straight-line distance."""
    if vehicle_capacity <= 0 or cruise_speed_kmh <= 0:
        raise ValueError("Vehicle capacity and cruise speed must be positive.")
    required_demand = {
        "city",
        "region_id",
        "longitude",
        "latitude",
        "predicted_demand",
    }
    required_allocation = {
        "city",
        "region_id",
        "vehicles_allocated",
    }
    if missing := sorted(required_demand - set(demand_frame.columns)):
        raise ValueError(f"Route demand is missing fields: {missing}")
    if missing := sorted(required_allocation - set(allocation.columns)):
        raise ValueError(f"Fleet allocation is missing fields: {missing}")

    demand = demand_frame[
        ["city", "region_id", "longitude", "latitude", "predicted_demand"]
    ].copy()
    demand["scenario_demand"] = demand["predicted_demand"].astype(float)
    demand["scenario_demand"] = demand["scenario_demand"].clip(lower=0)

    merged = demand.merge(
        allocation[["city", "region_id", "vehicles_allocated"]],
        on=["city", "region_id"],
        how="left",
        validate="one_to_one",
    )
    merged["vehicles_allocated"] = merged["vehicles_allocated"].fillna(0).astype(int)
    if merged["vehicles_allocated"].sum() > allocation["vehicles_allocated"].sum():
        raise RuntimeError("Route fleet exceeds the fleet allocation.")

    routes = []
    route_summaries = []
    unserved_by_region = {}
    route_number = 0
    for city, city_regions in merged.groupby("city", sort=True):
        city_regions = city_regions.sort_values("region_id").reset_index(drop=True)
        vehicles = []
        for row in city_regions.itertuples(index=False):
            for _ in range(int(row.vehicles_allocated)):
                route_number += 1
                vehicles.append(
                    {
                        "route_id": f"{city}-{route_number:04d}",
                        "home_region_id": str(row.region_id),
                        "home": (float(row.longitude), float(row.latitude)),
                        "remaining_capacity": float(vehicle_capacity),
                        "stops": {},
                        "current": (float(row.longitude), float(row.latitude)),
                    }
                )

        tasks = city_regions.loc[city_regions["scenario_demand"] > 0].copy()
        tasks = tasks.sort_values("scenario_demand", ascending=False)
        for task in tasks.itertuples(index=False):
            remaining = float(task.scenario_demand)
            target = (float(task.longitude), float(task.latitude))
            while remaining > 1e-8:
                candidates = [
                    vehicle
                    for vehicle in vehicles
                    if vehicle["remaining_capacity"] > 1e-8
                ]
                if not candidates:
                    unserved_by_region[(city, str(task.region_id))] = remaining
                    break
                vehicle = min(
                    candidates,
                    key=lambda item: haversine_km(
                        item["current"][0],
                        item["current"][1],
                        target[0],
                        target[1],
                    ),
                )
                amount = min(remaining, vehicle["remaining_capacity"])
                stop_key = str(task.region_id)
                vehicle["stops"][stop_key] = (
                    vehicle["stops"].get(stop_key, 0.0) + amount
                )
                vehicle["remaining_capacity"] -= amount
                vehicle["current"] = target
                remaining -= amount

        for vehicle in vehicles:
            unvisited = dict(vehicle["stops"])
            ordered_stops = []
            current = vehicle["home"]
            while unvisited:
                next_region = min(
                    unvisited,
                    key=lambda region_id: haversine_km(
                        current[0],
                        current[1],
                        float(
                            city_regions.loc[
                                city_regions["region_id"].astype(str) == region_id,
                                "longitude",
                            ].iloc[0]
                        ),
                        float(
                            city_regions.loc[
                                city_regions["region_id"].astype(str) == region_id,
                                "latitude",
                            ].iloc[0]
                        ),
                    ),
                )
                stop = city_regions.loc[
                    city_regions["region_id"].astype(str) == next_region
                ].iloc[0]
                coordinate = (float(stop["longitude"]), float(stop["latitude"]))
                ordered_stops.append(
                    {
                        "region_id": next_region,
                        "longitude": coordinate[0],
                        "latitude": coordinate[1],
                        "planned_delivery": float(unvisited.pop(next_region)),
                    }
                )
                current = coordinate

            path = [vehicle["home"]]
            path.extend(
                (stop["longitude"], stop["latitude"]) for stop in ordered_stops
            )
            path.append(vehicle["home"])
            distance_km = sum(
                haversine_km(a[0], a[1], b[0], b[1])
                for a, b in zip(path, path[1:])
            )
            load = float(sum(vehicle["stops"].values()))
            minutes = distance_km / cruise_speed_kmh * 60
            routes.append(
                {
                    "city": city,
                    "route_id": vehicle["route_id"],
                    "home_region_id": vehicle["home_region_id"],
                    "path": [[lon, lat] for lon, lat in path],
                    "stops": ordered_stops,
                    "planned_load": load,
                    "vehicle_capacity": int(vehicle_capacity),
                    "distance_km": distance_km,
                    "estimated_travel_minutes": minutes,
                    "distance_method": "Haversine straight-line simulation",
                }
            )
            route_summaries.append(
                {
                    "city": city,
                    "route_id": vehicle["route_id"],
                    "home_region_id": vehicle["home_region_id"],
                    "stop_count": len(ordered_stops),
                    "planned_load": load,
                    "vehicle_capacity": int(vehicle_capacity),
                    "distance_km": distance_km,
                    "estimated_travel_minutes": minutes,
                    "distance_method": "Haversine straight-line simulation",
                }
            )

    total_demand = float(merged["scenario_demand"].sum())
    served = float(sum(route["planned_load"] for route in routes))
    total_distance = float(sum(route["distance_km"] for route in routes))
    total_minutes = float(sum(route["estimated_travel_minutes"] for route in routes))
    summary = {
        "route_count": len(routes),
        "total_demand": total_demand,
        "planned_served_demand": served,
        "unserved_demand": float(max(total_demand - served, 0)),
        "demand_coverage_percent": (
            float(served / total_demand * 100) if total_demand else 100.0
        ),
        "total_travel_distance_km": total_distance,
        "average_route_distance_km": (
            total_distance / len(routes) if routes else 0.0
        ),
        "average_estimated_route_minutes": (
            total_minutes / len(routes) if routes else 0.0
        ),
        "cruise_speed_kmh_simulation_assumption": float(cruise_speed_kmh),
        "distance_method": "Haversine straight-line simulation",
        "unserved_by_region": [
            {
                "city": city,
                "region_id": region_id,
                "unserved_demand": quantity,
            }
            for (city, region_id), quantity in sorted(unserved_by_region.items())
        ],
    }
    if any(route["planned_load"] > vehicle_capacity + 1e-8 for route in routes):
        raise RuntimeError("A route exceeds its vehicle capacity.")
    return routes, pd.DataFrame(route_summaries), summary


def save_routes(routes, root_dir=ROOT_DIR):
    output = Path(root_dir) / "outputs" / "routes"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "current_routes.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(routes, file, indent=2)
    return path
