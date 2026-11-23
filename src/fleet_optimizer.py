from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
STRATEGIES = ("Static", "Reactive", "Proactive")


def _planning_demand(frame, strategy):
    if strategy == "Proactive":
        return frame["predicted_demand"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    if strategy == "Reactive":
        # Use only demand observed by the forecast's as-of time, never its target.
        return frame["last_observed_demand"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    raise ValueError(f"Unknown demand-based allocation strategy: {strategy}")


def allocate_fleet(
    hotspots,
    fleet_size,
    vehicle_capacity,
    strategy="Proactive",
    hotspot_bonus=0.50,
):
    """Allocate integer vehicles greedily, never exceeding fleet or regional need."""
    if fleet_size < 0:
        raise ValueError("Fleet size cannot be negative.")
    if vehicle_capacity <= 0:
        raise ValueError("Vehicle capacity must be positive.")
    if strategy not in STRATEGIES:
        raise ValueError(f"Strategy must be one of {STRATEGIES}.")
    required = {"city", "region_id", "predicted_demand", "hotspot_score"}
    missing = sorted(required - set(hotspots.columns))
    if missing:
        raise ValueError(f"Hotspot input is missing allocation fields: {missing}")
    if hotspots.duplicated(["city", "region_id"]).any():
        raise ValueError("Fleet allocation expects exactly one forecast per region.")

    result = hotspots.copy().reset_index(drop=True)
    allocation = np.zeros(len(result), dtype=int)
    if strategy == "Static":
        if len(result):
            for vehicle in range(fleet_size):
                allocation[vehicle % len(result)] += 1
        planning_demand = result["predicted_demand"].fillna(0).to_numpy(dtype=float)
    else:
        planning_demand = _planning_demand(result, strategy)
        priority = planning_demand.copy()
        if strategy == "Proactive":
            priority *= 1 + max(0.0, hotspot_bonus) * result[
                "hotspot_score"
            ].fillna(0).clip(0, 1).to_numpy(dtype=float)
        maximum_needed = np.ceil(planning_demand / vehicle_capacity).astype(int)
        for _ in range(fleet_size):
            eligible = np.flatnonzero(allocation < maximum_needed)
            if not len(eligible):
                break
            marginal_value = priority[eligible] / (allocation[eligible] + 1)
            best_index = int(eligible[int(np.argmax(marginal_value))])
            allocation[best_index] += 1

    result["strategy"] = strategy
    result["allocation_demand"] = planning_demand
    result["vehicles_allocated"] = allocation
    result["capacity_allocated"] = allocation * int(vehicle_capacity)
    result["planned_served_demand"] = np.minimum(
        planning_demand,
        result["capacity_allocated"],
    )
    result["planned_unserved_demand"] = np.maximum(
        planning_demand - result["planned_served_demand"],
        0,
    )
    result["total_fleet_size"] = int(fleet_size)
    result["vehicle_capacity"] = int(vehicle_capacity)
    if int(result["vehicles_allocated"].sum()) > fleet_size:
        raise RuntimeError("Fleet allocation exceeded the available vehicle count.")
    return result


def compare_strategies(
    hotspots,
    fleet_size,
    vehicle_capacity,
    hotspot_bonus=0.50,
):
    allocations = {
        strategy: allocate_fleet(
            hotspots,
            fleet_size,
            vehicle_capacity,
            strategy=strategy,
            hotspot_bonus=hotspot_bonus,
        )
        for strategy in STRATEGIES
    }
    return allocations


def allocation_metrics(allocation, evaluation_demand=None):
    demand = (
        allocation["predicted_demand"].to_numpy(dtype=float)
        if evaluation_demand is None
        else np.asarray(evaluation_demand, dtype=float)
    )
    if len(demand) != len(allocation):
        raise ValueError("Evaluation demand does not align with allocations.")
    fallback = allocation["predicted_demand"].to_numpy(dtype=float)
    demand = np.where(np.isfinite(demand), demand, fallback).clip(0)
    capacity = allocation["capacity_allocated"].to_numpy(dtype=float)
    served = np.minimum(demand, capacity)
    total_demand = float(demand.sum())
    total_capacity = float(capacity.sum())
    return {
        "Strategy": str(allocation["strategy"].iloc[0])
        if len(allocation)
        else "Unknown",
        "Fleet size": int(allocation["total_fleet_size"].iloc[0])
        if len(allocation)
        else 0,
        "Vehicles allocated": int(allocation["vehicles_allocated"].sum()),
        "Demand": total_demand,
        "Served demand estimate": float(served.sum()),
        "Unserved demand estimate": float(max(total_demand - served.sum(), 0)),
        "Demand coverage (%)": (
            float(served.sum() / total_demand * 100)
            if total_demand > 0
            else 100.0
        ),
        "Fleet utilization (%)": (
            float(served.sum() / total_capacity * 100)
            if total_capacity > 0
            else 0.0
        ),
    }


def save_allocations(allocations, root_dir=ROOT_DIR):
    output = Path(root_dir) / "outputs" / "fleet"
    output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(allocations.values(), ignore_index=True)
    path = output / "current_allocations.csv"
    combined.to_csv(path, index=False)
    return path
