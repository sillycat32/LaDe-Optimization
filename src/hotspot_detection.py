from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
COORDINATE_CACHE = ROOT_DIR / "outputs" / "geo" / "region_coordinates.csv"
COORDINATE_COLUMNS = ["city", "region_id", "longitude", "latitude"]
COORDINATE_FILTER_VERSION = 2
DEFAULT_SCORE_WEIGHTS = {
    "predicted_demand": 0.50,
    "high_demand_probability": 0.30,
    "neighbor_support": 0.20,
}


def _valid_global_coordinates(frame):
    return (
        frame["lng"].notna()
        & frame["lat"].notna()
        & np.isfinite(frame["lng"])
        & np.isfinite(frame["lat"])
        & frame["lng"].between(-180, 180)
        & frame["lat"].between(-90, 90)
    )


def derive_region_coordinates(root_dir=ROOT_DIR):
    """Use robust per-city filtering and median coordinates; demand rows are untouched."""
    root = Path(root_dir)
    files = sorted(root.glob("delivery_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw delivery parquet files found under {root}")

    coordinate_frames = []
    for path in files:
        raw = pd.read_parquet(
            path,
            columns=["city", "region_id", "lng", "lat"],
        )
        raw["city"] = raw["city"].astype(str)
        raw["region_id"] = raw["region_id"].astype(str)
        raw = raw.loc[_valid_global_coordinates(raw), ["city", "region_id", "lng", "lat"]]
        if raw.empty:
            continue

        for city, city_rows in raw.groupby("city", sort=False):
            city_coordinates = city_rows[["lng", "lat"]]
            center = city_coordinates.median()
            mad = (city_coordinates - center).abs().median()
            # City regions can be far from the urban center. Keep the broad
            # city envelope and let regional medians reject local GPS outliers.
            tolerance = np.maximum(6.0 * 1.4826 * mad.to_numpy(), [3.0, 3.0])
            keep = (
                (city_coordinates["lng"] - center["lng"]).abs() <= tolerance[0]
            ) & (
                (city_coordinates["lat"] - center["lat"]).abs() <= tolerance[1]
            )
            clean = city_rows.loc[keep]
            if clean.empty:
                raise ValueError(f"Coordinate filtering removed all points for {city}.")
            grouped = (
                clean.groupby(["city", "region_id"], as_index=False)
                .agg(
                    longitude=("lng", "median"),
                    latitude=("lat", "median"),
                    valid_coordinate_records=("lng", "size"),
                )
            )
            coordinate_frames.append(grouped)
        del raw

    if not coordinate_frames:
        raise ValueError("No valid geographic coordinates were found.")
    coordinates = pd.concat(coordinate_frames, ignore_index=True)
    if coordinates.duplicated(["city", "region_id"]).any():
        raise ValueError("Coordinate aggregation produced duplicate city-region keys.")
    coordinates["coordinate_filter_version"] = COORDINATE_FILTER_VERSION
    return coordinates.sort_values(["city", "region_id"]).reset_index(drop=True)


def load_region_coordinates(root_dir=ROOT_DIR, force_refresh=False):
    root = Path(root_dir)
    cache = root / "outputs" / "geo" / "region_coordinates.csv"
    refresh = force_refresh or not cache.is_file()
    if not refresh:
        cached = pd.read_csv(cache, dtype={"city": str, "region_id": str})
        refresh = (
            "coordinate_filter_version" not in cached
            or not (
                cached["coordinate_filter_version"]
                == COORDINATE_FILTER_VERSION
            ).all()
        )
    if refresh:
        coordinates = derive_region_coordinates(root)
        cache.parent.mkdir(parents=True, exist_ok=True)
        coordinates.to_csv(cache, index=False)
    coordinates = pd.read_csv(cache, dtype={"city": str, "region_id": str})
    missing = sorted(set(COORDINATE_COLUMNS) - set(coordinates.columns))
    if missing:
        raise ValueError(f"Coordinate cache is missing columns: {missing}")
    valid = (
        coordinates["longitude"].between(-180, 180)
        & coordinates["latitude"].between(-90, 90)
        & np.isfinite(coordinates["longitude"])
        & np.isfinite(coordinates["latitude"])
    )
    if not valid.all():
        raise ValueError("Coordinate cache contains invalid locations.")
    return coordinates


def haversine_km(longitude_a, latitude_a, longitude_b, latitude_b):
    earth_radius_km = 6371.0088
    lon_a, lat_a, lon_b, lat_b = np.radians(
        [longitude_a, latitude_a, longitude_b, latitude_b]
    )
    delta_lon = lon_b - lon_a
    delta_lat = lat_b - lat_a
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(delta_lon / 2) ** 2
    )
    return float(2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(value, 0, 1))))


def build_spatial_graph(coordinates, neighbors=5):
    if neighbors < 1:
        raise ValueError("The number of spatial neighbors must be positive.")
    graph = nx.Graph()
    by_city = {}
    for row in coordinates.itertuples(index=False):
        key = (str(row.city), str(row.region_id))
        graph.add_node(
            key,
            city=key[0],
            region_id=key[1],
            longitude=float(row.longitude),
            latitude=float(row.latitude),
        )
        by_city.setdefault(key[0], []).append(key)

    for city, nodes in by_city.items():
        for node in nodes:
            attrs = graph.nodes[node]
            distances = []
            for other in nodes:
                if other == node:
                    continue
                other_attrs = graph.nodes[other]
                distance = haversine_km(
                    attrs["longitude"],
                    attrs["latitude"],
                    other_attrs["longitude"],
                    other_attrs["latitude"],
                )
                distances.append((distance, other))
            for distance, other in sorted(distances)[: min(neighbors, len(nodes) - 1)]:
                if node[0] != other[0]:
                    raise ValueError("Spatial graph attempted a cross-city connection.")
                graph.add_edge(node, other, distance_km=distance)

    components = list(nx.connected_components(graph))
    summary = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "isolated_node_count": len(list(nx.isolates(graph))),
        "component_count": len(components),
        "components": [
            {
                "city": city,
                "component_count": sum(
                    1 for component in components if component and next(iter(component))[0] == city
                ),
                "region_count": len(nodes),
            }
            for city, nodes in sorted(by_city.items())
        ],
    }
    return graph, summary


def _robust_unit_scale(values):
    values = pd.to_numeric(values, errors="coerce").astype(float)
    low = values.quantile(0.05)
    high = values.quantile(0.95)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return ((values - low) / (high - low)).clip(0, 1)


def score_hotspots(
    forecasts,
    coordinates,
    graph,
    score_weights=None,
    elevated_threshold=0.40,
    hotspot_threshold=0.68,
    neighbor_support_threshold=0.45,
):
    if not 0 <= elevated_threshold < hotspot_threshold <= 1:
        raise ValueError("Hotspot thresholds must satisfy 0 <= elevated < hotspot <= 1.")
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if score_weights:
        weights.update(score_weights)
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Hotspot score weights must be nonnegative with positive sum.")
    total_weight = float(sum(weights.values()))
    weights = {key: value / total_weight for key, value in weights.items()}

    result = forecasts.merge(
        coordinates[COORDINATE_COLUMNS],
        on=["city", "region_id"],
        how="left",
        validate="many_to_one",
    )
    if result[["longitude", "latitude"]].isna().any().any():
        missing = result.loc[
            result["longitude"].isna() | result["latitude"].isna(),
            ["city", "region_id"],
        ].drop_duplicates()
        raise ValueError(
            "No valid region coordinates for: "
            + ", ".join(
                f"{row.city}/{row.region_id}"
                for row in missing.itertuples(index=False)
            )
        )

    neighbor_demand = []
    neighbor_probability = []
    for row in result.itertuples(index=False):
        node = (str(row.city), str(row.region_id))
        if node not in graph:
            raise ValueError(f"Region {node} is missing from the spatial graph.")
        adjacent = list(graph.neighbors(node))
        by_key = result.loc[
            (result["city"] == node[0])
            & (result["timestamp"] == row.timestamp)
        ].set_index("region_id")
        neighbor_rows = [
            by_key.loc[neighbor[1]]
            for neighbor in adjacent
            if neighbor[1] in by_key.index
        ]
        neighbor_demand.append(
            float(np.mean([item["predicted_demand"] for item in neighbor_rows]))
            if neighbor_rows
            else 0.0
        )
        neighbor_probability.append(
            float(
                np.mean(
                    [item["high_demand_probability"] for item in neighbor_rows]
                )
            )
            if neighbor_rows
            else 0.0
        )

    result["neighbor_predicted_demand"] = neighbor_demand
    result["neighbor_high_demand_probability"] = neighbor_probability
    scale_groups = ["city", "timestamp"]
    for source, output in [
        ("predicted_demand", "normalized_predicted_demand"),
        ("neighbor_predicted_demand", "normalized_neighbor_demand"),
    ]:
        result[output] = result.groupby(scale_groups, sort=False)[source].transform(
            _robust_unit_scale
        )
    result["normalized_high_demand_probability"] = (
        result["high_demand_probability"].astype(float).clip(0, 1)
    )
    result["normalized_neighbor_probability"] = (
        result["neighbor_high_demand_probability"].astype(float).clip(0, 1)
    )
    result["neighbor_support"] = (
        0.60 * result["normalized_neighbor_demand"]
        + 0.40 * result["normalized_neighbor_probability"]
    )
    result["hotspot_score"] = (
        weights["predicted_demand"] * result["normalized_predicted_demand"]
        + weights["high_demand_probability"]
        * result["normalized_high_demand_probability"]
        + weights["neighbor_support"] * result["neighbor_support"]
    ).clip(0, 1)

    result["hotspot_classification"] = "Normal"
    elevated = result["hotspot_score"] >= elevated_threshold
    result.loc[elevated, "hotspot_classification"] = "Elevated demand"
    hotspot = (
        (result["hotspot_score"] >= hotspot_threshold)
        & (result["neighbor_support"] >= neighbor_support_threshold)
    )
    result.loc[hotspot, "hotspot_classification"] = "High-demand hotspot"
    result["is_hotspot"] = (
        result["hotspot_classification"] == "High-demand hotspot"
    )
    result.attrs["score_weights"] = weights
    return result


def save_hotspots(hotspots, root_dir=ROOT_DIR):
    output = Path(root_dir) / "outputs" / "hotspots"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "current_hotspots.csv"
    hotspots.to_csv(path, index=False)
    return path
