import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluation import (
    fleet_comparison,
    forecast_metrics,
    hotspot_metrics,
    load_existing_model_performance,
    route_comparison,
)
from src.fleet_optimizer import (
    STRATEGIES,
    compare_strategies,
    save_allocations,
)
from src.forecasting import (
    DEFAULT_PROBABILITY_THRESHOLD,
    forecast_horizon,
    load_feature_data,
    load_models,
    save_predictions,
)
from src.hotspot_detection import (
    build_spatial_graph,
    load_region_coordinates,
    score_hotspots,
    save_hotspots,
)
from src.route_optimizer import optimize_routes, save_routes


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
MIN_HISTORICAL_DATE = date(2022, 5, 8)
MAX_HISTORICAL_DATE = date(2022, 10, 31)

st.set_page_config(
    page_title="LaDe-D Delivery Operations",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #edf4f8 100%);
        border: 1px solid #dce5eb; padding: 14px 16px; border-radius: 12px;
    }
    .prototype-note {
        padding: 12px 16px; border-radius: 10px;
        background: #fff8e6; border-left: 4px solid #e6a700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading saved forecasting models...")
def cached_models():
    return load_models(ROOT_DIR)


@st.cache_data(show_spinner="Loading processed hourly demand...")
def cached_features():
    return load_feature_data(ROOT_DIR / "processed" / "lade_hourly_features.parquet")


@st.cache_data(show_spinner="Calculating robust region coordinates...")
def cached_coordinates():
    return load_region_coordinates(ROOT_DIR)


@st.cache_resource(show_spinner="Building city-specific spatial graph...")
def cached_graph(coordinates, neighbor_count):
    return build_spatial_graph(coordinates, neighbors=neighbor_count)


def _geo_projection_scale(longitudes, latitudes):
    lon_span = float(np.ptp(longitudes))
    lat_span = float(np.ptp(latitudes))
    mean_latitude = float(np.mean(latitudes))
    city_span = max(lat_span, lon_span * np.cos(np.radians(mean_latitude)), 0.075)
    return min(2400.0, 180.0 / city_span)


def save_dashboard_outputs(
    forecasts,
    hotspots,
    allocations,
    route_results,
    graph_summary,
    fleet_table,
    route_table,
    forecast_result,
    hotspot_result,
):
    save_predictions(forecasts, ROOT_DIR)
    save_hotspots(hotspots, ROOT_DIR)
    save_allocations(allocations, ROOT_DIR)

    route_dir = OUTPUT_DIR / "routes"
    route_dir.mkdir(parents=True, exist_ok=True)
    for strategy, (routes, _, _) in route_results.items():
        save_routes(routes, ROOT_DIR)
        strategy_path = route_dir / f"{strategy.lower()}_routes.json"
        with strategy_path.open("w", encoding="utf-8") as file:
            json.dump(routes, file, indent=2)

    fleet_dir = OUTPUT_DIR / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    fleet_table.to_csv(fleet_dir / "strategy_comparison.csv", index=False)
    route_table.to_csv(route_dir / "strategy_comparison.csv", index=False)

    evaluation_dir = OUTPUT_DIR / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "forecast": forecast_result,
        "hotspots": hotspot_result,
        "spatial_graph": graph_summary,
        "note": (
            "This is a historical-data demonstration and simulated allocation/"
            "route evaluation, not a live delivery-system result."
        ),
    }
    with (evaluation_dir / "current_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2, default=str)


def plot_demand_map(city_frame, city):
    colors = {
        "Normal": "#5B8FF9",
        "Elevated demand": "#F6BD16",
        "High-demand hotspot": "#E8684A",
    }
    center = {
        "lat": float(city_frame["latitude"].median()),
        "lon": float(city_frame["longitude"].median()),
    }
    fig = px.scatter_geo(
        city_frame,
        lat="latitude",
        lon="longitude",
        color="hotspot_classification",
        size="predicted_demand",
        size_max=24,
        color_discrete_map=colors,
        hover_name="region_id",
        hover_data={
            "city": True,
            "predicted_demand": ":.1f",
            "high_demand_probability": ":.3f",
            "neighbor_predicted_demand": ":.1f",
            "neighbor_support": ":.2f",
            "hotspot_score": ":.2f",
            "latitude": False,
            "longitude": False,
        },
        projection="mercator",
        height=610,
        title=f"{city} — forecast regional demand and spatial hotspots",
    )
    fig.update_geos(
        center=center,
        projection_scale=_geo_projection_scale(
            city_frame["longitude"],
            city_frame["latitude"],
        ),
        showland=True,
        landcolor="#f2f5f7",
        showocean=True,
        oceancolor="#e7f0f7",
        showcountries=True,
        countrycolor="#bbc7d1",
        showcoastlines=True,
        coastlinecolor="#bbc7d1",
    )
    fig.update_layout(
        margin={"r": 0, "t": 52, "l": 0, "b": 0},
        legend_title_text="Region status",
    )
    return fig


def plot_routes(routes, city):
    city_routes = [route for route in routes if route["city"] == city]
    fig = go.Figure()
    palette = px.colors.qualitative.Safe
    for index, route in enumerate(city_routes):
        longitudes = [point[0] for point in route["path"]]
        latitudes = [point[1] for point in route["path"]]
        fig.add_trace(
            go.Scattergeo(
                lon=longitudes,
                lat=latitudes,
                mode="lines+markers",
                name=route["route_id"],
                line={"width": 3, "color": palette[index % len(palette)]},
                marker={"size": 7},
                hovertemplate=(
                    f"{route['route_id']}<br>"
                    f"Load: {route['planned_load']:.1f}/"
                    f"{route['vehicle_capacity']}<br>"
                    f"Distance: {route['distance_km']:.2f} km"
                    "<extra></extra>"
                ),
            )
        )
    if city_routes:
        longitudes = [
            point[0] for route in city_routes for point in route["path"]
        ]
        latitudes = [
            point[1] for route in city_routes for point in route["path"]
        ]
        center = {
            "lon": float(np.mean(longitudes)),
            "lat": float(np.mean(latitudes)),
        }
        projection_scale = _geo_projection_scale(longitudes, latitudes)
    else:
        center = {"lon": 120.5, "lat": 31.0}
        projection_scale = 1
    fig.update_layout(
        title=f"{city} — simulated proactive routes",
        geo={
            "center": center,
            "projection": {
                "type": "mercator",
                "scale": projection_scale,
            },
            "showland": True,
            "landcolor": "#f2f5f7",
            "showocean": True,
            "oceancolor": "#e7f0f7",
            "showcountries": True,
            "countrycolor": "#bbc7d1",
            "showcoastlines": True,
            "coastlinecolor": "#bbc7d1",
        },
        height=610,
        margin={"r": 0, "t": 52, "l": 0, "b": 0},
        legend_title_text="Simulated vehicle",
    )
    return fig


def _format_timestamp(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def main():
    st.title("LaDe-D | Delivery demand and fleet planning")
    st.caption(
        "Historical-data engineering prototype • Four Chinese cities • "
        "Hourly region forecasts"
    )
    st.markdown(
        '<div class="prototype-note"><b>Simulation notice:</b> This prototype '
        "uses historical LaDe-D records. Fleet positions, vehicle capacity, "
        "depots, speed and routes are configurable simulation assumptions. "
        "No live orders or road-network travel times are used.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Planning controls")
        forecast_mode = st.radio(
            "Data mode",
            ["Next-hour forecast", "Historical replay"],
            help=(
                "Historical replay forecasts after the selected as-of time. "
                "Future actual demand is evaluation-only and never a model input."
            ),
        )
        as_of_timestamp = None
        if forecast_mode == "Historical replay":
            selected_date = st.date_input(
                "Historical as-of date",
                value=MAX_HISTORICAL_DATE,
                min_value=MIN_HISTORICAL_DATE,
                max_value=MAX_HISTORICAL_DATE,
            )
            selected_hour = st.slider("As-of hour", 0, 23, 12)
            as_of_timestamp = pd.Timestamp(
                datetime.combine(selected_date, datetime.min.time())
            ) + pd.Timedelta(hours=selected_hour)
        horizon = st.slider(
            "Planning horizon (hours)",
            min_value=1,
            max_value=6,
            value=3,
            help=(
                "Hours after the as-of time. Beyond hour one, this is a recursive "
                "simulation with accumulating forecast error, not a validated "
                "multi-hour model."
            ),
        )
        if horizon == 1:
            planning_hour = 1
            st.caption("Fleet allocation uses the only forecast hour available.")
        else:
            planning_hour = st.slider(
                "Allocate fleet for forecast hour",
                min_value=1,
                max_value=horizon,
                value=1,
            )
        probability_threshold = st.slider(
            "High-demand probability threshold",
            min_value=0.05,
            max_value=0.95,
            value=DEFAULT_PROBABILITY_THRESHOLD,
            step=0.01,
            help=(
                "Initial default 0.90 from the prior classifier threshold. "
                "The saved sigmoid calibrator changes probability scale; the "
                "calibration experiment's cost-based values were approximately "
                "0.17–0.20. Adjust for the operating policy."
            ),
        )
        fleet_size = st.slider("Total fleet size", 5, 200, 40, step=1)
        vehicle_capacity = st.slider("Vehicle capacity (orders)", 1, 100, 20)
        neighbor_count = st.slider("Spatial neighbors (k)", 1, 10, 5)
        hotspot_bonus = st.slider(
            "Proactive hotspot allocation bonus",
            min_value=0.0,
            max_value=1.5,
            value=0.5,
            step=0.1,
        )

    features = cached_features()
    artifacts = cached_models()
    coordinates = cached_coordinates()

    if forecast_mode == "Next-hour forecast":
        anchor = pd.Timestamp(features["timestamp"].max())
    else:
        anchor = pd.Timestamp(as_of_timestamp)
        if not (features["timestamp"] == anchor).any():
            st.error(
                f"No processed hourly features exist at {anchor}. "
                "Select a date/hour on or after May 8, 2022."
            )
            st.stop()

    with st.spinner("Generating forecasts, spatial scores, fleet plans and routes..."):
        forecasts = forecast_horizon(
            features,
            artifacts,
            as_of_timestamp=anchor,
            horizon=horizon,
            probability_threshold=probability_threshold,
        )
        graph, graph_summary = cached_graph(coordinates, neighbor_count)
        scored_forecasts = score_hotspots(
            forecasts,
            coordinates,
            graph,
        )
        selected = scored_forecasts.loc[
            scored_forecasts["forecast_hour"] == planning_hour
        ].copy()
        if selected.empty:
            raise RuntimeError("The chosen planning hour contains no forecast rows.")

        allocations = compare_strategies(
            selected,
            fleet_size,
            vehicle_capacity,
            hotspot_bonus=hotspot_bonus,
        )
        scenario_actual = {
            strategy: selected["actual_demand"].to_numpy(dtype=float)
            for strategy in STRATEGIES
        }
        fleet_table = fleet_comparison(
            allocations,
            vehicle_capacity,
            actual_demand=scenario_actual,
        )

        route_results = {}
        route_summaries = {}
        for strategy in STRATEGIES:
            routes, route_rows, summary = optimize_routes(
                selected,
                allocations[strategy],
                vehicle_capacity,
            )
            route_results[strategy] = (routes, route_rows, summary)
            route_summaries[strategy] = summary
        route_table = route_comparison(route_summaries)

        forecast_result = forecast_metrics(
            forecasts.loc[forecasts["forecast_hour"] == planning_hour]
        )
        hotspot_result = hotspot_metrics(selected)
        save_dashboard_outputs(
            forecasts,
            scored_forecasts,
            allocations,
            route_results,
            graph_summary,
            fleet_table,
            route_table,
            forecast_result,
            hotspot_result,
        )

    target_timestamp = anchor + pd.Timedelta(hours=planning_hour)
    metrics_demand = float(selected["predicted_demand"].sum())
    hotspot_count = int(selected["is_hotspot"].sum())
    proactive_coverage = float(
        fleet_table.loc[
            fleet_table["Strategy"] == "Proactive",
            "Demand coverage (%)",
        ].iloc[0]
    )

    st.subheader(
        f"Planning view — {_format_timestamp(target_timestamp)} "
        f"(hour {planning_hour} of {horizon})"
    )
    overview = st.columns(6)
    region_count = selected[["city", "region_id"]].drop_duplicates().shape[0]
    overview[0].metric("Regions", f"{region_count:,}")
    overview[1].metric("Predicted demand", f"{metrics_demand:,.0f} orders")
    overview[2].metric("Spatial hotspots", f"{hotspot_count:,}")
    overview[3].metric("Fleet", f"{fleet_size:,} vehicles")
    overview[4].metric("Proactive coverage", f"{proactive_coverage:.1f}%")
    overview[5].metric("Forecast target", _format_timestamp(target_timestamp))

    tabs = st.tabs(
        [
            "Overview",
            "Demand map",
            "Hotspot analysis",
            "Fleet allocation",
            "Route planning",
            "Model performance",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### Forecast summary")
            st.write(
                f"Forecast origin: **{_format_timestamp(anchor)}**. "
                f"The selected lead is **{planning_hour} hour(s)**."
            )
            if planning_hour > 1:
                st.warning(
                    "This lead uses recursive predicted lags. Error accumulation "
                    "is expected; only the one-hour model was evaluated in the "
                    "existing validation experiment."
                )
            if selected["actual_demand"].notna().any():
                st.info(
                    "Historical replay: actual target demand is shown only for "
                    "evaluation. It was not used to generate forecasts, alerts, "
                    "fleet decisions or routes."
                )
            else:
                st.info(
                    "Future forecast: actual target demand is unavailable; "
                    "model metrics are not reported as live performance."
                )
            st.markdown("#### Spatial graph")
            st.write(
                f"{graph_summary['node_count']} nodes, "
                f"{graph_summary['edge_count']} edges, "
                f"{graph_summary['component_count']} connected components, "
                f"{graph_summary['isolated_node_count']} isolated regions."
            )
            st.write(
                "Nearest-neighbor links are computed within each city only "
                "using Haversine distance."
            )
        with right:
            city_counts = (
                selected.groupby(
                    ["city", "hotspot_classification"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "regions"})
            )
            chart = px.bar(
                city_counts,
                x="city",
                y="regions",
                color="hotspot_classification",
                color_discrete_map={
                    "Normal": "#5B8FF9",
                    "Elevated demand": "#F6BD16",
                    "High-demand hotspot": "#E8684A",
                },
                title="Regional demand status by city",
            )
            chart.update_layout(
                margin={"r": 0, "t": 48, "l": 0, "b": 0},
                legend_title_text="Classification",
            )
            st.plotly_chart(chart, width="stretch")

    with tabs[1]:
        city = st.selectbox(
            "Map city",
            sorted(selected["city"].unique()),
            key="demand_map_city",
        )
        city_frame = selected.loc[selected["city"] == city].copy()
        st.plotly_chart(
            plot_demand_map(city_frame, city),
            width="stretch",
        )
        st.caption(
            "Marker size represents predicted demand. Colors show the "
            "neighbor-supported hotspot class. Map points are region medians "
            "from valid raw delivery coordinates."
        )

    with tabs[2]:
        st.markdown(
            "High hotspot status requires both a high weighted score and "
            "neighbor support; a high model probability alone is not sufficient."
        )
        city = st.selectbox(
            "Analyze city",
            sorted(selected["city"].unique()),
            key="hotspot_city",
        )
        city_hotspots = selected.loc[selected["city"] == city].sort_values(
            ["is_hotspot", "hotspot_score"],
            ascending=[False, False],
        )
        show_columns = [
            "region_id",
            "predicted_demand",
            "high_demand_probability",
            "neighbor_predicted_demand",
            "neighbor_support",
            "hotspot_score",
            "hotspot_classification",
        ]
        st.dataframe(
            city_hotspots[show_columns],
            width="stretch",
            hide_index=True,
        )
        distribution = (
            selected.groupby(
                ["city", "hotspot_classification"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "regions"})
        )
        st.dataframe(distribution, width="stretch", hide_index=True)
        st.caption(
            "Score weights are initial engineering choices (0.50 demand, "
            "0.30 calibrated high-demand probability, 0.20 neighbor support); "
            "they are not claimed to be scientifically optimal."
        )

    with tabs[3]:
        allocation_strategy = st.selectbox(
            "Allocation to inspect",
            STRATEGIES,
            index=2,
        )
        selected_allocation = allocations[allocation_strategy]
        allocation_city = st.selectbox(
            "Fleet city",
            sorted(selected_allocation["city"].unique()),
            key="fleet_city",
        )
        city_alloc = selected_allocation.loc[
            selected_allocation["city"] == allocation_city
        ].sort_values("vehicles_allocated", ascending=False)
        alloc_chart = px.bar(
            city_alloc,
            x="region_id",
            y="vehicles_allocated",
            color="hotspot_classification",
            color_discrete_map={
                "Normal": "#5B8FF9",
                "Elevated demand": "#F6BD16",
                "High-demand hotspot": "#E8684A",
            },
            title=f"{allocation_strategy} vehicles by region — {allocation_city}",
        )
        alloc_chart.update_layout(
            margin={"r": 0, "t": 48, "l": 0, "b": 0},
            xaxis_title="Region",
            yaxis_title="Vehicles",
        )
        st.plotly_chart(alloc_chart, width="stretch")
        st.dataframe(fleet_table, width="stretch", hide_index=True)
        st.caption(
            "Static distributes vehicles evenly. Reactive uses the latest "
            "observed demand available at the as-of time. Proactive uses "
            "predicted demand boosted by hotspot score. Remaining demand and "
            "coverage are evaluated against the same target demand when "
            "historical actuals are available."
        )

    with tabs[4]:
        route_strategy = st.selectbox(
            "Route strategy",
            STRATEGIES,
            index=2,
            key="route_strategy",
        )
        routes, route_rows, route_summary = route_results[route_strategy]
        city = st.selectbox(
            "Route city",
            sorted(selected["city"].unique()),
            key="route_city",
        )
        route_figure = plot_routes(routes, city)
        st.plotly_chart(route_figure, width="stretch")
        st.dataframe(
            route_rows.loc[route_rows["city"] == city],
            width="stretch",
            hide_index=True,
        )
        st.metric(
            "Total route distance (simulation)",
            f"{route_summary['total_travel_distance_km']:.1f} km",
        )
        st.caption(
            "Routes use nearest-neighbor stop ordering and Haversine "
            "straight-line distances. Simulated vehicles start at the region "
            "where they are allocated. This is not road distance, a live "
            "routing service, or a full VRPTW solver."
        )
        st.dataframe(route_table, width="stretch", hide_index=True)

    with tabs[5]:
        st.markdown("#### Existing held-out model results")
        existing_metrics = load_existing_model_performance(ROOT_DIR)
        st.dataframe(
            existing_metrics[
                existing_metrics["Model"].isin(
                    [
                        "Original XGBoost",
                        "Weighted XGBoost",
                        "Existing two-stage",
                        "Sigmoid calibrated two-stage",
                    ]
                )
            ],
            width="stretch",
            hide_index=True,
        )
        if selected["actual_demand"].notna().any():
            st.markdown("#### Historical replay metrics")
            metrics = forecast_metrics(selected)
            hotspot_result = hotspot_metrics(selected)
            st.write(
                {
                    "Forecast MAE": metrics["MAE"],
                    "Forecast RMSE": metrics["RMSE"],
                    "Forecast WAPE (%)": metrics["WAPE (%)"],
                    "Hotspot precision": hotspot_result["precision"],
                    "Hotspot recall": hotspot_result["recall"],
                    "Hotspot F1": hotspot_result["F1"],
                }
            )
            st.caption(hotspot_result["label_note"])
        else:
            st.info(
                "This forecast timestamp has no observed target demand. "
                "Only the previously measured September/October performance "
                "is shown above."
            )
        st.caption(
            "The primary forecaster is the saved original XGBoost model. "
            "The weighted model is a comparison only. The sigmoid classifier "
            "is used for calibrated high-demand probability, not as the "
            "primary demand regressor."
        )

    st.divider()
    st.caption(
        "Current outputs are saved under outputs/ (predictions, hotspots, "
        "fleet allocations, routes and evaluation). LaDe-D covers Chinese "
        "cities and does not directly validate performance on other delivery "
        "networks."
    )


if __name__ == "__main__":
    main()
