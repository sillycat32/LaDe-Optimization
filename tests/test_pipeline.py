import unittest

import numpy as np
import pandas as pd

from src.evaluation import forecast_metrics, hotspot_metrics
from src.fleet_optimizer import (
    STRATEGIES,
    allocate_fleet,
    allocation_metrics,
    compare_strategies,
)
from src.forecasting import forecast_horizon, load_feature_data, load_models
from src.hotspot_detection import (
    build_spatial_graph,
    load_region_coordinates,
    score_hotspots,
)
from src.route_optimizer import optimize_routes


class ForecastPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = load_feature_data()
        cls.artifacts = load_models()

    def test_saved_models_and_feature_schemas_load(self):
        self.assertEqual(
            set(
                self.artifacts.primary_model.named_steps[
                    "preprocessor"
                ].feature_names_in_
            ),
            set(
                self.artifacts.classifier.named_steps[
                    "preprocessor"
                ].feature_names_in_
            ),
        )

    def test_next_hour_recursive_forecast_is_finite_and_nonnegative(self):
        anchor = self.features["timestamp"].max()
        forecast = forecast_horizon(
            self.features,
            self.artifacts,
            horizon=2,
            probability_threshold=0.90,
        )
        expected_regions = self.features.loc[
            self.features["timestamp"] == anchor,
            ["city", "region_id"],
        ].drop_duplicates()
        self.assertEqual(len(forecast), 2 * len(expected_regions))
        self.assertEqual(
            forecast.loc[forecast["forecast_hour"] == 1, "timestamp"].nunique(),
            1,
        )
        self.assertEqual(
            forecast.loc[forecast["forecast_hour"] == 1, "timestamp"].iloc[0],
            anchor + pd.Timedelta(hours=1),
        )
        self.assertTrue(forecast["actual_demand"].isna().all())
        self.assertTrue((forecast["predicted_demand"] >= 0).all())
        self.assertTrue(
            np.isfinite(
                forecast[
                    [
                        "predicted_demand",
                        "high_demand_probability",
                        "last_observed_demand",
                    ]
                ].to_numpy()
            ).all()
        )
        first = forecast.loc[forecast["forecast_hour"] == 1].set_index(
            ["city", "region_id"]
        )
        second = forecast.loc[forecast["forecast_hour"] == 2].set_index(
            ["city", "region_id"]
        )
        pd.testing.assert_series_equal(
            first["last_observed_demand"],
            second["last_observed_demand"],
            check_names=False,
        )

    def test_historical_replay_keeps_actual_target_out_of_prediction_features(self):
        anchor = pd.Timestamp("2022-09-15 12:00:00")
        forecast = forecast_horizon(
            self.features,
            self.artifacts,
            as_of_timestamp=anchor,
            horizon=1,
        )
        expected = self.features.loc[
            self.features["timestamp"] == anchor + pd.Timedelta(hours=1),
            ["city", "region_id", "demand"],
        ].rename(columns={"demand": "expected_actual"})
        aligned = forecast.merge(
            expected,
            on=["city", "region_id"],
            validate="one_to_one",
        )
        self.assertTrue(aligned["actual_demand"].notna().all())
        np.testing.assert_array_equal(
            aligned["actual_demand"].to_numpy(),
            aligned["expected_actual"].to_numpy(),
        )
        observed_at_anchor = self.features.loc[
            self.features["timestamp"] == anchor,
            ["city", "region_id", "demand"],
        ].rename(columns={"demand": "anchor_demand"})
        aligned = aligned.merge(
            observed_at_anchor,
            on=["city", "region_id"],
            validate="one_to_one",
        )
        np.testing.assert_array_equal(
            aligned["last_observed_demand"].to_numpy(),
            aligned["anchor_demand"].to_numpy(),
        )


class SpatialFleetRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coordinates = load_region_coordinates()

    def test_all_regions_have_valid_coordinates_and_city_isolated_graph(self):
        features = load_feature_data()
        expected = features[["city", "region_id"]].drop_duplicates()
        joined = expected.merge(
            self.coordinates,
            on=["city", "region_id"],
            how="left",
            validate="one_to_one",
        )
        self.assertFalse(joined[["longitude", "latitude"]].isna().any().any())
        self.assertTrue(joined["longitude"].between(-180, 180).all())
        self.assertTrue(joined["latitude"].between(-90, 90).all())

        graph, summary = build_spatial_graph(self.coordinates, neighbors=5)
        self.assertEqual(graph.number_of_nodes(), len(self.coordinates))
        self.assertGreater(summary["edge_count"], 0)
        self.assertTrue(all(left[0] == right[0] for left, right in graph.edges))

    def test_hotspot_scores_and_neighbor_support_are_valid(self):
        features = load_feature_data()
        artifacts = load_models()
        forecast = forecast_horizon(features, artifacts, horizon=1)
        graph, _ = build_spatial_graph(self.coordinates, neighbors=5)
        hotspots = score_hotspots(forecast, self.coordinates, graph)
        self.assertTrue(hotspots["hotspot_score"].between(0, 1).all())
        self.assertTrue(hotspots["neighbor_support"].between(0, 1).all())
        self.assertTrue(
            set(hotspots["hotspot_classification"].unique()).issubset(
                {"Normal", "Elevated demand", "High-demand hotspot"}
            )
        )

    def test_reactive_allocation_uses_only_last_observed_demand(self):
        frame = pd.DataFrame(
            {
                "city": ["A", "A", "A"],
                "region_id": ["1", "2", "3"],
                "predicted_demand": [90.0, 60.0, 20.0],
                "last_observed_demand": [10.0, 70.0, 30.0],
                "actual_demand": [999.0, 0.0, 0.0],
                "hotspot_score": [0.9, 0.2, 0.1],
            }
        )
        reactive = allocate_fleet(
            frame,
            fleet_size=4,
            vehicle_capacity=25,
            strategy="Reactive",
        )
        self.assertEqual(
            reactive["allocation_demand"].tolist(),
            [10.0, 70.0, 30.0],
        )
        for strategy in STRATEGIES:
            allocation = allocate_fleet(
                frame,
                fleet_size=4,
                vehicle_capacity=25,
                strategy=strategy,
            )
            self.assertLessEqual(allocation["vehicles_allocated"].sum(), 4)
            self.assertTrue(
                (allocation["vehicles_allocated"] >= 0).all()
            )

    def test_comparison_and_route_plans_respect_fleet_and_capacity(self):
        sample = pd.DataFrame(
            {
                "city": ["A", "A", "A"],
                "region_id": ["1", "2", "3"],
                "longitude": [120.0, 120.1, 120.2],
                "latitude": [30.0, 30.1, 30.0],
                "predicted_demand": [15.0, 35.0, 10.0],
                "actual_demand": [10.0, 40.0, 8.0],
                "last_observed_demand": [11.0, 30.0, 7.0],
                "hotspot_score": [0.1, 0.9, 0.2],
                "hotspot_classification": [
                    "Normal",
                    "High-demand hotspot",
                    "Normal",
                ],
            }
        )
        allocations = compare_strategies(
            sample,
            fleet_size=4,
            vehicle_capacity=20,
        )
        comparison_rows = []
        for strategy, allocation in allocations.items():
            self.assertLessEqual(allocation["vehicles_allocated"].sum(), 4)
            metrics = allocation_metrics(
                allocation,
                sample["actual_demand"].to_numpy(),
            )
            comparison_rows.append(metrics)
            routes, _, summary = optimize_routes(
                sample,
                allocation,
                vehicle_capacity=20,
            )
            self.assertLessEqual(
                sum(route["planned_load"] for route in routes),
                sum(route["vehicle_capacity"] for route in routes),
            )
            self.assertEqual(
                summary["distance_method"],
                "Haversine straight-line simulation",
            )
            for route in routes:
                self.assertLessEqual(route["planned_load"], 20)
                self.assertGreaterEqual(len(route["path"]), 2)
                self.assertTrue(
                    all(
                        -180 <= longitude <= 180 and -90 <= latitude <= 90
                        for longitude, latitude in route["path"]
                    )
                )
        self.assertEqual(len(comparison_rows), 3)

    def test_evaluation_metrics_are_defined_for_historical_data(self):
        frame = pd.DataFrame(
            {
                "actual_demand": [10.0, 20.0, 0.0],
                "predicted_demand": [12.0, 18.0, 1.0],
                "is_hotspot": [False, True, False],
            }
        )
        self.assertEqual(forecast_metrics(frame)["observations"], 3)
        self.assertEqual(hotspot_metrics(frame)["observations"], 3)


if __name__ == "__main__":
    unittest.main()
