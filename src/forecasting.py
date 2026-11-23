from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT_DIR / "processed" / "lade_hourly_features.parquet"
PRIMARY_MODEL_PATH = ROOT_DIR / "models" / "xgboost_demand_model.joblib"
WEIGHTED_MODEL_PATH = ROOT_DIR / "models" / "xgboost_weighted_model.joblib"
CLASSIFIER_PATH = (
    ROOT_DIR / "models" / "two_stage" / "high_demand_classifier.joblib"
)
CALIBRATOR_PATH = (
    ROOT_DIR / "models" / "calibration_experiment" / "sigmoid_calibrator.joblib"
)
HIGH_DEMAND_THRESHOLD = 48.0
DEFAULT_PROBABILITY_THRESHOLD = 0.90
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 168]
CATEGORICAL_FEATURES = {"city", "region_id"}


@dataclass(frozen=True)
class ForecastArtifacts:
    primary_model: object
    weighted_model: object
    classifier: object
    sigmoid_calibrator: object


def load_feature_data(path=FEATURES_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed feature data not found: {path}")
    data = pd.read_parquet(path)
    required = {
        "city",
        "region_id",
        "timestamp",
        "demand",
        *[f"lag_{lag}" for lag in LAGS],
        *[f"rolling_mean_{window}h" for window in ROLLING_WINDOWS],
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Processed feature data is missing columns: {missing}")
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["city"] = data["city"].astype(str)
    data["region_id"] = data["region_id"].astype(str)
    return data.sort_values(
        ["city", "region_id", "timestamp"]
    ).reset_index(drop=True)


def load_models(root_dir=ROOT_DIR):
    root = Path(root_dir)
    paths = {
        "primary_model": root / "models" / "xgboost_demand_model.joblib",
        "weighted_model": root / "models" / "xgboost_weighted_model.joblib",
        "classifier": (
            root / "models" / "two_stage" / "high_demand_classifier.joblib"
        ),
        "sigmoid_calibrator": (
            root
            / "models"
            / "calibration_experiment"
            / "sigmoid_calibrator.joblib"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required saved model artifacts are missing:\n" + "\n".join(missing)
        )

    artifacts = ForecastArtifacts(
        primary_model=joblib.load(paths["primary_model"]),
        weighted_model=joblib.load(paths["weighted_model"]),
        classifier=joblib.load(paths["classifier"]),
        sigmoid_calibrator=joblib.load(paths["sigmoid_calibrator"]),
    )
    schemas = {
        name: tuple(
            model.named_steps["preprocessor"].feature_names_in_.tolist()
        )
        for name, model in [
            ("primary", artifacts.primary_model),
            ("weighted", artifacts.weighted_model),
            ("classifier", artifacts.classifier),
        ]
    }
    if any(set(schema) != set(schemas["primary"]) for schema in schemas.values()):
        raise ValueError(f"Saved models have incompatible feature schemas: {schemas}")
    return artifacts


def _model_features(frame, model):
    expected = list(model.named_steps["preprocessor"].feature_names_in_)
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction features are missing model inputs: {missing}")
    X = frame.loc[:, expected].copy()
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(str)
    return X


def _apply_sigmoid_calibration(probability, calibrator):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probability / (1 - probability)).reshape(-1, 1)
    return np.clip(calibrator.predict_proba(logits)[:, 1], 0, 1)


def _predict_rows(frame, artifacts, probability_threshold):
    primary_features = _model_features(frame, artifacts.primary_model)
    classifier_features = _model_features(frame, artifacts.classifier)
    weighted_features = _model_features(frame, artifacts.weighted_model)

    predicted_demand = np.maximum(
        artifacts.primary_model.predict(primary_features),
        0,
    )
    raw_probability = artifacts.classifier.predict_proba(
        classifier_features
    )[:, 1]
    high_probability = _apply_sigmoid_calibration(
        raw_probability,
        artifacts.sigmoid_calibrator,
    )
    weighted_prediction = np.maximum(
        artifacts.weighted_model.predict(weighted_features),
        0,
    )

    result = frame[["city", "region_id", "timestamp"]].copy()
    result["actual_demand"] = (
        frame["actual_demand"].to_numpy()
        if "actual_demand" in frame
        else np.nan
    )
    result["predicted_demand"] = predicted_demand
    result["high_demand_probability"] = high_probability
    result["predicted_high_demand"] = (
        high_probability >= probability_threshold
    ).astype(np.int8)
    result["weighted_xgboost_prediction"] = weighted_prediction
    result["probability_threshold"] = float(probability_threshold)
    result["last_observed_demand"] = frame["last_observed_demand"].to_numpy(
        dtype=float
    )
    result["forecast_hour"] = frame["forecast_hour"].to_numpy(dtype=int)

    numeric = result[
        [
            "predicted_demand",
            "high_demand_probability",
            "weighted_xgboost_prediction",
            "last_observed_demand",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("A model produced a non-finite forecast.")
    if (result["predicted_demand"] < 0).any():
        raise ValueError("Demand forecasts must be nonnegative.")
    return result


def _calendar_features(timestamp):
    return {
        "hour": timestamp.hour,
        "day_of_week": timestamp.day_of_week,
        "day_of_month": timestamp.day,
        "month": timestamp.month,
        "is_weekend": int(timestamp.day_of_week >= 5),
        "hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
        "hour_cos": np.cos(2 * np.pi * timestamp.hour / 24),
        "dow_sin": np.sin(2 * np.pi * timestamp.day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * timestamp.day_of_week / 7),
    }


def _build_next_hour_rows(histories, anchor, forecast_hour):
    target_timestamp = anchor + pd.Timedelta(hours=1)
    rows = []
    for (city, region_id), history in histories.items():
        values = history["demand"]
        timestamps = history["timestamp"]
        if len(values) < max(max(LAGS), max(ROLLING_WINDOWS)):
            raise ValueError(
                f"Insufficient causal history for {city}/{region_id} at {anchor}."
            )
        if timestamps[-1] != anchor:
            raise ValueError(
                f"Region {city}/{region_id} has no observation at anchor {anchor}."
            )

        row = {
            "city": city,
            "region_id": region_id,
            "timestamp": target_timestamp,
            "actual_demand": np.nan,
            "last_observed_demand": float(values[-1]),
            "forecast_hour": forecast_hour,
            **_calendar_features(target_timestamp),
        }
        for lag in LAGS:
            row[f"lag_{lag}"] = float(values[-lag])
        for window in ROLLING_WINDOWS:
            row[f"rolling_mean_{window}h"] = float(
                np.mean(values[-window:])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _historical_actual_lookup(data, target_timestamps):
    targets = data.loc[
        data["timestamp"].isin(target_timestamps),
        ["city", "region_id", "timestamp", "demand"],
    ]
    return {
        (row.city, row.region_id, row.timestamp): float(row.demand)
        for row in targets.itertuples(index=False)
    }


def forecast_horizon(
    data,
    artifacts,
    as_of_timestamp=None,
    horizon=1,
    probability_threshold=DEFAULT_PROBABILITY_THRESHOLD,
):
    """Forecast iteratively from observations available at the requested anchor."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one hour.")
    if not 0 <= probability_threshold <= 1:
        raise ValueError("Probability threshold must be between zero and one.")
    if data.empty:
        raise ValueError("Cannot forecast from an empty feature dataset.")

    if as_of_timestamp is None:
        anchor = pd.Timestamp(data["timestamp"].max())
    else:
        anchor = pd.Timestamp(as_of_timestamp)
    observed = data.loc[data["timestamp"] <= anchor].copy()
    if observed.empty or not (observed["timestamp"] == anchor).any():
        raise ValueError(f"No processed observations exist at anchor {anchor}.")

    histories = {}
    for (city, region_id), group in observed.groupby(
        ["city", "region_id"],
        sort=True,
    ):
        group = group.sort_values("timestamp")
        tail = group.tail(168)
        histories[(str(city), str(region_id))] = {
            "timestamp": tail["timestamp"].to_numpy(),
            "demand": tail["demand"].to_numpy(dtype=float),
        }
    anchor_observations = {
        key: float(history["demand"][-1])
        for key, history in histories.items()
    }

    actual_lookup = _historical_actual_lookup(
        data,
        [
            anchor + pd.Timedelta(hours=step)
            for step in range(1, horizon + 1)
        ],
    )
    outputs = []
    current_anchor = anchor
    for forecast_hour in range(1, horizon + 1):
        inputs = _build_next_hour_rows(
            histories,
            current_anchor,
            forecast_hour,
        )
        predictions = _predict_rows(
            inputs,
            artifacts,
            probability_threshold,
        )
        predictions["last_observed_demand"] = [
            anchor_observations[(row.city, row.region_id)]
            for row in predictions.itertuples(index=False)
        ]
        target_timestamp = current_anchor + pd.Timedelta(hours=1)
        if actual_lookup:
            predictions["actual_demand"] = [
                actual_lookup.get((row.city, row.region_id, row.timestamp), np.nan)
                for row in predictions.itertuples(index=False)
            ]
        outputs.append(predictions)

        for row in predictions.itertuples(index=False):
            key = (str(row.city), str(row.region_id))
            histories[key]["demand"] = np.append(
                histories[key]["demand"],
                float(row.predicted_demand),
            )
            histories[key]["timestamp"] = np.append(
                histories[key]["timestamp"],
                np.datetime64(target_timestamp),
            )
        current_anchor = target_timestamp

    return pd.concat(outputs, ignore_index=True).sort_values(
        ["forecast_hour", "city", "region_id"]
    ).reset_index(drop=True)


def save_predictions(predictions, root_dir=ROOT_DIR):
    output = Path(root_dir) / "outputs" / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "current_forecast.parquet"
    predictions.to_parquet(path, index=False)
    return path
