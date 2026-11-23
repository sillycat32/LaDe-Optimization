import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "lade_hourly_features.parquet"
MODEL_DIR = BASE_DIR / "models"
TWO_STAGE_DIR = MODEL_DIR / "two_stage"
OUTPUT_DIR = MODEL_DIR / "calibration_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2022-08-31 23:00:00")
CALIBRATION_FOLDS = [
    {
        "train_end": pd.Timestamp("2022-06-30 23:00:00"),
        "calibration_start": pd.Timestamp("2022-07-01 00:00:00"),
        "calibration_end": pd.Timestamp("2022-07-31 23:00:00"),
    },
    {
        "train_end": pd.Timestamp("2022-07-31 23:00:00"),
        "calibration_start": pd.Timestamp("2022-08-01 00:00:00"),
        "calibration_end": pd.Timestamp("2022-08-31 23:00:00"),
    },
]
PERIODS = {
    "September": (
        pd.Timestamp("2022-09-01 00:00:00"),
        pd.Timestamp("2022-09-30 23:00:00"),
        TWO_STAGE_DIR / "validation_predictions.parquet",
    ),
    "October": (
        pd.Timestamp("2022-10-01 00:00:00"),
        pd.Timestamp("2022-10-31 23:00:00"),
        TWO_STAGE_DIR / "october_predictions.parquet",
    ),
}

HIGH_DEMAND_THRESHOLD = 48.0
EXTREME_DEMAND_THRESHOLD = 74.0
PROBABILITY_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0
COST_THRESHOLD_CANDIDATES = sorted(
    set(
        PROBABILITY_THRESHOLDS
        + [0.05, 0.10, 0.15, 1 / 6, 0.20, 0.25, 0.95]
    )
)
RANDOM_STATE = 42
EPSILON = 1e-6

IDENTIFIERS = ["city", "region_id", "timestamp"]
BASE_PREDICTION_COLUMNS = [
    "demand",
    "high_demand_probability",
    "low_prediction",
    "high_prediction",
    "combined_prediction",
    "original_xgboost_prediction",
    "weighted_xgboost_prediction",
]


def require_files(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required project artifacts are missing:\n" + "\n".join(missing))


def normalize_identifiers(frame):
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["city"] = result["city"].astype(str)
    result["region_id"] = result["region_id"].astype(str)
    return result


def load_aligned_predictions(data, name, start, end, path):
    predictions = normalize_identifiers(pd.read_parquet(path))
    expected = normalize_identifiers(
        data.loc[
            (data["timestamp"] >= start) & (data["timestamp"] <= end),
            IDENTIFIERS + ["demand"],
        ]
    )

    if predictions.duplicated(IDENTIFIERS).any():
        raise ValueError(f"{name} prediction file contains duplicate region-hour keys.")
    if len(predictions) != len(expected):
        raise ValueError(
            f"{name} prediction count {len(predictions):,} does not match "
            f"the feature data count {len(expected):,}."
        )

    expected_keys = expected[IDENTIFIERS].sort_values(IDENTIFIERS).reset_index(drop=True)
    prediction_keys = (
        predictions[IDENTIFIERS]
        .sort_values(IDENTIFIERS)
        .reset_index(drop=True)
    )
    if not expected_keys.equals(prediction_keys):
        raise ValueError(f"{name} predictions are not aligned to the expected region-hours.")

    predictions = predictions.merge(
        expected[IDENTIFIERS + ["demand"]].rename(
            columns={"demand": "verified_demand"}
        ),
        on=IDENTIFIERS,
        how="left",
        validate="one_to_one",
    )
    if not np.array_equal(
        predictions["demand"].to_numpy(),
        predictions["verified_demand"].to_numpy(),
    ):
        raise ValueError(f"{name} prediction targets do not match the feature dataset.")
    predictions = predictions.drop(columns="verified_demand")

    missing_columns = sorted(set(BASE_PREDICTION_COLUMNS) - set(predictions.columns))
    if missing_columns:
        raise ValueError(
            f"{name} prediction file is missing required columns: {missing_columns}"
        )
    return predictions


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def fit_time_aware_calibrators(data, feature_columns, base_classifier):
    oof_parts = []
    for fold_number, fold in enumerate(CALIBRATION_FOLDS, start=1):
        train = data.loc[data["timestamp"] <= fold["train_end"]]
        calibration = data.loc[
            (data["timestamp"] >= fold["calibration_start"])
            & (data["timestamp"] <= fold["calibration_end"])
        ]
        if train.empty or calibration.empty:
            raise ValueError(f"Calibration fold {fold_number} is empty.")

        X_train = train[feature_columns]
        y_train = (train["demand"] >= HIGH_DEMAND_THRESHOLD).astype(int)
        X_calibration = calibration[feature_columns]
        y_calibration = (calibration["demand"] >= HIGH_DEMAND_THRESHOLD).astype(int)
        if y_train.nunique() != 2 or y_calibration.nunique() != 2:
            raise ValueError(f"Calibration fold {fold_number} does not contain both classes.")

        fold_positive_count = int(y_train.sum())
        fold_negative_count = int(len(y_train) - fold_positive_count)
        fold_scale_pos_weight = fold_negative_count / fold_positive_count
        fold_model = clone(base_classifier)
        fold_model.set_params(classifier__scale_pos_weight=fold_scale_pos_weight)
        fold_model.fit(X_train, y_train)
        raw_probability = fold_model.predict_proba(X_calibration)[:, 1]

        part = calibration[IDENTIFIERS + ["demand"]].copy()
        part["fold"] = fold_number
        part["fold_scale_pos_weight"] = fold_scale_pos_weight
        part["raw_probability"] = raw_probability
        oof_parts.append(part)
        print(
            f"Calibration fold {fold_number}: "
            f"train through {fold['train_end']}, "
            f"out-of-time predictions {fold['calibration_start']} "
            f"to {fold['calibration_end']} ({len(part):,} rows), "
            f"fold-only scale_pos_weight={fold_scale_pos_weight:.4f}"
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    oof_events = (oof["demand"].to_numpy() >= HIGH_DEMAND_THRESHOLD).astype(int)
    raw_scores = oof["raw_probability"].to_numpy(dtype=float)

    sigmoid = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    sigmoid.fit(logit(raw_scores), oof_events)

    isotonic = IsotonicRegression(
        y_min=EPSILON,
        y_max=1 - EPSILON,
        out_of_bounds="clip",
    )
    isotonic.fit(raw_scores, oof_events)

    oof["sigmoid_probability"] = sigmoid.predict_proba(logit(raw_scores))[:, 1]
    oof["isotonic_probability"] = isotonic.predict(raw_scores)
    oof.to_parquet(OUTPUT_DIR / "calibration_oof_predictions.parquet", index=False)
    joblib.dump(sigmoid, OUTPUT_DIR / "sigmoid_calibrator.joblib")
    joblib.dump(isotonic, OUTPUT_DIR / "isotonic_calibrator.joblib")

    print(
        "Sigmoid slope:",
        float(sigmoid.coef_[0, 0]),
        "(positive slope preserves ranking; negative slope reverses it)",
    )
    return oof, sigmoid, isotonic


def calibrate_probability(probability, sigmoid, isotonic):
    probability = np.asarray(probability, dtype=float)
    return {
        "Uncalibrated": np.clip(probability, EPSILON, 1 - EPSILON),
        "Sigmoid": np.clip(
            sigmoid.predict_proba(logit(probability))[:, 1],
            EPSILON,
            1 - EPSILON,
        ),
        "Isotonic": np.clip(
            isotonic.predict(probability),
            EPSILON,
            1 - EPSILON,
        ),
    }


def threshold_row(actual_events, probability, threshold, period, method, selection):
    predicted_events = probability >= threshold
    tp = int(np.sum(actual_events & predicted_events))
    fp = int(np.sum(~actual_events & predicted_events))
    fn = int(np.sum(actual_events & ~predicted_events))
    tn = int(np.sum(~actual_events & ~predicted_events))
    cost = FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn
    return {
        "Period": period,
        "Probability method": method,
        "Threshold": float(threshold),
        "Threshold selection": selection,
        "Precision": float(precision_score(actual_events, predicted_events, zero_division=0)),
        "Recall": float(recall_score(actual_events, predicted_events, zero_division=0)),
        "F1": float(f1_score(actual_events, predicted_events, zero_division=0)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "False-positive cost": FALSE_POSITIVE_COST * fp,
        "False-negative cost": FALSE_NEGATIVE_COST * fn,
        "Total cost": float(cost),
        "Cost per observation": float(cost / len(actual_events)),
    }


def select_thresholds(september, calibrated_probabilities):
    actual_events = september["demand"].to_numpy() >= HIGH_DEMAND_THRESHOLD
    threshold_rows = []
    selected = {}

    for method, probability in calibrated_probabilities.items():
        rows = [
            threshold_row(
                actual_events,
                probability,
                threshold,
                "September",
                method,
                "Fixed grid",
            )
            for threshold in PROBABILITY_THRESHOLDS
        ]
        threshold_rows.extend(rows)
        table = pd.DataFrame(rows)
        f1_row = table.sort_values(
            ["F1", "Threshold"],
            ascending=[False, False],
        ).iloc[0]

        cost_candidates = [
            threshold_row(
                actual_events,
                probability,
                threshold,
                "September",
                method,
                (
                    "Fixed grid"
                    if threshold in PROBABILITY_THRESHOLDS
                    else "Operational cost candidate"
                ),
            )
            for threshold in COST_THRESHOLD_CANDIDATES
        ]
        threshold_rows.extend(
            row
            for row in cost_candidates
            if row["Threshold"] not in PROBABILITY_THRESHOLDS
        )
        cost_table = pd.DataFrame(cost_candidates)
        cost_row = cost_table.sort_values(
            ["Total cost", "Threshold"],
            ascending=[True, False],
        ).iloc[0]
        selected[method] = {
            "best_f1_threshold": float(f1_row["Threshold"]),
            "operational_threshold": float(cost_row["Threshold"]),
        }
        selected_cost_row = threshold_row(
            actual_events,
            probability,
            selected[method]["operational_threshold"],
            "September",
            method,
            "Selected minimum cost",
        )
        selected_cost_row["Selected by cost"] = True
        threshold_rows.append(selected_cost_row)

    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(OUTPUT_DIR / "threshold_comparison.csv", index=False)
    return threshold_table, selected


def classifier_metrics(period, actual_events, probability, threshold, method):
    predicted_events = probability >= threshold
    tp = int(np.sum(actual_events & predicted_events))
    fp = int(np.sum(~actual_events & predicted_events))
    fn = int(np.sum(actual_events & ~predicted_events))
    tn = int(np.sum(~actual_events & ~predicted_events))
    return {
        "Period": period,
        "Probability method": method,
        "Operational threshold": threshold,
        "Accuracy": float(accuracy_score(actual_events, predicted_events)),
        "Precision": float(precision_score(actual_events, predicted_events, zero_division=0)),
        "Recall": float(recall_score(actual_events, predicted_events, zero_division=0)),
        "F1": float(f1_score(actual_events, predicted_events, zero_division=0)),
        "Brier score": float(brier_score_loss(actual_events, probability)),
        "Log loss": float(log_loss(actual_events, probability, labels=[0, 1])),
        "PR-AUC": float(average_precision_score(actual_events, probability)),
        "ROC-AUC": float(roc_auc_score(actual_events, probability)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Positive observations": int(np.sum(actual_events)),
        "Observations": int(len(actual_events)),
    }


def plot_reliability(period, probabilities, actual_events):
    fig, axis = plt.subplots(figsize=(7.5, 6))
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for method, probability in probabilities.items():
        observed_frequency, mean_probability = calibration_curve(
            actual_events,
            probability,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(
            mean_probability,
            observed_frequency,
            marker="o",
            linewidth=1.5,
            label=method,
        )
    axis.set(
        title=f"{period} high-demand reliability",
        xlabel="Mean predicted high-demand probability",
        ylabel="Observed high-demand frequency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{period.lower()}_reliability.png", dpi=160)
    plt.close(fig)


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    total_actual = float(np.sum(actual))
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE (%)": (
            float(np.sum(np.abs(actual - predicted)) / total_actual * 100)
            if total_actual > 0
            else float("nan")
        ),
    }


def event_metrics(actual_events, predicted_events):
    return {
        "Extreme precision": float(
            precision_score(actual_events, predicted_events, zero_division=0)
        ),
        "Extreme recall": float(
            recall_score(actual_events, predicted_events, zero_division=0)
        ),
        "Extreme F1": float(
            f1_score(actual_events, predicted_events, zero_division=0)
        ),
    }


def fit_high_expert_variants(data, feature_columns, high_model):
    train = data.loc[data["timestamp"] <= TRAIN_END].copy()
    high_mask = train["demand"] >= HIGH_DEMAND_THRESHOLD
    X_high = train.loc[high_mask, feature_columns]
    y_high = train.loc[high_mask, "demand"].astype(float)
    if X_high.empty:
        raise ValueError("No high-demand samples are available to fit the experts.")

    high_params = high_model.named_steps["regressor"].get_params()
    sample_weights = np.ones(len(y_high), dtype=float)
    sample_weights[y_high.to_numpy() >= EXTREME_DEMAND_THRESHOLD] = 3.0

    variants = {
        "Baseline depth 8": high_model,
        "Extreme-weighted depth 8": clone(high_model),
        "Depth 6": clone(high_model),
        "Depth 10": clone(high_model),
        "Quantile median depth 8": clone(high_model),
    }
    variants["Extreme-weighted depth 8"].fit(
        X_high,
        y_high,
        regressor__sample_weight=sample_weights,
    )
    variants["Depth 6"].set_params(regressor__max_depth=6)
    variants["Depth 6"].fit(X_high, y_high)
    variants["Depth 10"].set_params(regressor__max_depth=10)
    variants["Depth 10"].fit(X_high, y_high)
    variants["Quantile median depth 8"].set_params(
        regressor__objective="reg:quantileerror",
        regressor__quantile_alpha=0.5,
    )
    variants["Quantile median depth 8"].fit(X_high, y_high)

    model_names = {
        "Baseline depth 8": "baseline_high_expert.joblib",
        "Extreme-weighted depth 8": "extreme_weighted_high_expert.joblib",
        "Depth 6": "depth_6_high_expert.joblib",
        "Depth 10": "depth_10_high_expert.joblib",
        "Quantile median depth 8": "quantile_median_high_expert.joblib",
    }
    for name, model in variants.items():
        joblib.dump(model, OUTPUT_DIR / model_names[name])

    settings = {
        name: {
            key: value
            for key, value in model.named_steps["regressor"].get_params().items()
            if key
            in {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "reg_alpha",
                "reg_lambda",
                "objective",
                "quantile_alpha",
                "random_state",
            }
        }
        for name, model in variants.items()
    }
    settings["Extreme-weighted depth 8"]["sample_weight_rule"] = (
        "1.0 for demand < 74; 3.0 for demand >= 74 (within the >=48 training subset)"
    )
    settings["Quantile median depth 8"]["selection_note"] = (
        "Median quantile is included as a conditional-median alternative to squared-error mean fitting."
    )
    return variants, settings, high_params


def add_candidate_predictions(
    frame,
    feature_frame,
    feature_columns,
    variants,
    sigmoid,
    isotonic,
):
    result = frame.copy()
    calibrated = calibrate_probability(
        result["high_demand_probability"].to_numpy(),
        sigmoid,
        isotonic,
    )
    for method, probability in calibrated.items():
        result[f"{method.lower()}_probability"] = probability

    feature_values = [
        column for column in feature_columns if column not in IDENTIFIERS
    ]
    working = result.merge(
        feature_frame[IDENTIFIERS + feature_values],
        on=IDENTIFIERS,
        how="left",
        validate="one_to_one",
    )
    if working[feature_values].isna().any().any():
        raise ValueError("Missing validation features after prediction-key alignment.")
    X = working[feature_columns].copy()
    for name, model in variants.items():
        key = name.lower().replace(" ", "_")
        high_prediction = np.maximum(model.predict(X), 0)
        result[f"high_expert_{key}"] = high_prediction
        for method in ("Uncalibrated", "Sigmoid", "Isotonic"):
            probability = calibrated[method]
            result[f"combined_{method.lower()}_{key}"] = np.maximum(
                (1 - probability) * result["low_prediction"].to_numpy()
                + probability * high_prediction,
                0,
            )

    return result


def evaluate_regressions(period, predictions, selected_calibration):
    actual = predictions["demand"].to_numpy(dtype=float)
    high_mask = actual >= HIGH_DEMAND_THRESHOLD
    extreme_mask = actual >= EXTREME_DEMAND_THRESHOLD
    extreme_events = actual >= EXTREME_DEMAND_THRESHOLD

    model_columns = [
        ("Original XGBoost", "original_xgboost_prediction"),
        ("Weighted XGBoost", "weighted_xgboost_prediction"),
        ("Existing two-stage", "combined_prediction"),
        ("Sigmoid calibrated two-stage", "combined_sigmoid_baseline_depth_8"),
        ("Isotonic calibrated two-stage", "combined_isotonic_baseline_depth_8"),
    ]
    selected_variant_key = selected_calibration.lower().replace(" ", "_")
    for variant_name in [
        "Baseline depth 8",
        "Extreme-weighted depth 8",
        "Depth 6",
        "Depth 10",
        "Quantile median depth 8",
    ]:
        variant_key = variant_name.lower().replace(" ", "_")
        if variant_name != "Baseline depth 8":
            model_columns.append(
                (
                    f"{selected_calibration} two-stage + {variant_name}",
                    f"combined_{selected_variant_key}_{variant_key}",
                )
            )

    rows = []
    for model_name, column in model_columns:
        predicted = np.maximum(predictions[column].to_numpy(dtype=float), 0)
        row = {
            "Period": period,
            "Model": model_name,
            **regression_metrics(actual, predicted),
            "High-demand observations": int(high_mask.sum()),
            "High-demand MAE (actual >=48)": (
                float(mean_absolute_error(actual[high_mask], predicted[high_mask]))
                if high_mask.any()
                else float("nan")
            ),
            "Extreme-demand observations": int(extreme_mask.sum()),
            "Extreme-demand MAE (actual >=74)": (
                float(mean_absolute_error(actual[extreme_mask], predicted[extreme_mask]))
                if extreme_mask.any()
                else float("nan")
            ),
            **event_metrics(
                extreme_events,
                predicted >= EXTREME_DEMAND_THRESHOLD,
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_experts(period, predictions):
    actual = predictions["demand"].to_numpy(dtype=float)
    high_mask = actual >= HIGH_DEMAND_THRESHOLD
    extreme_mask = actual >= EXTREME_DEMAND_THRESHOLD
    rows = []
    for name in [
        "Baseline depth 8",
        "Extreme-weighted depth 8",
        "Depth 6",
        "Depth 10",
        "Quantile median depth 8",
    ]:
        key = name.lower().replace(" ", "_")
        predicted = predictions[f"high_expert_{key}"].to_numpy(dtype=float)
        rows.append(
            {
                "Period": period,
                "Expert": name,
                "High-demand observations": int(high_mask.sum()),
                "High-demand MAE (actual >=48)": (
                    float(mean_absolute_error(actual[high_mask], predicted[high_mask]))
                    if high_mask.any()
                    else float("nan")
                ),
                "Extreme-demand observations": int(extreme_mask.sum()),
                "Extreme-demand MAE (actual >=74)": (
                    float(mean_absolute_error(actual[extreme_mask], predicted[extreme_mask]))
                    if extreme_mask.any()
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    required = [
        DATA_PATH,
        MODEL_DIR / "xgboost_demand_model.joblib",
        MODEL_DIR / "xgboost_weighted_model.joblib",
        TWO_STAGE_DIR / "high_demand_classifier.joblib",
        TWO_STAGE_DIR / "low_demand_regressor.joblib",
        TWO_STAGE_DIR / "high_demand_regressor.joblib",
    ] + [path for _, _, path in PERIODS.values()]
    require_files(required)

    data = pd.read_parquet(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["city"] = data["city"].astype(str)
    data["region_id"] = data["region_id"].astype(str)
    data = data.sort_values(["timestamp", "city", "region_id"]).reset_index(drop=True)

    target_columns = {"demand", "timestamp", "ds"}
    feature_columns = [column for column in data.columns if column not in target_columns]
    if "demand" in feature_columns or "timestamp" in feature_columns:
        raise ValueError("Target or timestamp was incorrectly selected as a feature.")

    classifier = joblib.load(TWO_STAGE_DIR / "high_demand_classifier.joblib")
    low_model = joblib.load(TWO_STAGE_DIR / "low_demand_regressor.joblib")
    high_model = joblib.load(TWO_STAGE_DIR / "high_demand_regressor.joblib")
    joblib.dump(classifier, OUTPUT_DIR / "base_high_demand_classifier.joblib")
    joblib.dump(low_model, OUTPUT_DIR / "base_low_demand_regressor.joblib")
    del low_model

    expected_features = set(
        classifier.named_steps["preprocessor"].feature_names_in_.tolist()
    )
    if set(feature_columns) != expected_features:
        raise ValueError(
            "Processed feature columns differ from the saved classifier inputs: "
            f"data-only={sorted(set(feature_columns) - expected_features)}, "
            f"model-only={sorted(expected_features - set(feature_columns))}"
        )

    training = data.loc[data["timestamp"] <= TRAIN_END]
    if training.empty:
        raise ValueError("The fixed training period is empty.")
    print("=" * 72)
    print("CALIBRATION AND HIGH-DEMAND EXPERT EXPERIMENT")
    print("=" * 72)
    print("Processed data shape:", data.shape)
    print("Feature columns:", len(feature_columns))
    print("Training rows through August:", f"{len(training):,}")
    print(
        "Leakage check: target/timestamp excluded; observed features are calendar fields "
        "and shifted lags/rolling means."
    )

    september_start, september_end, september_path = PERIODS["September"]
    september = load_aligned_predictions(
        data,
        "September",
        september_start,
        september_end,
        september_path,
    )
    october_start, october_end, october_path = PERIODS["October"]
    october = load_aligned_predictions(
        data,
        "October",
        october_start,
        october_end,
        october_path,
    )
    print("Verified saved September rows:", f"{len(september):,}")
    print("Verified saved October rows:", f"{len(october):,}")

    oof, sigmoid, isotonic = fit_time_aware_calibrators(
        data,
        feature_columns,
        classifier,
    )
    oof_events = oof["demand"].to_numpy() >= HIGH_DEMAND_THRESHOLD
    oof_metrics = []
    for method in ["raw_probability", "sigmoid_probability", "isotonic_probability"]:
        probability = oof[method].to_numpy(dtype=float)
        oof_metrics.append(
            {
                "Period": "July-August OOF calibration",
                "Probability method": method.replace("_probability", "").title(),
                "Brier score": float(brier_score_loss(oof_events, probability)),
                "Log loss": float(log_loss(oof_events, probability, labels=[0, 1])),
                "PR-AUC": float(average_precision_score(oof_events, probability)),
                "ROC-AUC": float(roc_auc_score(oof_events, probability)),
                "Observations": int(len(oof)),
            }
        )

    september_probabilities = calibrate_probability(
        september["high_demand_probability"].to_numpy(),
        sigmoid,
        isotonic,
    )
    october_probabilities = calibrate_probability(
        october["high_demand_probability"].to_numpy(),
        sigmoid,
        isotonic,
    )
    _, selected_thresholds = select_thresholds(september, september_probabilities)

    classifier_metric_rows = list(oof_metrics)
    for period, frame, probability_map in [
        ("September", september, september_probabilities),
        ("October", october, october_probabilities),
    ]:
        actual_events = frame["demand"].to_numpy() >= HIGH_DEMAND_THRESHOLD
        for method, probability in probability_map.items():
            threshold = selected_thresholds[method]["operational_threshold"]
            classifier_metric_rows.append(
                classifier_metrics(
                    period,
                    actual_events,
                    probability,
                    threshold,
                    method,
                )
            )
            frame[f"{method.lower()}_probability"] = probability
            frame[f"predicted_high_demand_{method.lower()}"] = (
                probability >= threshold
            ).astype(np.int8)
        plot_reliability(period, probability_map, actual_events)

    calibration_metrics = pd.DataFrame(classifier_metric_rows)
    calibration_metrics.to_csv(OUTPUT_DIR / "calibration_metrics.csv", index=False)

    print("\nCalibration metrics:")
    print(
        calibration_metrics[
            [
                "Period",
                "Probability method",
                "Brier score",
                "Log loss",
                "PR-AUC",
                "ROC-AUC",
            ]
        ].to_string(index=False)
    )
    print("\nSeptember-selected operational thresholds:")
    print(json.dumps(selected_thresholds, indent=2))

    variants, variant_settings, high_params = fit_high_expert_variants(
        data,
        feature_columns,
        high_model,
    )
    for period, frame in [("September", september), ("October", october)]:
        start, end, _ = PERIODS[period]
        period_feature_columns = [
            column for column in feature_columns if column not in IDENTIFIERS
        ]
        period_features = data.loc[
            (data["timestamp"] >= start) & (data["timestamp"] <= end),
            IDENTIFIERS + period_feature_columns,
        ].copy()
        frame = add_candidate_predictions(
            frame,
            period_features,
            feature_columns,
            variants,
            sigmoid,
            isotonic,
        )
        if period == "September":
            september = frame
        else:
            october = frame

    september_experts = evaluate_experts("September", september)
    october_experts = evaluate_experts("October", october)
    expert_metrics = pd.concat(
        [september_experts, october_experts],
        ignore_index=True,
    )
    expert_metrics.to_csv(OUTPUT_DIR / "high_demand_expert_metrics.csv", index=False)

    september_expert_selection = september_experts.sort_values(
        [
            "High-demand MAE (actual >=48)",
            "Extreme-demand MAE (actual >=74)",
        ],
        ascending=True,
    ).iloc[0]
    selected_expert = str(september_expert_selection["Expert"])
    selected_calibration = min(
        ["Sigmoid", "Isotonic"],
        key=lambda method: brier_score_loss(
            september["demand"].to_numpy() >= HIGH_DEMAND_THRESHOLD,
            september[f"{method.lower()}_probability"].to_numpy(),
        ),
    )
    print(f"\nSeptember-selected calibration method: {selected_calibration}")
    print(f"September-selected high-demand expert: {selected_expert}")

    september_comparison = evaluate_regressions(
        "September",
        september,
        selected_calibration,
    )
    october_comparison = evaluate_regressions(
        "October",
        october,
        selected_calibration,
    )
    model_comparison = pd.concat(
        [september_comparison, october_comparison],
        ignore_index=True,
    )
    model_comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    september.to_parquet(OUTPUT_DIR / "september_predictions.parquet", index=False)
    october.to_parquet(OUTPUT_DIR / "october_predictions.parquet", index=False)

    best_threshold = {
        method: {
            **values,
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
        }
        for method, values in selected_thresholds.items()
    }
    configuration = {
        "random_state": RANDOM_STATE,
        "demand_event_threshold": HIGH_DEMAND_THRESHOLD,
        "extreme_demand_threshold": EXTREME_DEMAND_THRESHOLD,
        "training_end": TRAIN_END.isoformat(),
        "calibration_folds": [
            {
                "training_end": fold["train_end"].isoformat(),
                "out_of_time_start": fold["calibration_start"].isoformat(),
                "out_of_time_end": fold["calibration_end"].isoformat(),
                "fold_only_scale_pos_weight": float(
                    oof.loc[oof["fold"] == fold_number, "fold_scale_pos_weight"].iloc[0]
                ),
            }
            for fold_number, fold in enumerate(CALIBRATION_FOLDS, start=1)
        ],
        "validation_period": "2022-09-01 through 2022-09-30",
        "final_test_period": "2022-10-01 through 2022-10-31",
        "threshold_grid": PROBABILITY_THRESHOLDS,
        "cost_threshold_candidates": COST_THRESHOLD_CANDIDATES,
        "bayes_cost_threshold_if_calibrated": (
            FALSE_POSITIVE_COST / (FALSE_POSITIVE_COST + FALSE_NEGATIVE_COST)
        ),
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "thresholds_selected_on_september": best_threshold,
        "calibration_method_selected_on_september": selected_calibration,
        "high_expert_selected_on_september": selected_expert,
        "base_classifier_parameters": classifier.named_steps[
            "classifier"
        ].get_params(),
        "base_high_expert_parameters": high_params,
        "high_expert_variants": variant_settings,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "scipy": scipy.__version__,
        },
        "notes": [
            "July and August calibration scores are generated out of time by expanding-window classifiers; each fold's class weight is computed from its training labels only.",
            "September selects calibration/expert variants and operational thresholds; October is final verification only.",
            "The original saved September/October predictions are used as aligned regression baselines.",
            "The extreme-weighted expert uses weight 3 for training demand >=74 and weight 1 otherwise within the >=48 subset.",
            "The median quantile regressor tests conditional-median fitting; it is not assumed to be superior for mean demand.",
            "False-negative cost 5 and false-positive cost 1 are illustrative relative operational costs, not empirically estimated fleet costs.",
        ],
    }
    with (OUTPUT_DIR / "experiment_configuration.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=2, default=str)

    print("\nSeptember model comparison:")
    print(september_comparison.to_string(index=False))
    print("\nOctober final model comparison:")
    print(october_comparison.to_string(index=False))
    print("\nSaved experiment artifacts to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
