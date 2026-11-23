import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "lade_hourly_features.parquet"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = MODEL_DIR / "two_stage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2022-08-31 23:00:00")
SELECTION_START = pd.Timestamp("2022-09-01 00:00:00")
SELECTION_END = pd.Timestamp("2022-09-30 23:00:00")
FINAL_START = pd.Timestamp("2022-10-01 00:00:00")
FINAL_END = pd.Timestamp("2022-10-31 23:00:00")

HIGH_DEMAND_THRESHOLD = 48.0
EXTREME_DEMAND_THRESHOLD = 74.0
RANDOM_STATE = 42


def calculate_regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)

    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    wape = (
        float(np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100)
        if np.sum(actual) > 0
        else float("nan")
    )
    return {"MAE": mae, "RMSE": rmse, "WAPE (%)": wape}


def event_metrics(actual_events, predicted_events):
    precision = precision_score(actual_events, predicted_events, zero_division=0)
    recall = recall_score(actual_events, predicted_events, zero_division=0)
    f1 = f1_score(actual_events, predicted_events, zero_division=0)
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def make_preprocessor(categorical_features, numerical_features):
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numerical", "passthrough", numerical_features),
        ],
        remainder="drop",
    )


print("=" * 70)
print("TWO-STAGE LADE DEMAND FORECASTING")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

data = pd.read_parquet(DATA_PATH)
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values(["timestamp", "city", "region_id"]).reset_index(drop=True)

train = data[data["timestamp"] <= TRAIN_END].copy()
selection = data[
    (data["timestamp"] >= SELECTION_START)
    & (data["timestamp"] <= SELECTION_END)
].copy()
final = data[
    (data["timestamp"] >= FINAL_START)
    & (data["timestamp"] <= FINAL_END)
].copy()

for name, frame in [
    ("training", train),
    ("September selection", selection),
    ("October final", final),
]:
    print(f"{name} rows: {len(frame):,}")

if train.empty or selection.empty or final.empty:
    raise ValueError("Training, September selection or October final split is empty.")

excluded = {"demand", "timestamp", "ds"}
feature_columns = [column for column in data.columns if column not in excluded]
categorical_features = ["city", "region_id"]
numerical_features = [
    column for column in feature_columns if column not in categorical_features
]

X_train = train[feature_columns].copy()
y_train = train["demand"].copy()

for frame in (train, selection, final):
    for column in categorical_features:
        frame[column] = frame[column].astype(str)

for column in categorical_features:
    X_train[column] = X_train[column].astype(str)

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {categorical_features}")
print(f"Fixed high-demand threshold: {HIGH_DEMAND_THRESHOLD:.2f}")

train_high = (y_train >= HIGH_DEMAND_THRESHOLD).astype(int)
positive_count = int(train_high.sum())
negative_count = int(len(train_high) - positive_count)
scale_pos_weight = negative_count / positive_count if positive_count else 1.0

print(f"Training high-demand observations: {positive_count:,}")
print(f"Training normal-demand observations: {negative_count:,}")
print(f"Classifier scale_pos_weight: {scale_pos_weight:.4f}")

preprocessor = make_preprocessor(categorical_features, numerical_features)

classifier = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

low_mask = y_train < HIGH_DEMAND_THRESHOLD
high_mask = y_train >= HIGH_DEMAND_THRESHOLD

low_regressor = Pipeline(
    [
        (
            "preprocessor",
            make_preprocessor(categorical_features, numerical_features),
        ),
        (
            "regressor",
            XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

high_regressor = Pipeline(
    [
        (
            "preprocessor",
            make_preprocessor(categorical_features, numerical_features),
        ),
        (
            "regressor",
            XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

print("\nTraining high-demand classifier...")
classifier.fit(X_train, train_high)

print("Training low-demand regressor...")
low_regressor.fit(X_train.loc[low_mask], y_train.loc[low_mask])

print("Training high-demand regressor...")
high_regressor.fit(X_train.loc[high_mask], y_train.loc[high_mask])

joblib.dump(
    classifier,
    OUTPUT_DIR / "high_demand_classifier.joblib",
)
joblib.dump(
    low_regressor,
    OUTPUT_DIR / "low_demand_regressor.joblib",
)
joblib.dump(
    high_regressor,
    OUTPUT_DIR / "high_demand_regressor.joblib",
)


def generate_predictions(frame):
    X = frame[feature_columns].copy()
    for column in categorical_features:
        X[column] = X[column].astype(str)

    probability = classifier.predict_proba(X)[:, 1]
    low_prediction = np.maximum(low_regressor.predict(X), 0)
    high_prediction = np.maximum(high_regressor.predict(X), 0)

    combined = (1 - probability) * low_prediction + probability * high_prediction
    constrained_low = np.minimum(low_prediction, HIGH_DEMAND_THRESHOLD)
    constrained_high = np.maximum(high_prediction, HIGH_DEMAND_THRESHOLD)
    constrained_combined = (
        (1 - probability) * constrained_low
        + probability * constrained_high
    )

    result = frame[
        ["city", "region_id", "timestamp", "demand"]
    ].copy()
    result["high_demand_probability"] = probability
    result["low_prediction"] = low_prediction
    result["high_prediction"] = high_prediction
    result["combined_prediction"] = np.maximum(combined, 0)
    result["constrained_low_prediction"] = constrained_low
    result["constrained_high_prediction"] = constrained_high
    result["constrained_combined_prediction"] = np.maximum(
        constrained_combined,
        0,
    )

    actual = frame["demand"].to_numpy(dtype=float)
    result["naive_prediction"] = frame["lag_1"].to_numpy(dtype=float)
    result["seasonal_prediction"] = frame["lag_24"].to_numpy(dtype=float)

    original_model_path = MODEL_DIR / "xgboost_demand_model.joblib"
    weighted_model_path = MODEL_DIR / "xgboost_weighted_model.joblib"

    if original_model_path.exists():
        original_model = joblib.load(original_model_path)
        result["original_xgboost_prediction"] = np.maximum(
            original_model.predict(X),
            0,
        )

    if weighted_model_path.exists():
        weighted_model = joblib.load(weighted_model_path)
        result["weighted_xgboost_prediction"] = np.maximum(
            weighted_model.predict(X),
            0,
        )

    return result


selection_predictions = generate_predictions(selection)
final_predictions = generate_predictions(final)


def evaluate_period(name, predictions):
    actual = predictions["demand"].to_numpy(dtype=float)
    probability = predictions["high_demand_probability"].to_numpy(dtype=float)
    actual_events = actual >= HIGH_DEMAND_THRESHOLD

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    thresholds = np.arange(0.05, 0.96, 0.05)
    threshold_results = []
    for threshold in thresholds:
        predicted_events = probability >= threshold
        metrics = event_metrics(actual_events, predicted_events)
        threshold_results.append(
            {
                "Threshold": threshold,
                **metrics,
            }
        )

    threshold_table = pd.DataFrame(threshold_results)
    if name.startswith("September"):
        best_threshold_row = threshold_table.loc[
            threshold_table["F1"].idxmax()
        ]
        classification_threshold = float(best_threshold_row["Threshold"])
    else:
        # The threshold is selected on September only and then frozen for October.
        classification_threshold = float(SEPTEMBER_THRESHOLD)

    predicted_events = probability >= classification_threshold
    print(f"Classification threshold: {classification_threshold:.2f}")
    print(
        "Classifier confusion matrix:"
        f"\n{confusion_matrix(actual_events, predicted_events)}"
    )

    classifier_metrics = {
        "Accuracy": float(accuracy_score(actual_events, predicted_events)),
        "Precision": float(
            precision_score(actual_events, predicted_events, zero_division=0)
        ),
        "Recall": float(recall_score(actual_events, predicted_events, zero_division=0)),
        "F1": float(f1_score(actual_events, predicted_events, zero_division=0)),
        "PR-AUC": float(average_precision_score(actual_events, probability)),
    }
    try:
        classifier_metrics["ROC-AUC"] = float(
            roc_auc_score(actual_events, probability)
        )
    except ValueError:
        classifier_metrics["ROC-AUC"] = float("nan")

    print("Classifier metrics:")
    for metric, value in classifier_metrics.items():
        print(f"{metric}: {value:.4f}")

    predictions["predicted_high_demand"] = predicted_events.astype(int)

    model_columns = [
        ("Naive lag_1", "naive_prediction"),
        ("Seasonal naive lag_24", "seasonal_prediction"),
    ]
    if "original_xgboost_prediction" in predictions:
        model_columns.append(
            ("Original XGBoost", "original_xgboost_prediction")
        )
    if "weighted_xgboost_prediction" in predictions:
        model_columns.append(
            ("Weighted XGBoost", "weighted_xgboost_prediction")
        )
    model_columns.extend(
        [
            ("Two-stage unconstrained", "combined_prediction"),
            (
                "Two-stage constrained",
                "constrained_combined_prediction",
            ),
        ]
    )

    overall_rows = []
    high_rows = []
    extreme_rows = []
    expert_rows = []

    low_actual_mask = actual < HIGH_DEMAND_THRESHOLD
    high_actual_mask = actual >= HIGH_DEMAND_THRESHOLD
    extreme_actual_mask = actual >= EXTREME_DEMAND_THRESHOLD

    for model_name, column in model_columns:
        predicted = predictions[column].to_numpy(dtype=float)
        predicted = np.maximum(predicted, 0)
        overall_rows.append(
            {
                "Model": model_name,
                **calculate_regression_metrics(actual, predicted),
            }
        )

        high_rows.append(
            {
                "Model": model_name,
                "Threshold": HIGH_DEMAND_THRESHOLD,
                "Observations": int(high_actual_mask.sum()),
                "MAE": float(
                    mean_absolute_error(
                        actual[high_actual_mask],
                        predicted[high_actual_mask],
                    )
                ),
            }
        )
        extreme_rows.append(
            {
                "Model": model_name,
                "Threshold": EXTREME_DEMAND_THRESHOLD,
                "Observations": int(extreme_actual_mask.sum()),
                "MAE": float(
                    mean_absolute_error(
                        actual[extreme_actual_mask],
                        predicted[extreme_actual_mask],
                    )
                )
                if extreme_actual_mask.sum()
                else float("nan"),
                **event_metrics(
                    extreme_actual_mask,
                    predicted >= EXTREME_DEMAND_THRESHOLD,
                ),
            }
        )

    low_expert_actual = actual[low_actual_mask]
    low_expert_prediction = predictions["low_prediction"].to_numpy(dtype=float)[
        low_actual_mask
    ]
    high_expert_actual = actual[high_actual_mask]
    high_expert_prediction = predictions["high_prediction"].to_numpy(dtype=float)[
        high_actual_mask
    ]

    expert_rows.extend(
        [
            {
                "Expert": "Low-demand",
                "Actual regime observations": int(low_actual_mask.sum()),
                **calculate_regression_metrics(
                    low_expert_actual,
                    low_expert_prediction,
                ),
            },
            {
                "Expert": "High-demand",
                "Actual regime observations": int(high_actual_mask.sum()),
                **calculate_regression_metrics(
                    high_expert_actual,
                    high_expert_prediction,
                ),
            },
        ]
    )

    overall = pd.DataFrame(overall_rows)
    high_demand = pd.DataFrame(high_rows)
    extreme_events = pd.DataFrame(extreme_rows)
    experts = pd.DataFrame(expert_rows)

    print("\nOverall regression comparison:")
    print(overall.to_string(index=False))
    print("\nHigh-demand MAE:")
    print(high_demand.to_string(index=False))
    print("\nExtreme-event MAE and detection:")
    print(extreme_events.to_string(index=False))
    print("\nConditional expert evaluation:")
    print(experts.to_string(index=False))

    classifier_rows = [{"Period": name, **classifier_metrics}]
    if name.startswith("September"):
        threshold_table.to_csv(
            OUTPUT_DIR / "selection_classifier_thresholds.csv",
            index=False,
        )

    return {
        "classifier": pd.DataFrame(classifier_rows),
        "threshold_table": threshold_table,
        "overall": overall,
        "high_demand": high_demand,
        "extreme_events": extreme_events,
        "experts": experts,
    }


selection_evaluation = evaluate_period(
    "September model selection",
    selection_predictions,
)

SEPTEMBER_THRESHOLD = float(
    selection_evaluation["threshold_table"].loc[
        selection_evaluation["threshold_table"]["F1"].idxmax()
    ]["Threshold"]
)

final_evaluation = evaluate_period(
    "October final evaluation",
    final_predictions,
)

selection_predictions.to_parquet(
    OUTPUT_DIR / "validation_predictions.parquet",
    index=False,
)
final_predictions.to_parquet(
    OUTPUT_DIR / "october_predictions.parquet",
    index=False,
)

pd.concat(
    [
        selection_evaluation["classifier"],
        final_evaluation["classifier"],
    ],
    ignore_index=True,
).to_csv(
    OUTPUT_DIR / "classifier_metrics.csv",
    index=False,
)

selection_evaluation["overall"].assign(Period="September").to_csv(
    OUTPUT_DIR / "selection_model_comparison.csv",
    index=False,
)
final_evaluation["overall"].assign(Period="October").to_csv(
    OUTPUT_DIR / "october_model_comparison.csv",
    index=False,
)

pd.concat(
    [
        selection_evaluation["high_demand"].assign(Period="September"),
        final_evaluation["high_demand"].assign(Period="October"),
    ],
    ignore_index=True,
).to_csv(
    OUTPUT_DIR / "high_demand_mae.csv",
    index=False,
)

pd.concat(
    [
        selection_evaluation["extreme_events"].assign(Period="September"),
        final_evaluation["extreme_events"].assign(Period="October"),
    ],
    ignore_index=True,
).to_csv(
    OUTPUT_DIR / "extreme_event_metrics.csv",
    index=False,
)

pd.concat(
    [
        selection_evaluation["experts"].assign(Period="September"),
        final_evaluation["experts"].assign(Period="October"),
    ],
    ignore_index=True,
).to_csv(
    OUTPUT_DIR / "conditional_expert_metrics.csv",
    index=False,
)

pd.Series(
    {
        "selected_threshold": SEPTEMBER_THRESHOLD,
        "high_demand_threshold": HIGH_DEMAND_THRESHOLD,
        "extreme_demand_threshold": EXTREME_DEMAND_THRESHOLD,
    }
).to_csv(
    OUTPUT_DIR / "configuration.csv",
    header=["Value"],
)

print("\n" + "=" * 70)
print("TWO-STAGE EXPERIMENT COMPLETE")
print("=" * 70)
print(f"Models: {OUTPUT_DIR}")
print(f"September predictions: {OUTPUT_DIR / 'validation_predictions.parquet'}")
print(f"October predictions: {OUTPUT_DIR / 'october_predictions.parquet'}")
print(f"September threshold: {SEPTEMBER_THRESHOLD:.2f}")
