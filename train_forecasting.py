
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "lade_hourly_features.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "demand"

TRAIN_END = pd.Timestamp("2022-08-31 23:00:00")
VALIDATION_START = pd.Timestamp("2022-09-01 00:00:00")
VALIDATION_END = pd.Timestamp("2022-09-30 23:00:00")

CATEGORICAL_FEATURES = [
    "city",
    "region_id"
]

# ==========================================================
# STEP 1: LOAD DATA
# ==========================================================

print("=" * 70)
print("LADE DEMAND FORECASTING")
print("=" * 70)

print("\nLoading processed dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_FILE}"
    )

data = pd.read_parquet(DATA_FILE)

data["timestamp"] = pd.to_datetime(data["timestamp"])

data = data.sort_values(
    ["timestamp", "city", "region_id"]
).reset_index(drop=True)

print("Dataset shape:", data.shape)

print(
    "Date range:",
    data["timestamp"].min(),
    "to",
    data["timestamp"].max()
)

# ==========================================================
# STEP 2: CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ==========================================================

print("\nCreating chronological split...")

train = data[
    data["timestamp"] <= TRAIN_END
].copy()

validation = data[
    (data["timestamp"] >= VALIDATION_START) &
    (data["timestamp"] <= VALIDATION_END)
].copy()

print("Training rows:", len(train))
print("Validation rows:", len(validation))

print(
    "Training period:",
    train["timestamp"].min(),
    "to",
    train["timestamp"].max()
)

print(
    "Validation period:",
    validation["timestamp"].min(),
    "to",
    validation["timestamp"].max()
)

if train.empty or validation.empty:
    raise ValueError("Training or validation split is empty.")

# ==========================================================
# STEP 3: SELECT FEATURES
# ==========================================================

# Target and timestamp are not model inputs.
# Timestamp is used only to create the chronological split.
#
# city and region_id are categorical features.
# All other columns are numerical features.

excluded_columns = [
    TARGET,
    "timestamp"
]

FEATURES = [
    column
    for column in data.columns
    if column not in excluded_columns
]

X_train = train[FEATURES].copy()
y_train = train[TARGET].copy()

X_val = validation[FEATURES].copy()
y_val = validation[TARGET].copy()

for column in CATEGORICAL_FEATURES:
    X_train[column] = X_train[column].astype(str)
    X_val[column] = X_val[column].astype(str)

NUMERICAL_FEATURES = [
    column
    for column in FEATURES
    if column not in CATEGORICAL_FEATURES
]

print("\nNumerical features:", len(NUMERICAL_FEATURES))
print("Categorical features:", CATEGORICAL_FEATURES)

print("\nFeatures:")
print(FEATURES)

# ==========================================================
# STEP 4: EVALUATION FUNCTIONS
# ==========================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = np.sum(actual)

    if total_actual > 0:
        wape = (
            np.sum(np.abs(actual - predicted))
            / total_actual
        ) * 100
    else:
        wape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE (%)": wape
    }


def evaluate_model(name, actual, predicted):

    metrics = calculate_metrics(
        actual,
        predicted
    )

    print(f"\n{name}")
    print("-" * 50)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def evaluate_by_city(name, frame, predictions):

    results = []

    for city in sorted(frame["city"].unique()):

        mask = (
            frame["city"].to_numpy() == city
        )

        metrics = calculate_metrics(
            frame.loc[mask, TARGET],
            predictions[mask]
        )

        results.append({
            "Model": name,
            "City": city,
            **metrics
        })

    return results


# ==========================================================
# STEP 5: BASELINE 1 - PREVIOUS HOUR
# ==========================================================

print("\n" + "=" * 70)
print("BASELINE 1: PREVIOUS HOUR")
print("=" * 70)

# Predict current demand using the previous hour's demand.

naive_predictions = validation["lag_1"].to_numpy()

naive_metrics = evaluate_model(
    "Naive (lag_1)",
    y_val,
    naive_predictions
)

# ==========================================================
# STEP 6: BASELINE 2 - PREVIOUS DAY
# ==========================================================

print("\n" + "=" * 70)
print("BASELINE 2: PREVIOUS DAY")
print("=" * 70)

# Predict current demand using demand at the same hour
# on the previous day.

seasonal_predictions = validation["lag_24"].to_numpy()

seasonal_metrics = evaluate_model(
    "Seasonal Naive (lag_24)",
    y_val,
    seasonal_predictions
)

# ==========================================================
# STEP 7: XGBOOST PREPROCESSING
# ==========================================================

print("\n" + "=" * 70)
print("BUILDING XGBOOST MODEL")
print("=" * 70)

# One-hot encode city and region identifiers.
# Numerical features pass through unchanged.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            "passthrough",
            NUMERICAL_FEATURES
        )
    ],
    remainder="drop"
)

# ==========================================================
# STEP 8: XGBOOST MODEL
# ==========================================================

xgb_model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,

    tree_method="hist",
    n_jobs=-1,

    random_state=42,
    eval_metric="rmse"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_model)
    ]
)

# ==========================================================
# STEP 9: TRAIN
# ==========================================================

print("\nTraining XGBoost...")
print("This may take a while.")

model.fit(
    X_train,
    y_train
)

print("\nTraining completed.")

# ==========================================================
# STEP 10: VALIDATION PREDICTIONS
# ==========================================================

print("\nGenerating validation predictions...")

xgb_predictions = model.predict(X_val)

# Demand cannot be negative.
xgb_predictions = np.maximum(
    xgb_predictions,
    0
)

xgb_metrics = evaluate_model(
    "XGBoost",
    y_val,
    xgb_predictions
)

# ==========================================================
# STEP 11: PER-CITY EVALUATION
# ==========================================================

print("\n" + "=" * 70)
print("PER-CITY VALIDATION RESULTS")
print("=" * 70)

city_results = []

city_results.extend(
    evaluate_by_city(
        "Naive",
        validation,
        naive_predictions
    )
)

city_results.extend(
    evaluate_by_city(
        "Seasonal Naive",
        validation,
        seasonal_predictions
    )
)

city_results.extend(
    evaluate_by_city(
        "XGBoost",
        validation,
        xgb_predictions
    )
)

city_results_df = pd.DataFrame(city_results)

print(city_results_df.to_string(index=False))

city_results_df.to_csv(
    OUTPUT_DIR / "validation_city_metrics.csv",
    index=False
)

# ==========================================================
# STEP 12: COMPARE MODELS
# ==========================================================

print("\n" + "=" * 70)
print("OVERALL MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame([
    {
        "Model": "Naive (lag_1)",
        **naive_metrics
    },
    {
        "Model": "Seasonal Naive (lag_24)",
        **seasonal_metrics
    },
    {
        "Model": "XGBoost",
        **xgb_metrics
    }
])

print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "validation_model_comparison.csv",
    index=False
)

# ==========================================================
# STEP 13: SAVE MODEL
# ==========================================================

model_path = OUTPUT_DIR / "xgboost_demand_model.joblib"

joblib.dump(
    model,
    model_path
)

# ==========================================================
# STEP 14: SAVE VALIDATION PREDICTIONS
# ==========================================================

prediction_output = validation[
    ["city", "region_id", "timestamp", "demand"]
].copy()

prediction_output["naive_prediction"] = (
    naive_predictions
)

prediction_output["seasonal_prediction"] = (
    seasonal_predictions
)

prediction_output["xgboost_prediction"] = (
    xgb_predictions
)

prediction_output.to_parquet(
    OUTPUT_DIR / "validation_predictions.parquet",
    index=False
)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("FORECASTING BENCHMARK COMPLETE")
print("=" * 70)

print("\nSaved files:")

print("1.", model_path)

print(
    "2.",
    OUTPUT_DIR / "validation_model_comparison.csv"
)

print(
    "3.",
    OUTPUT_DIR / "validation_city_metrics.csv"
)

print(
    "4.",
    OUTPUT_DIR / "validation_predictions.parquet"
)

print("\nDone.")
