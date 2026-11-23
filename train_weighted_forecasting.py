
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "processed" / "lade_hourly_features.parquet"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT = MODEL_DIR / "xgboost_weighted_model.joblib"
PREDICTION_OUTPUT = MODEL_DIR / "weighted_validation_predictions.parquet"
METRICS_OUTPUT = MODEL_DIR / "weighted_model_comparison.csv"

TRAIN_END = "2022-08-31 23:00:00"
VALIDATION_START = "2022-09-01 00:00:00"
VALIDATION_END = "2022-09-30 23:00:00"

# Weighting configuration
DEMAND_Q75_WEIGHT = 2.0
DEMAND_Q90_WEIGHT = 4.0
DEMAND_Q95_WEIGHT = 6.0


# ============================================================
# LOAD DATA
# ============================================================

print("Loading processed dataset...")

df = pd.read_parquet(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Dataset shape:", df.shape)


# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

target_column = "demand"

categorical_features = [
    "city",
    "region_id"
]

excluded_columns = [
    target_column,
    "timestamp",
    "ds"
]

numerical_features = [
    col for col in df.columns
    if col not in categorical_features + excluded_columns
]

feature_columns = categorical_features + numerical_features

X = df[feature_columns]
y = df[target_column]


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_mask = df["timestamp"] <= TRAIN_END

validation_mask = (
    (df["timestamp"] >= VALIDATION_START)
    & (df["timestamp"] <= VALIDATION_END)
)

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[validation_mask]
y_val = y.loc[validation_mask]

validation_data = df.loc[validation_mask].copy()

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))

if len(X_train) == 0 or len(X_val) == 0:
    raise ValueError("Training or validation split is empty.")


# ============================================================
# CALCULATE TRAINING-ONLY DEMAND THRESHOLDS
# ============================================================

print("\nCalculating training demand thresholds...")

positive_train = y_train[y_train > 0]

if positive_train.empty:
    raise ValueError("Training set contains no positive demand.")

q75 = positive_train.quantile(0.75)
q90 = positive_train.quantile(0.90)
q95 = positive_train.quantile(0.95)

print(f"75th percentile: {q75:.2f}")
print(f"90th percentile: {q90:.2f}")
print(f"95th percentile: {q95:.2f}")


# ============================================================
# GENERATE SAMPLE WEIGHTS
# ============================================================

sample_weights = np.ones(len(y_train), dtype=float)

sample_weights[y_train.to_numpy() >= q75] = DEMAND_Q75_WEIGHT
sample_weights[y_train.to_numpy() >= q90] = DEMAND_Q90_WEIGHT
sample_weights[y_train.to_numpy() >= q95] = DEMAND_Q95_WEIGHT

print("\nTraining sample weights:")
print(pd.Series(sample_weights).value_counts().sort_index())


# ============================================================
# PREPROCESSING
# ============================================================

print("\nBuilding preprocessing pipeline...")

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_transformer, categorical_features),
    ("numerical", numerical_transformer, numerical_features)
])


# ============================================================
# XGBOOST MODEL
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", xgb_model)
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining weighted XGBoost model...")

pipeline.fit(
    X_train,
    y_train,
    regressor__sample_weight=sample_weights
)

print("Training complete.")


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating validation predictions...")

weighted_predictions = pipeline.predict(X_val)

weighted_predictions = np.clip(
    weighted_predictions,
    0,
    None
)

validation_data["weighted_xgboost_prediction"] = weighted_predictions

# Preserve the original XGBoost validation predictions
# if they exist in the previous validation output.
old_prediction_path = MODEL_DIR / "validation_predictions.parquet"

if old_prediction_path.exists():

    old_predictions = pd.read_parquet(old_prediction_path)

    old_predictions = old_predictions[
        [
            "city",
            "region_id",
            "timestamp",
            "xgboost_prediction"
        ]
    ]

    validation_data = validation_data.merge(
        old_predictions,
        on=["city", "region_id", "timestamp"],
        how="left"
    )


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating models...")

actual = y_val.to_numpy()

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    wape = (
        np.abs(y_true - y_pred).sum()
        / np.abs(y_true).sum()
        * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape
    }


results = []

# Existing baseline
if "xgboost_prediction" in validation_data.columns:

    baseline = validation_data["xgboost_prediction"].to_numpy()

    results.append({
        "Model": "Original XGBoost",
        **calculate_metrics(actual, baseline)
    })

# Weighted model
results.append({
    "Model": "Weighted XGBoost",
    **calculate_metrics(actual, weighted_predictions)
})

# Seasonal naive baseline
if "seasonal_prediction" in validation_data.columns:

    results.append({
        "Model": "Seasonal Naive",
        **calculate_metrics(
            actual,
            validation_data["seasonal_prediction"].to_numpy()
        )
    })

metrics_df = pd.DataFrame(results)

print("\nOverall comparison:")
print(metrics_df.to_string(index=False))

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


# ============================================================
# HIGH-DEMAND EVALUATION
# ============================================================


# ============================================================
# HIGH-DEMAND EVALUATION
# ============================================================

print("\nHigh-demand evaluation:")

evaluation_thresholds = {
    "High demand": q90,
    "Extreme demand": q95
}

# Convert predictions into aligned NumPy arrays.
weighted_pred = np.asarray(weighted_predictions)

if "xgboost_prediction" in validation_data.columns:
    original_pred = validation_data[
        "xgboost_prediction"
    ].to_numpy()
else:
    original_pred = None

for category, threshold in evaluation_thresholds.items():

    mask = actual >= threshold

    print(f"\n{category}: threshold >= {threshold:.2f}")
    print(f"Observations: {mask.sum():,}")

    if mask.sum() == 0:
        continue

    if original_pred is not None:

        original_mae = mean_absolute_error(
            actual[mask],
            original_pred[mask]
        )

        print(f"Original XGBoost MAE: {original_mae:.4f}")

    weighted_mae = mean_absolute_error(
        actual[mask],
        weighted_pred[mask]
    )

    print(f"Weighted XGBoost MAE: {weighted_mae:.4f}")


# ============================================================
# EXTREME EVENT DETECTION
# ============================================================

print("\nExtreme event detection:")

threshold = q95

actual_events = actual >= threshold

models_to_evaluate = {
    "Weighted XGBoost": weighted_pred
}

if original_pred is not None:
    models_to_evaluate["Original XGBoost"] = original_pred

for model_name, prediction in models_to_evaluate.items():

    predicted_events = prediction >= threshold

    tp = np.sum(actual_events & predicted_events)
    fp = np.sum(~actual_events & predicted_events)
    fn = np.sum(actual_events & ~predicted_events)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print(f"\n{model_name}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")



# ============================================================
# EXTREME EVENT DETECTION
# ============================================================

print("\nExtreme event detection:")

threshold = q95

actual_events = actual >= threshold

for model_name, prediction in [
    ("Weighted XGBoost", weighted_predictions)
]:

    predicted_events = prediction >= threshold

    tp = np.sum(actual_events & predicted_events)
    fp = np.sum(~actual_events & predicted_events)
    fn = np.sum(actual_events & ~predicted_events)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    print(f"\n{model_name}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


# ============================================================
# SAVE MODEL AND PREDICTIONS
# ============================================================

print("\nSaving model and predictions...")

joblib.dump(
    pipeline,
    MODEL_OUTPUT
)

validation_data.to_parquet(
    PREDICTION_OUTPUT,
    index=False
)

print("\n" + "=" * 65)
print("WEIGHTED XGBOOST EXPERIMENT COMPLETE")
print("=" * 65)

print(f"Model: {MODEL_OUTPUT}")
print(f"Predictions: {PREDICTION_OUTPUT}")
print(f"Metrics: {METRICS_OUTPUT}")

