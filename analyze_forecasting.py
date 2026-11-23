
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.joblib"
PREDICTIONS_PATH = BASE_DIR / "models" / "validation_predictions.parquet"

OUTPUT_DIR = BASE_DIR / "models" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SCATTER_SAMPLE_SIZE = 15000


# ============================================================
# LOAD DATA
# ============================================================

print("Loading model and validation predictions...")

model = joblib.load(MODEL_PATH)
predictions = pd.read_parquet(PREDICTIONS_PATH)

required_columns = {
    "city",
    "region_id",
    "timestamp",
    "demand",
    "naive_prediction",
    "seasonal_prediction",
    "xgboost_prediction"
}

missing = required_columns - set(predictions.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

predictions = predictions.copy()

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"]
)

predictions["hour"] = predictions["timestamp"].dt.hour

numeric_columns = [
    "demand",
    "naive_prediction",
    "seasonal_prediction",
    "xgboost_prediction"
]

for col in numeric_columns:
    predictions[col] = pd.to_numeric(
        predictions[col], errors="coerce"
    )

predictions = predictions.dropna(
    subset=numeric_columns + ["city", "region_id", "timestamp"]
)

# Clip predictions to avoid physically impossible negative demand.
for col in [
    "naive_prediction",
    "seasonal_prediction",
    "xgboost_prediction"
]:
    predictions[col] = predictions[col].clip(lower=0)

actual = predictions["demand"]
predicted = predictions["xgboost_prediction"]

print(f"Validation observations: {len(predictions):,}")
print(f"Date range: {predictions['timestamp'].min()} to "
      f"{predictions['timestamp'].max()}")


# ============================================================
# 1. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)
print("1. FEATURE IMPORTANCE")
print("=" * 65)

try:
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = xgb_model.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    feature_importance = feature_importance.sort_values(
        "importance", ascending=False
    )

    feature_importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    print("\nTop 20 features:")
    print(feature_importance.head(20).to_string(index=False))

    top_features = feature_importance.head(20).sort_values(
        "importance"
    )

    plt.figure(figsize=(12, 9))
    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 XGBoost Features")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=300
    )
    plt.close()

except Exception as e:
    print(f"Feature importance analysis failed: {e}")


# ============================================================
# 2. DEMAND CATEGORY ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("2. DEMAND CATEGORY ERROR ANALYSIS")
print("=" * 65)

positive_demand = actual[actual > 0]

if positive_demand.empty:
    raise ValueError("No positive demand observations found.")

q50 = positive_demand.quantile(0.50)
q75 = positive_demand.quantile(0.75)
q90 = positive_demand.quantile(0.90)
q95 = positive_demand.quantile(0.95)

print("\nPositive-demand thresholds:")
print(f"50th percentile: {q50:.2f}")
print(f"75th percentile: {q75:.2f}")
print(f"90th percentile: {q90:.2f}")
print(f"95th percentile: {q95:.2f}")


def categorize_demand(value):

    if value == 0:
        return "Zero demand"
    elif value <= q50:
        return "Low demand"
    elif value <= q75:
        return "Normal demand"
    elif value <= q90:
        return "High demand"
    else:
        return "Extreme demand"


predictions["demand_category"] = actual.apply(
    categorize_demand
)

predictions["absolute_error"] = (
    actual - predicted
).abs()

predictions["signed_error"] = (
    actual - predicted
)

category_order = [
    "Zero demand",
    "Low demand",
    "Normal demand",
    "High demand",
    "Extreme demand"
]

category_results = (
    predictions
    .groupby("demand_category")
    .agg(
        Observations=("demand", "count"),
        Mean_actual_demand=("demand", "mean"),
        Mean_prediction=("xgboost_prediction", "mean"),
        MAE=("absolute_error", "mean"),
        Mean_signed_error=("signed_error", "mean")
    )
    .reindex(category_order)
    .reset_index()
)

category_results.rename(
    columns={"demand_category": "Category"},
    inplace=True
)

category_results.to_csv(
    OUTPUT_DIR / "error_by_demand_category.csv",
    index=False
)

print("\nError by demand category:")
print(category_results.to_string(index=False))


# ============================================================
# 3. EXTREME DEMAND EVENT DETECTION
# ============================================================

print("\n" + "=" * 65)
print("3. EXTREME DEMAND EVENT ANALYSIS")
print("=" * 65)

# Diagnostic threshold: positive-demand 90th percentile.
extreme_threshold = q90

actual_extreme = actual >= extreme_threshold
predicted_extreme = predicted >= extreme_threshold

tp = (actual_extreme & predicted_extreme).sum()
fp = (~actual_extreme & predicted_extreme).sum()
fn = (actual_extreme & ~predicted_extreme).sum()

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)

extreme_mae = (
    mean_absolute_error(
        actual[actual_extreme],
        predicted[actual_extreme]
    )
    if actual_extreme.sum() > 0
    else np.nan
)

print(f"\nExtreme demand threshold: {extreme_threshold:.2f}")
print(f"Actual extreme events: {actual_extreme.sum():,}")
print(f"Predicted extreme events: {predicted_extreme.sum():,}")
print(f"True positives: {tp:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")
print(f"Extreme-event MAE: {extreme_mae:.4f}")

extreme_results = pd.DataFrame([{
    "threshold": extreme_threshold,
    "actual_extreme_events": int(actual_extreme.sum()),
    "predicted_extreme_events": int(predicted_extreme.sum()),
    "true_positive": int(tp),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "extreme_mae": extreme_mae
}])

extreme_results.to_csv(
    OUTPUT_DIR / "extreme_event_metrics.csv",
    index=False
)


# ============================================================
# 4. ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

print("\n" + "=" * 65)
print("4. ACTUAL VS PREDICTED")
print("=" * 65)

scatter_data = predictions.sample(
    min(len(predictions), SCATTER_SAMPLE_SIZE),
    random_state=RANDOM_STATE
)

max_value = max(
    scatter_data["demand"].max(),
    scatter_data["xgboost_prediction"].max()
)

plt.figure(figsize=(9, 8))

plt.scatter(
    scatter_data["demand"],
    scatter_data["xgboost_prediction"],
    alpha=0.25,
    s=12
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Demand - XGBoost")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300
)
plt.close()


# ============================================================
# 5. POSITIVE DEMAND SCATTER
# ============================================================

positive_predictions = predictions[
    predictions["demand"] > 0
]

positive_scatter = positive_predictions.sample(
    min(len(positive_predictions), SCATTER_SAMPLE_SIZE),
    random_state=RANDOM_STATE
)

if not positive_scatter.empty:

    max_value = max(
        positive_scatter["demand"].max(),
        positive_scatter["xgboost_prediction"].max()
    )

    plt.figure(figsize=(9, 8))

    plt.scatter(
        positive_scatter["demand"],
        positive_scatter["xgboost_prediction"],
        alpha=0.25,
        s=12
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Positive Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Actual vs Predicted - Positive Demand Only")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "positive_actual_vs_predicted.png",
        dpi=300
    )
    plt.close()


# ============================================================
# 6. CITY-WISE ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("5. CITY-WISE ERROR ANALYSIS")
print("=" * 65)

city_results = (
    predictions
    .groupby("city")
    .agg(
        Observations=("demand", "count"),
        Actual_mean=("demand", "mean"),
        Predicted_mean=("xgboost_prediction", "mean"),
        MAE=("absolute_error", "mean"),
        Mean_signed_error=("signed_error", "mean"),
        RMSE=(
            "signed_error",
            lambda x: np.sqrt(np.mean(x ** 2))
        )
    )
    .reset_index()
)

city_results.to_csv(
    OUTPUT_DIR / "error_by_city.csv",
    index=False
)

print(city_results.to_string(index=False))


# ============================================================
# 7. HOURLY ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("6. HOURLY ERROR ANALYSIS")
print("=" * 65)

hourly_results = (
    predictions
    .groupby("hour")
    .agg(
        actual_mean=("demand", "mean"),
        predicted_mean=("xgboost_prediction", "mean"),
        mae=("absolute_error", "mean")
    )
    .reset_index()
    .sort_values("hour")
)

hourly_results.to_csv(
    OUTPUT_DIR / "hourly_errors.csv",
    index=False
)

print(hourly_results.to_string(index=False))

plt.figure(figsize=(12, 6))

plt.plot(
    hourly_results["hour"],
    hourly_results["actual_mean"],
    marker="o",
    label="Actual demand"
)

plt.plot(
    hourly_results["hour"],
    hourly_results["predicted_mean"],
    marker="o",
    label="XGBoost prediction"
)

plt.xlabel("Hour of Day")
plt.ylabel("Mean Demand")
plt.title("Actual vs Predicted Average Hourly Demand")
plt.xticks(range(24))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "hourly_demand_comparison.png",
    dpi=300
)
plt.close()


# ============================================================
# 8. COMPARE ALL THREE FORECASTING MODELS
# ============================================================

print("\n" + "=" * 65)
print("7. MODEL COMPARISON")
print("=" * 65)

model_columns = {
    "Naive": "naive_prediction",
    "Seasonal Naive": "seasonal_prediction",
    "XGBoost": "xgboost_prediction"
}

comparison = []

for model_name, column in model_columns.items():

    y_pred = predictions[column]

    mae = mean_absolute_error(actual, y_pred)

    rmse = np.sqrt(
        mean_squared_error(actual, y_pred)
    )

    wape = (
        np.abs(actual - y_pred).sum()
        / actual.sum()
        * 100
        if actual.sum() > 0
        else np.nan
    )

    comparison.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape
    })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print(comparison_df.to_string(index=False))


# ============================================================
# 9. SAVE COMPLETE ANALYSIS DATA
# ============================================================

predictions.to_parquet(
    OUTPUT_DIR / "validation_predictions_analyzed.parquet",
    index=False
)

print("\n" + "=" * 65)
print("FORECAST ANALYSIS COMPLETE")
print("=" * 65)

print(f"\nAll outputs saved to:\n{OUTPUT_DIR}")

print("""
Generated files:

1. feature_importance.csv
2. feature_importance.png
3. error_by_demand_category.csv
4. extreme_event_metrics.csv
5. actual_vs_predicted.png
6. positive_actual_vs_predicted.png
7. error_by_city.csv
8. hourly_errors.csv
9. hourly_demand_comparison.png
10. model_comparison.csv
11. validation_predictions_analyzed.parquet
""")
