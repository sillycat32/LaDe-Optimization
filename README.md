# LaDe-D delivery demand and fleet prototype

This project is an engineering demonstration built on the processed LaDe-D
hourly demand data for Chongqing, Hangzhou, Shanghai and Yantai. It loads the
existing trained models; dashboard startup does not retrain them.

## What the prototype does

1. Loads the saved original XGBoost demand regressor and forecasts the next
   hour from the latest causal demand history.
2. Loads the saved high-demand classifier and sigmoid calibrator to estimate
   the probability that regional demand is at least 48.
3. Derives representative region coordinates as medians of valid raw delivery
   coordinates. Coordinate outliers are filtered only for spatial calculations;
   they are not removed from the demand data.
4. Builds a city-isolated k-nearest-neighbor graph and calculates an
   explainable neighbor-supported hotspot score.
5. Compares static, reactive and proactive greedy fleet allocations under the
   same fleet size and vehicle capacity.
6. Creates simulated, capacity-constrained delivery routes ordered by nearest
   neighbor.
7. Shows forecasts, hotspots, fleet scenarios, route maps and existing held-out
   model metrics in Streamlit.

The demand and route maps use Plotly's built-in geographic layer, so their
basemap does not depend on external map tiles. They show regional points and
simulated route geometry, not street-level roads.

## Windows setup

Use Python 3.12, which is compatible with the supplied model artifacts:

```powershell
Set-Location "D:\EL\AIML5\LaDe-D"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run pip and Streamlit by their
full paths instead:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\streamlit.exe" run app.py
```

## Launch

```powershell
Set-Location "D:\EL\AIML5\LaDe-D"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The app loads these existing artifacts:

- `models/xgboost_demand_model.joblib` — primary demand forecast.
- `models/xgboost_weighted_model.joblib` — comparison forecast.
- `models/two_stage/high_demand_classifier.joblib` — classifier for demand
  `>=48`.
- `models/calibration_experiment/sigmoid_calibrator.joblib` — sigmoid
  probability calibration.

The application checks the saved feature schemas and sends each model its own
recorded input-column order. Missing or incompatible artifacts raise an
explicit error.

## Forecast and replay semantics

- **Next-hour forecast:** uses demand observations through the latest available
  hour and predicts the following hour. Actual target demand is unavailable.
- **Historical replay:** the selected timestamp is the information cutoff.
  The application predicts after that cutoff. Actual demand at the target hour
  is attached only after prediction for evaluation; it is not used as a model
  input, allocation input or route-planning input.
- **Planning horizons beyond one hour:** recursively insert model predictions
  into subsequent lag and rolling features. This is a scenario simulation, not
  a validated multi-hour forecast; forecast error accumulates.

The high-demand probability control starts at 0.90 to match the supplied
operational default. The sigmoid calibrator changes probability scale; the
recent calibration experiment selected cost-based cutoffs near 0.17–0.20.
Change the control to match the intended false-alarm/missed-demand costs.

## Spatial scoring

The graph uses Haversine distances and is constructed separately within each
city. The initial score is:

```text
0.50 * normalized local predicted demand
+ 0.30 * calibrated high-demand probability
+ 0.20 * neighbor support
```

Neighbor support combines normalized neighbor demand and neighbor probability.
A high-demand hotspot requires both a high score and sufficient neighbor
support, so a high local prediction alone is not enough. These weights and
classification thresholds are engineering defaults, not scientifically
optimized parameters.

Region coordinates are cached at
`outputs/geo/region_coordinates.csv`. They are robust medians from valid raw
`lng`/`lat` points. Invalid GPS records do not remove demand observations.

## Fleet and route simulation

- **Static:** distributes vehicles evenly across regions.
- **Reactive:** uses the latest demand observed at the as-of time.
- **Proactive:** uses predicted demand with an adjustable hotspot-score bonus.
- Vehicles are assigned greedily, with a capacity limit and a maximum regional
  vehicle need based on planning demand.
- Simulated vehicles start at the region where they are allocated. Delivery
  tasks are assigned to available capacity and ordered with a nearest-neighbor
  heuristic.
- The route distance fallback is **Haversine straight-line distance**, not road
  distance. The displayed travel time assumes a configurable prototype speed
  and excludes traffic, service time, time windows and road restrictions. The
  system does not claim to solve a full VRPTW.
- No live OSMnx road network, depot locations, courier GPS positions or
  real-time orders are used.

Static, reactive and proactive demand coverage is evaluated on the same target
demand scenario when historical actuals exist. Future forecast coverage is
explicitly a simulation estimate. Hotspot precision/recall/F1 in historical
replay use same-hour actual demand `>=48` as a proxy label; this is not a
validated future hotspot-emergence experiment.

## Outputs

Generated files are stored under `outputs/`:

- `predictions/current_forecast.parquet`
- `geo/region_coordinates.csv`
- `hotspots/current_hotspots.csv`
- `fleet/current_allocations.csv`
- `fleet/strategy_comparison.csv`
- `routes/current_routes.json`
- `routes/static_routes.json`, `routes/reactive_routes.json`,
  `routes/proactive_routes.json`
- `routes/strategy_comparison.csv`
- `evaluation/current_metrics.json`

## Tests

Run the end-to-end component tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests load the saved models, verify feature compatibility, exercise latest
forecast and historical replay, validate geographic graph isolation, ensure
fleet and route capacity constraints, and check evaluation outputs.

## Limitations

- LaDe-D contains Chinese delivery data and does not establish performance on
  other countries or operators.
- The saved primary forecaster was evaluated on September/October historical
  periods; a dashboard forecast is not a live operational guarantee.
- Recursive forecasts beyond one hour are unvalidated and accumulate error.
- Demand-based allocation, hotspot weights, simulated depot starts, vehicle
  capacities and route speed require real operational data before deployment.
- Straight-line routing can substantially understate actual road distance and
  travel time. Route results are simulated and should not be described as
  real-world fleet improvement.
- Current hotspot scoring predicts same-hour demand concentration. Future
  spatial hotspot emergence is a separate task and is not claimed as validated
  here.
