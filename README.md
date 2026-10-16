# Dynamic Pricing Optimization System

A Python project foundation for an advisory hotel room-night pricing demo. It will forecast demand and recommend prices from historical demand, inventory, seasonality, and optionally timestamped competitor prices. Recommendations are not automatically applied.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the roadmap and [configs/requirements.md](configs/requirements.md) for the initial scope, assumptions, and demo guardrails.
See [configs/data_contract.md](configs/data_contract.md) for accepted hotel/e-commerce CSV fields and the platform-neutral loader.
See [configs/feature_contract.md](configs/feature_contract.md) for feature meanings, temporal leakage rules, and analysis limits.

## Requirements

- Python 3.11, 3.12, or 3.13
- Git

## Local setup

From the repository root, create and activate a virtual environment, then install the project and development tools:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If Python 3.11 is unavailable, use `py -3.13` in the first command. The project currently allows Python 3.11 through 3.13.

## Run the development shells

```powershell
dynamic-pricing api
```

Open `http://127.0.0.1:8000/health` for process health and `http://127.0.0.1:8000/docs` for the API shell documentation. The health endpoint does not mean a model is trained or ready.

In a second terminal with the virtual environment activated:

```powershell
dynamic-pricing dashboard
```

The Streamlit page remains a placeholder; recommendations and KPI views arrive in later phases.

## Phase 2 data layer

The `dynamic_pricing.data` package adapts hotel and e-commerce source columns into the shared observation schema, validates rows, and provides deterministic synthetic demo observations. The loader accepts either a path or a file-like object, so the same code can be used with Streamlit uploads or another platform's request/storage layer. It does not persist uploads. See the data contract for required timestamp, provenance, and inventory semantics.

## Phase 3 analysis and features

Use `profile_observations(data)` and `render_profile_markdown(profile)` to generate descriptive data coverage, missingness, inventory saturation, and price-demand summaries. Use `DemandFeatureTransformer` fitted on training history to generate features for subsequent rows. Feature generation blocks current/future target outcomes from lag history and does not write platform-specific files. See the feature contract for timestamp and label assumptions.

## Phase 4 demand forecasting

Train and evaluate on deterministic synthetic hotel data:

```powershell
dynamic-pricing train-demo --domain hotel --days 120 --artifact-dir artifacts
```

The pipeline sorts by target date and makes non-overlapping chronological 70/15/15 train, validation, and test splits. Validation MAE selects between the same-weekday seasonal naive baseline, scikit-learn HistGradientBoosting, and XGBoost. The selected predictor is assessed on the later test period using MAE, RMSE, and WAPE. If an ML candidate does not beat the baseline on validation, the baseline is persisted; if it loses on test, the result includes a warning. Domain-specific joblib bundles and JSON metadata include the training window, split dates, metrics, model version, and feature schema. Loading checks artifact format, domain, and schema.

For canonical real data, call `train_forecaster(observations, domain="hotel", artifact_dir="artifacts")` after loading through the Phase 2 data adapters. Synthetic results only verify the software pipeline; they do not establish real forecast accuracy or revenue lift. Keep private data and generated artifacts out of Git.

### Train the first model on `hotel_bookings.csv`

The supplied hotel booking file can train a daily **completed reservation arrivals** model. Run this from the repository root, replacing the path if the file is stored elsewhere:

```powershell
dynamic-pricing train-hotel --data "C:\Users\aadit\OneDrive\Desktop\hotel_bookings.csv" --artifact-dir artifacts
```

This trains one scikit-learn HistGradientBoosting model using calendar features and prior daily arrivals, with date-ordered train/validation/test periods and a same-weekday naive baseline. It removes exact duplicate source rows, excludes canceled and no-show reservations, and writes a local model plus JSON metrics/metadata under `artifacts/`. The metadata contains the input file hash, split dates, overall and per-hotel test scores, and limitations.

The dataset does not contain inventory snapshots or a usable price history, so this model forecasts arrival counts only; it cannot estimate price response, available room nights, or revenue. The target counts reservation records, not rooms or room nights. The Amazon product/review file has no sales dates or units sold, so it cannot train a demand-forecast model. Do not combine these datasets into one training table.

## Configuration

Copy `.env.example` to `.env` and edit local values as needed. `.env` is ignored by Git. Defaults are demo assumptions documented in `configs/requirements.md`, not property-specific recommendations. Keep raw/private data and generated model artifacts out of version control.

## Current project state

Phases 0–4 provide documented assumptions, data and feature contracts, Python packaging, environment settings, API/dashboard shells, data validation/adapters, synthetic demo observations, profiling, leakage-aware features, and chronological demand forecasting. Price recommendation and model-backed API serving remain for later phases.
