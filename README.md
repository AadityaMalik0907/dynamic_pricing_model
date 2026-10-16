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
python -m pip install -e ".[notebooks]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

If Python 3.11 is unavailable, use `py -3.13` in the first command. The project currently allows Python 3.11 through 3.13.

## Run the development shells

```powershell
dynamic-pricing api
```

Open `http://127.0.0.1:8000/health` for process health, `http://127.0.0.1:8000/ready` for notebook-artifact readiness, and `http://127.0.0.1:8000/docs` for the API. The forecast endpoint requires the exported model and 28 consecutive daily history rows.

In a second terminal with the virtual environment activated:

```powershell
dynamic-pricing dashboard
```

The Streamlit page reads the same exported model and processed daily history, then displays the next-day arrival forecast and holdout metrics.

## Phase 2 data layer

The `dynamic_pricing.data` package adapts hotel and e-commerce source columns into the shared observation schema, validates rows, and provides deterministic synthetic demo observations. The loader accepts either a path or a file-like object, so the same code can be used with Streamlit uploads or another platform's request/storage layer. It does not persist uploads. See the data contract for required timestamp, provenance, and inventory semantics.

## Phase 3 analysis and features

Use `profile_observations(data)` and `render_profile_markdown(profile)` to generate descriptive data coverage, missingness, inventory saturation, and price-demand summaries. Use `DemandFeatureTransformer` fitted on training history to generate features for subsequent rows. Feature generation blocks current/future target outcomes from lag history and does not write platform-specific files. See the feature contract for timestamp and label assumptions.

## Phase 4 model training notebook

Model experimentation and training happen in Jupyter, separate from API/dashboard inference. Install the notebook extra and launch the hotel training notebook:

```powershell
python -m pip install -e ".[notebooks]"
jupyter lab notebooks/01_hotel_arrivals_training.ipynb
```

The notebook compares Linear Regression, Random Forest, HistGradientBoosting, XGBoost, and a median reference. It uses chronological train/validation/test periods, selects by validation MAE, scores the later test period, and exports the selected pipeline plus metadata. The API and Streamlit dashboard load this artifact; they never fit a model.

The supplied source CSVs remain in their original locations:

- `C:\Users\aadit\OneDrive\Desktop\hotel_bookings.csv`
- `C:\Users\aadit\OneDrive\Desktop\amazon.csv`

After training, these local files are created (they are excluded from Git):

- `artifacts/models/hotel_arrivals_model.joblib` — fitted model pipeline
- `artifacts/models/hotel_arrivals_model.json` — selected model, split dates, metrics, and feature schema
- `artifacts/models/hotel_daily_arrivals.csv` — processed daily history for the dashboard

FastAPI exposes `GET /ready` and `POST /forecast/hotel-arrivals`; interactive API docs are at `/docs`. To deploy the model, provide the saved model and metadata at `MODEL_ARTIFACT_DIR` on the hosting platform.

The hotel data supports completed reservation-arrival counts only; it has no inventory snapshots or forecast-time prices, so it cannot estimate room-night demand, price response, or revenue. The Amazon file has no dated sales or units sold, so it cannot train an e-commerce demand model. We are not using the prior review-count popularity model for pricing. A proper e-commerce model needs time-stamped sales with units sold, price, and inventory.

## Configuration

Copy `.env.example` to `.env` and edit local values as needed. `.env` is ignored by Git. Defaults are demo assumptions documented in `configs/requirements.md`, not property-specific recommendations. Keep raw/private data and generated model artifacts out of version control.

## Current project state

Phases 0–4 provide documented assumptions, data and feature contracts, Python packaging, environment settings, API/dashboard inference shells, data validation/adapters, profiling, leakage-aware features, and notebook-trained hotel forecasting whose artifact is consumed by the API/dashboard. Price recommendation and e-commerce demand training remain for later phases and require suitable sales data.
