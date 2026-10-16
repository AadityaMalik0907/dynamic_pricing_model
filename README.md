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

The Streamlit page is currently a placeholder. Training, recommendation logic, and KPI views arrive in later phases.

## Phase 2 data layer

The `dynamic_pricing.data` package adapts hotel and e-commerce source columns into the shared observation schema, validates rows, and provides deterministic synthetic demo observations. The loader accepts either a path or a file-like object, so the same code can be used with Streamlit uploads or another platform's request/storage layer. It does not persist uploads. See the data contract for required timestamp, provenance, and inventory semantics.

## Phase 3 analysis and features

Use `profile_observations(data)` and `render_profile_markdown(profile)` to generate descriptive data coverage, missingness, inventory saturation, and price-demand summaries. Use `DemandFeatureTransformer` fitted on training history to generate features for subsequent rows. Feature generation blocks current/future target outcomes from lag history and does not write platform-specific files. See the feature contract for timestamp and label assumptions.

## Configuration

Copy `.env.example` to `.env` and edit local values as needed. `.env` is ignored by Git. Defaults are demo assumptions documented in `configs/requirements.md`, not property-specific recommendations. Keep raw/private data and generated model artifacts out of version control.

## Current project state

Phases 0–3 provide documented assumptions, data and feature contracts, Python packaging, environment settings, API/dashboard shells, data validation/adapters, synthetic demo observations, profiling, and leakage-aware features. There is no trained model or recommendation endpoint yet.
