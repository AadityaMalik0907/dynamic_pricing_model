# Dynamic Pricing Optimization System — Project Plan

## Goal

Build a Python system that forecasts demand and recommends prices for hotel rooms and e-commerce products using historical demand, inventory, seasonality, and competitor prices. The system should provide auditable recommendations, a Streamlit KPI dashboard, and FastAPI endpoints.

Hotels and e-commerce have different inventory and booking behavior. The implementation will use a shared feature and recommendation pipeline with domain-specific data validation and feature preparation; it will not assume that one model trained on one domain transfers to the other. Keep domain identity explicit (`hotel` or `ecommerce`) throughout the data and API.

## Proposed stack

- Python 3.11+
- pandas, NumPy, scikit-learn, XGBoost
- Pydantic for schemas and settings
- FastAPI and Uvicorn for serving
- Streamlit and Plotly for the dashboard
- pytest for unit and integration tests
- Ruff for linting and formatting; mypy where practical
- Joblib for versioned model artifacts
- SQLite for a local demo, behind a repository interface that can later use PostgreSQL
- GitHub Actions for CI

Pin compatible dependency versions in `pyproject.toml` or a lock file once implementation begins. Do not commit secrets, private customer data, or unlicensed competitor data.

## Target repository layout

```text
.
├── src/dynamic_pricing/
│   ├── api/                 # FastAPI routes and request/response schemas
│   ├── dashboard/           # Streamlit application
│   ├── data/                # Loading, validation, and domain adapters
│   ├── features/            # Leakage-safe feature construction
│   ├── models/              # Training, evaluation, and model registry/loading
│   ├── pricing/             # Constraints, objective, and recommendation logic
│   ├── storage/             # Persistence interfaces and local implementation
│   └── settings.py
├── configs/                 # Requirements and runtime configuration
├── data/                    # .gitkeep only; raw data stays out of Git
├── artifacts/               # ignored model outputs
├── tests/
├── scripts/
├── .github/workflows/
├── .env.example
├── .gitignore
├── pyproject.toml           # Exact direct dependency pins
├── README.md
└── PROJECT_PLAN.md
```

## Phases and completion criteria

### Phase 0 — Requirements and operating assumptions

**Work:** Define the first demonstrable use case, supported price currency and cadence, recommendation consumers, price floor/ceiling, maximum permitted price movement, and how often data and models update. Identify the permitted data sources and their terms. Decide whether hotel and e-commerce demos need separate datasets and models.

**Done when:** A short requirements/config document records these choices, known limitations, and the difference between a recommendation and an automatically applied price. Initial operation should be advisory-only.

### Phase 1 — Repository and environment foundation

**Work:** Create the package structure above, Python environment setup, pinned dependencies, configuration handling, structured logging, `.gitignore`, `.env.example`, and a README with reproducible setup commands. Add CI to install dependencies and run quality checks.

**Done when:** A clean checkout can install the project and run its CLI/API/dashboard entry points in documented development mode; CI checks pass. No credentials or generated data/artifacts are tracked.

### Phase 2 — Data contracts and data quality

**Work:** Define canonical observation fields: entity ID, domain, observation timestamp, target date/time bucket, price, units/bookings or demand, available inventory/capacity, and optional competitor price and event/seasonality fields. Add domain adapters (for example, room-night and product-day) and validation for types, units, duplicates, missingness, impossible values, and timestamp ordering. Provide clearly labeled synthetic demo data if licensed or user-provided data is unavailable.

**Done when:** Valid sample data loads for each supported domain; invalid data produces actionable errors; provenance, units, and limitations are documented. Synthetic data is never described as observed market data. The shared loader accepts both local paths and in-memory file objects so the data layer is independent of a hosting platform.

### Phase 3 — Exploratory analysis and leakage-safe features

**Work:** Profile distributions, coverage, missing values, seasonality, inventory censoring, and price/demand relationships. Build reusable feature transformations for calendar effects, lagged demand, rolling statistics, inventory pressure, and competitor-price differences where available. Fit transformations only on training data and ensure every feature is knowable at recommendation time.

**Done when:** A reproducible analysis report describes data limitations and baselines; feature generation has tests for time ordering, missing data, and leakage; training and serving use the same transformations. The feature contract distinguishes realized demand labels from snapshot booking pace.

### Phase 4 — Demand forecasting models

**Work:** Establish a naive seasonal baseline and a scikit-learn baseline, then evaluate XGBoost as an additional model when data volume and quality justify it. Use chronological train/validation/test splits or rolling-origin evaluation, not random splits. Measure MAE/RMSE and an appropriate scaled metric; break results down by domain and relevant segments. Persist model, feature schema, training window, metrics, and version metadata together.

**Done when:** The selected model beats or has a documented reason not to beat the baseline on held-out time periods; evaluation is reproducible; model artifacts load with schema/version checks. Forecast accuracy is not presented as proof of revenue lift.

### Phase 5 — Price recommendation and offline evaluation

**Work:** Define a transparent objective, such as expected revenue = candidate price × predicted units sold, with optional documented costs or margin constraints. Search a bounded candidate-price grid and enforce configured floor, ceiling, inventory, and maximum-change rules. Include a safe fallback when inputs/models are unavailable. Evaluate against the current-price and simple rule baselines using historical replay, while clearly documenting that replay is not a causal experiment and historical demand may be censored by stockouts.

**Done when:** Recommendations include the chosen price, expected demand/revenue estimate, input/model version, constraints applied, and a human-readable reason. Invalid inputs and out-of-range outputs are rejected. The objective and offline evaluation limitations are documented.

### Phase 6 — FastAPI service

**Work:** Add health/readiness and version endpoints plus a validated recommendation endpoint. Use explicit request/response schemas, bounded payloads, useful error responses, model loading at startup, and structured logs without sensitive data. Keep serving code separate from training code.

**Done when:** API documentation describes fields and examples; endpoint behavior is covered by automated tests; startup fails clearly when required artifacts/configuration are invalid; the service can return a safe error/fallback for unavailable model state.

### Phase 7 — Streamlit KPI dashboard

**Work:** Display data coverage, historical and forecast demand, current versus recommended prices, expected revenue estimates, model metrics/version, and recommendation constraints. Add domain/date/entity filters and distinguish actuals, forecasts, and estimates visually. Explain missing competitor data and other caveats.

**Done when:** Dashboard works against the documented local data/API setup, handles empty or invalid filters cleanly, and never labels estimates as realized revenue. The core recommendation can be inspected and traced to its inputs/model version.

### Phase 8 — Integration, release, and operations

**Work:** Add end-to-end checks, dependency/security review, reproducible training instructions, sample configuration, API/dashboard run instructions, model refresh and rollback steps, monitoring guidance for input drift and forecast errors, and a deployment guide (for example, Docker only if selected). Add CI gates for tests, linting, and build/import checks.

**Done when:** A reviewer can follow README instructions from a clean environment to load demo data, train or load a model, request a recommendation, and view the dashboard. CI is green; known risks, data rights, model limitations, and operational steps are documented. Production deployment and automatic price changes require a separately reviewed rollout and monitoring plan.

## Cross-cutting quality rules

- Keep training, evaluation, inference, API, and UI responsibilities separate.
- Use chronological evaluation and prevent target leakage at every split.
- Validate all data at system boundaries; log errors with context but never secrets.
- Make recommendations reproducible from input snapshot, model version, configuration, and code version.
- Treat competitor prices as optional, timestamped, source-attributed data with freshness checks.
- Store raw/private data and model artifacts outside version control unless explicitly approved and safe to publish.
- Make each phase's completion criteria pass before building dependent phases.

## First implementation milestone

Start with one domain and one well-documented dataset, complete Phases 0–5, and expose recommendations in a local demo. Add the second domain after its schema, data quality, and evaluation are understood. This reduces the chance of combining incompatible hotel and retail behavior into an untrustworthy model while preserving a shared system architecture.
