"""Train one hotel arrivals forecaster from the supplied booking-level CSV."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

_MONTHS = {
    month: number
    for number, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
_CATEGORICAL_FEATURES = ["hotel"]
_NUMERIC_FEATURES = [
    "day_of_week",
    "month",
    "is_weekend",
    "year_sin",
    "year_cos",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "rolling_mean_7",
    "rolling_mean_28",
]
_FEATURE_SCHEMA = [*_CATEGORICAL_FEATURES, *_NUMERIC_FEATURES]


@dataclass(frozen=True)
class HotelArrivalsTrainingResult:
    """Summary for the supplied hotel bookings training run."""

    artifact_path: str
    metadata_path: str
    row_counts: dict[str, int]
    date_ranges: dict[str, str]
    model_metrics: dict[str, float | None]
    seasonal_naive_metrics: dict[str, float | None]
    model_beats_baseline: bool
    segment_metrics: dict[str, dict[str, float | None]]
    warnings: tuple[str, ...]


def build_daily_hotel_arrivals(bookings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate unique completed hotel reservations into daily arrival counts.

    Exact duplicate source rows are removed because this dataset has no booking
    identifier and exact duplicate records are known to inflate row counts.
    Canceled and no-show reservations are not realized arrivals.
    """
    required = {
        "hotel",
        "is_canceled",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
        "reservation_status",
    }
    missing = sorted(required.difference(bookings.columns))
    if missing:
        raise ValueError(f"hotel booking data is missing required column(s): {', '.join(missing)}")
    if bookings.empty:
        raise ValueError("hotel booking data contains no rows")

    source = bookings.copy()
    original_rows = len(source)
    source = source.drop_duplicates().copy()
    exact_duplicates = original_rows - len(source)
    month = source["arrival_date_month"].astype("string").str.strip().str.title().map(_MONTHS)
    arrival = pd.to_datetime(
        {
            "year": pd.to_numeric(source["arrival_date_year"], errors="coerce"),
            "month": month,
            "day": pd.to_numeric(source["arrival_date_day_of_month"], errors="coerce"),
        },
        errors="coerce",
    )
    if arrival.isna().any():
        raise ValueError(f"{int(arrival.isna().sum())} row(s) have invalid arrival dates")
    if source["hotel"].isna().any() or source["hotel"].astype(str).str.strip().eq("").any():
        raise ValueError("hotel column must contain a non-empty value for every row")

    canceled = pd.to_numeric(source["is_canceled"], errors="coerce")
    if canceled.isna().any() or not canceled.isin([0, 1]).all():
        raise ValueError("is_canceled must contain only 0 or 1")
    completed = source["reservation_status"].astype("string").str.strip().eq("Check-Out")
    if not (completed == canceled.eq(0)).all():
        raise ValueError(
            "reservation_status and is_canceled disagree; resolve booking outcomes before training"
        )

    source["arrival_at"] = arrival
    source["hotel"] = source["hotel"].astype("string").str.strip()
    completed_rows = source.loc[completed, ["hotel", "arrival_at"]]
    observed_counts = (
        completed_rows.groupby(["hotel", "arrival_at"], observed=True)
        .size()
        .rename("arrivals")
        .reset_index()
    )
    all_dates = pd.date_range(
        observed_counts["arrival_at"].min(), observed_counts["arrival_at"].max(), freq="D"
    )
    hotels = sorted(source["hotel"].unique().tolist())
    grid = pd.MultiIndex.from_product([hotels, all_dates], names=["hotel", "arrival_at"])
    daily = (
        observed_counts.set_index(["hotel", "arrival_at"]).reindex(grid, fill_value=0).reset_index()
    )
    daily["arrivals"] = daily["arrivals"].astype(int)
    counts = {
        "source_rows": original_rows,
        "exact_duplicate_rows_removed": exact_duplicates,
        "unique_source_rows": len(source),
        "completed_reservation_rows": int(completed.sum()),
        "canceled_or_no_show_rows_excluded": int((~completed).sum()),
        "daily_hotel_date_rows": len(daily),
        "hotels": len(hotels),
    }
    return daily, counts


def build_arrival_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build lagged predictors using prior dates only, separately by hotel."""
    required = {"hotel", "arrival_at", "arrivals"}
    if not required.issubset(daily.columns):
        raise ValueError("daily data must contain hotel, arrival_at, and arrivals")
    result = daily.sort_values(["hotel", "arrival_at"], kind="mergesort").copy()
    result["arrival_at"] = pd.to_datetime(result["arrival_at"])
    day_of_year = result["arrival_at"].dt.dayofyear
    angle = 2 * np.pi * (day_of_year - 1) / 365.2425
    result["day_of_week"] = result["arrival_at"].dt.dayofweek
    result["month"] = result["arrival_at"].dt.month
    result["is_weekend"] = result["day_of_week"].ge(5).astype(int)
    result["year_sin"] = np.sin(angle)
    result["year_cos"] = np.cos(angle)
    by_hotel = result.groupby("hotel", observed=True)["arrivals"]
    result["demand_lag_1"] = by_hotel.shift(1)
    result["demand_lag_7"] = by_hotel.shift(7)
    result["demand_lag_14"] = by_hotel.shift(14)
    result["rolling_mean_7"] = by_hotel.transform(lambda values: values.shift(1).rolling(7).mean())
    result["rolling_mean_28"] = by_hotel.transform(
        lambda values: values.shift(1).rolling(28).mean()
    )
    return result


def _regressor() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), _NUMERIC_FEATURES),
            (
                "hotel",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                _CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=150,
                    learning_rate=0.06,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | None]:
    truth = actual.to_numpy(dtype=float)
    total = float(np.abs(truth).sum())
    return {
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "wape": float(np.abs(truth - predicted).sum() / total) if total else None,
    }


def _seasonal_naive(features: pd.DataFrame) -> np.ndarray:
    lag_7 = pd.to_numeric(features["demand_lag_7"], errors="coerce")
    lag_1 = pd.to_numeric(features["demand_lag_1"], errors="coerce")
    return lag_7.fillna(lag_1).fillna(0).to_numpy(dtype=float)


def train_hotel_arrivals(
    csv_path: str | Path,
    *,
    artifact_dir: str | Path = "artifacts",
) -> HotelArrivalsTrainingResult:
    """Train one scikit-learn model and evaluate once on later arrival dates."""
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"hotel booking CSV does not exist: {csv_file}")
    source_hash = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    bookings = pd.read_csv(csv_file)
    daily, source_counts = build_daily_hotel_arrivals(bookings)
    featured = build_arrival_features(daily)
    dates = np.array(sorted(featured["arrival_at"].unique()))
    if len(dates) < 60:
        raise ValueError(
            "at least 60 distinct arrival dates are required for chronological evaluation"
        )
    train_end = int(len(dates) * 0.70)
    validation_end = int(len(dates) * 0.85)
    train = featured.loc[featured["arrival_at"].isin(dates[:train_end])]
    validation = featured.loc[featured["arrival_at"].isin(dates[train_end:validation_end])]
    test = featured.loc[featured["arrival_at"].isin(dates[validation_end:])]
    train_model = _regressor()
    train_model.fit(train[_FEATURE_SCHEMA], train["arrivals"])
    validation_prediction = np.maximum(train_model.predict(validation[_FEATURE_SCHEMA]), 0.0)
    validation_baseline = _seasonal_naive(validation[_FEATURE_SCHEMA])
    # Refit the same model on train + validation after model selection is fixed.
    fit_rows = pd.concat([train, validation], ignore_index=True)
    final_model = _regressor().fit(fit_rows[_FEATURE_SCHEMA], fit_rows["arrivals"])
    test_prediction = np.maximum(final_model.predict(test[_FEATURE_SCHEMA]), 0.0)
    test_baseline = _seasonal_naive(test[_FEATURE_SCHEMA])
    model_metrics = _metrics(test["arrivals"], test_prediction)
    baseline_metrics = _metrics(test["arrivals"], test_baseline)
    segment_metrics = {
        str(hotel): _metrics(
            segment["arrivals"],
            np.maximum(final_model.predict(segment[_FEATURE_SCHEMA]), 0.0),
        )
        for hotel, segment in test.groupby("hotel", observed=True)
    }

    warnings: list[str] = []
    if (
        _metrics(validation["arrivals"], validation_prediction)["mae"]
        >= _metrics(validation["arrivals"], validation_baseline)["mae"]
    ):
        warnings.append("The ML model did not beat the seasonal naive baseline on validation MAE.")
    beats_baseline = model_metrics["mae"] < baseline_metrics["mae"]
    if not beats_baseline:
        warnings.append(
            "The ML model did not beat the seasonal naive baseline on held-out test MAE."
        )

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "hotel-arrivals-hist-gradient-boosting.joblib"
    metadata_path = output / "hotel-arrivals-hist-gradient-boosting.json"
    joblib.dump(final_model, model_path)
    split = {
        "train_start": str(pd.Timestamp(dates[0]).date()),
        "train_end": str(pd.Timestamp(dates[train_end - 1]).date()),
        "validation_start": str(pd.Timestamp(dates[train_end]).date()),
        "validation_end": str(pd.Timestamp(dates[validation_end - 1]).date()),
        "test_start": str(pd.Timestamp(dates[validation_end]).date()),
        "test_end": str(pd.Timestamp(dates[-1]).date()),
    }
    metadata: dict[str, Any] = {
        "artifact_format_version": 1,
        "model_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "model": "scikit-learn HistGradientBoostingRegressor",
        "target": "completed reservation arrivals per hotel per calendar day",
        "target_unit": "reservation arrivals (not room nights)",
        "feature_schema": _FEATURE_SCHEMA,
        "chronological_split": split,
        "row_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            **source_counts,
        },
        "validation_metrics": {
            "model": _metrics(validation["arrivals"], validation_prediction),
            "seasonal_naive": _metrics(validation["arrivals"], validation_baseline),
        },
        "test_metrics": {"model": model_metrics, "seasonal_naive": baseline_metrics},
        "test_metrics_by_hotel": segment_metrics,
        "model_beats_baseline_test_mae": beats_baseline,
        "dataset_file": csv_file.name,
        "dataset_sha256": source_hash,
        "excluded_outcomes": ["Canceled", "No-Show"],
        "dropped_exact_duplicate_rows": source_counts["exact_duplicate_rows_removed"],
        "warnings": warnings,
        "limitations": [
            "No inventory or room-capacity snapshots are present.",
            "No price-at-forecast-time history is present; this model does not estimate "
            "price elasticity.",
            "Prediction is a one-day-ahead count of completed booking records, not room nights "
            "or revenue.",
            "Forecasts use prior observed daily arrivals; future-day features require refreshed "
            "actuals each day.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return HotelArrivalsTrainingResult(
        artifact_path=str(model_path),
        metadata_path=str(metadata_path),
        row_counts={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            **source_counts,
        },
        date_ranges=split,
        model_metrics=model_metrics,
        seasonal_naive_metrics=baseline_metrics,
        model_beats_baseline=beats_baseline,
        segment_metrics=segment_metrics,
        warnings=tuple(warnings),
    )
