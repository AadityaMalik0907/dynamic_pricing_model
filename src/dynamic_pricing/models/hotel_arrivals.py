"""Data preparation and leakage-safe features shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
