import json

import pandas as pd
import pytest

from dynamic_pricing.models import (
    build_arrival_features,
    build_daily_hotel_arrivals,
    train_hotel_arrivals,
)


def _booking_rows(days: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=days, freq="D")
    rows = []
    for hotel_number, hotel in enumerate(("City Hotel", "Resort Hotel")):
        for date_number, arrival in enumerate(dates):
            arrivals = 2 + (date_number % 7 == 5) + hotel_number
            for reservation in range(int(arrivals)):
                rows.append(
                    {
                        "hotel": hotel,
                        "is_canceled": 0,
                        "arrival_date_year": arrival.year,
                        "arrival_date_month": arrival.strftime("%B"),
                        "arrival_date_day_of_month": arrival.day,
                        "reservation_status": "Check-Out",
                        "reservation_status_date": f"{arrival.date()}-{reservation}",
                    }
                )
    rows.extend(
        [
            {
                "hotel": "City Hotel",
                "is_canceled": 1,
                "arrival_date_year": 2016,
                "arrival_date_month": "January",
                "arrival_date_day_of_month": 1,
                "reservation_status": "Canceled",
                "reservation_status_date": "cancelled",
            },
            {
                "hotel": "City Hotel",
                "is_canceled": 1,
                "arrival_date_year": 2016,
                "arrival_date_month": "January",
                "arrival_date_day_of_month": 1,
                "reservation_status": "No-Show",
                "reservation_status_date": "no-show",
            },
        ]
    )
    return pd.DataFrame(rows)


def test_daily_arrivals_deduplicate_and_exclude_unrealized_bookings() -> None:
    bookings = _booking_rows(days=30)
    duplicate = pd.concat([bookings, bookings.iloc[[0]]], ignore_index=True)

    daily, counts = build_daily_hotel_arrivals(duplicate)

    assert counts["exact_duplicate_rows_removed"] == 1
    assert counts["canceled_or_no_show_rows_excluded"] == 2
    assert daily["arrivals"].sum() == len(bookings) - 2
    assert set(daily["hotel"]) == {"City Hotel", "Resort Hotel"}


def test_lagged_features_do_not_depend_on_future_arrivals() -> None:
    daily, _ = build_daily_hotel_arrivals(_booking_rows(days=35))
    changed = daily.copy()
    last_date = changed["arrival_at"].max()
    changed.loc[changed["arrival_at"].eq(last_date), "arrivals"] += 100

    original_features = build_arrival_features(daily)
    changed_features = build_arrival_features(changed)
    earlier = original_features["arrival_at"].lt(last_date)
    pd.testing.assert_frame_equal(
        original_features.loc[earlier].reset_index(drop=True),
        changed_features.loc[earlier].reset_index(drop=True),
    )


def test_user_dataset_training_writes_chronological_metrics_and_artifact(tmp_path) -> None:
    source = tmp_path / "hotel_bookings.csv"
    _booking_rows().to_csv(source, index=False)

    result = train_hotel_arrivals(source, artifact_dir=tmp_path / "artifacts")

    assert result.row_counts["train"] > result.row_counts["validation"]
    assert result.row_counts["test"] > 0
    assert result.date_ranges["train_end"] < result.date_ranges["validation_start"]
    assert result.date_ranges["validation_end"] < result.date_ranges["test_start"]
    assert set(result.segment_metrics) == {"City Hotel", "Resort Hotel"}
    assert (tmp_path / "artifacts" / "hotel-arrivals-hist-gradient-boosting.joblib").is_file()
    metadata = json.loads(
        (tmp_path / "artifacts" / "hotel-arrivals-hist-gradient-boosting.json").read_text()
    )
    assert metadata["target_unit"] == "reservation arrivals (not room nights)"
    assert metadata["dataset_sha256"]


def test_daily_arrival_builder_rejects_inconsistent_booking_outcomes() -> None:
    booking = _booking_rows(days=30)
    booking.loc[0, "is_canceled"] = 1

    with pytest.raises(ValueError, match="disagree"):
        build_daily_hotel_arrivals(booking)
