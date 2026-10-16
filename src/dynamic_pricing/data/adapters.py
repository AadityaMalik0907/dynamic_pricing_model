"""Transform domain-specific source columns into canonical observations."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .validation import DataValidationError, validate_observations


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], domain: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataValidationError(
            [f"{domain} source is missing required column(s): {', '.join(missing)}"]
        )


def _require_identifiers(frame: pd.DataFrame, columns: tuple[str, ...], domain: str) -> None:
    issues = []
    for column in columns:
        invalid = frame[column].isna() | frame[column].astype("string").str.strip().eq("")
        for index in frame.index[invalid]:
            issues.append(f"row {index}: {domain} identifier '{column}' must be non-empty")
    if issues:
        raise DataValidationError(issues)


def _copy_optional(
    source: pd.DataFrame,
    target: pd.DataFrame,
    mapping: Mapping[str, str],
) -> None:
    for source_name, target_name in mapping.items():
        if source_name in source.columns:
            target[target_name] = source[source_name]


def adapt_hotel_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt property/room-night data to the shared observation contract.

    Required source columns: property_id, room_type_id, observed_at, stay_date,
    price, bookings, rooms_available, currency, and source. Optional columns:
    competitor_rate, competitor_source, event, and seasonality.
    """
    required = (
        "property_id",
        "room_type_id",
        "observed_at",
        "stay_date",
        "price",
        "bookings",
        "rooms_available",
        "currency",
        "source",
    )
    _require_columns(frame, required, "hotel")
    _require_identifiers(frame, ("property_id", "room_type_id"), "hotel")
    result = pd.DataFrame(index=frame.index)
    result["entity_id"] = (
        frame["property_id"].astype("string").str.strip()
        + "::"
        + frame["room_type_id"].astype("string").str.strip()
    )
    result["domain"] = "hotel"
    result["observed_at"] = frame["observed_at"]
    result["target_at"] = frame["stay_date"]
    result["price"] = frame["price"]
    result["demand"] = frame["bookings"]
    result["inventory"] = frame["rooms_available"]
    result["currency"] = frame["currency"]
    result["source"] = frame["source"]
    result["property_id"] = frame["property_id"]
    result["room_type_id"] = frame["room_type_id"]
    _copy_optional(
        frame,
        result,
        {
            "competitor_rate": "competitor_price",
            "competitor_source": "competitor_source",
            "event": "event",
            "seasonality": "seasonality",
        },
    )
    return validate_observations(result, expected_domain="hotel")


def adapt_ecommerce_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt product-day data to the shared observation contract.

    Required source columns: product_id, observed_at, target_at, price,
    units_sold, inventory, currency, and source. Optional columns:
    competitor_price, competitor_source, event, seasonality, and category.
    """
    required = (
        "product_id",
        "observed_at",
        "target_at",
        "price",
        "units_sold",
        "inventory",
        "currency",
        "source",
    )
    _require_columns(frame, required, "e-commerce")
    _require_identifiers(frame, ("product_id",), "e-commerce")
    result = pd.DataFrame(index=frame.index)
    result["entity_id"] = frame["product_id"]
    result["domain"] = "ecommerce"
    result["observed_at"] = frame["observed_at"]
    result["target_at"] = frame["target_at"]
    result["price"] = frame["price"]
    result["demand"] = frame["units_sold"]
    result["inventory"] = frame["inventory"]
    result["currency"] = frame["currency"]
    result["source"] = frame["source"]
    _copy_optional(
        frame,
        result,
        {
            "competitor_price": "competitor_price",
            "competitor_source": "competitor_source",
            "event": "event",
            "seasonality": "seasonality",
            "category": "product_category",
        },
    )
    return validate_observations(result, expected_domain="ecommerce")
