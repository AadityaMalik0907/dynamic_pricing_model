from io import StringIO

import pandas as pd
import pytest

from dynamic_pricing.data import (
    DataValidationError,
    adapt_ecommerce_observations,
    adapt_hotel_observations,
    load_observations_csv,
    make_demo_observations,
    validate_observations,
)


def test_hotel_demo_is_canonical_deterministic_and_labeled() -> None:
    first = make_demo_observations(domain="hotel", days=4, entities=2)
    second = make_demo_observations(domain="hotel", days=4, entities=2)

    assert len(first) == 8
    assert set(first["domain"]) == {"hotel"}
    assert set(first["source"]) == {"synthetic_demo"}
    pd.testing.assert_frame_equal(first, second)


def test_ecommerce_demo_passes_ecommerce_adapter() -> None:
    frame = make_demo_observations(domain="ecommerce", days=5, entities=1)

    assert len(frame) == 5
    assert set(frame["domain"]) == {"ecommerce"}
    assert frame["product_category"].eq("synthetic_demo_category").all()


def test_hotel_adapter_builds_stable_entity_id_and_preserves_input() -> None:
    source = pd.DataFrame(
        {
            "property_id": ["P1"],
            "room_type_id": ["KING"],
            "observed_at": ["2025-01-01T12:00:00+05:30"],
            "stay_date": ["2025-01-02"],
            "price": [7500],
            "bookings": [4],
            "rooms_available": [8],
            "currency": ["INR"],
            "source": ["example_source"],
        }
    )
    original = source.copy(deep=True)

    canonical = adapt_hotel_observations(source)

    assert canonical.loc[0, "entity_id"] == "P1::KING"
    assert canonical.loc[0, "observed_at"] == pd.Timestamp("2025-01-01T06:30:00Z")
    pd.testing.assert_frame_equal(source, original)


def test_ecommerce_adapter_maps_units_and_inventory() -> None:
    source = pd.DataFrame(
        {
            "product_id": ["SKU1"],
            "observed_at": ["2025-01-01T00:00:00Z"],
            "target_at": ["2025-01-01"],
            "price": [100.0],
            "units_sold": [2],
            "inventory": [10],
            "currency": ["INR"],
            "source": ["example_source"],
        }
    )

    canonical = adapt_ecommerce_observations(source)

    assert canonical.loc[0, "entity_id"] == "SKU1"
    assert canonical.loc[0, "demand"] == 2
    assert canonical.loc[0, "inventory"] == 10


def test_uploaded_csv_file_object_uses_same_contract() -> None:
    csv = StringIO(
        "product_id,observed_at,target_at,price,units_sold,inventory,currency,source\n"
        "SKU1,2025-01-01T00:00:00Z,2025-01-02,100,2,10,INR,catalog_demo\n"
    )

    result = load_observations_csv(csv, domain="ecommerce")

    assert len(result) == 1
    assert result.loc[0, "source"] == "catalog_demo"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("price", -1, "price.*greater than zero"),
        ("demand", -1, "demand.*cannot be negative"),
        ("inventory", float("inf"), "inventory.*finite"),
        ("currency", "inr", "currency.*uppercase"),
        ("observed_at", "2025-01-01T00:00:00", "observed_at.*timezone"),
    ],
)
def test_invalid_values_raise_actionable_error(field: str, value: object, message: str) -> None:
    frame = make_demo_observations(days=2, entities=1).iloc[[0]].copy()
    if field == "observed_at":
        # Preserve the input text so pandas does not silently coerce a naive
        # timestamp into the timezone-aware dtype before validation sees it.
        frame["observed_at"] = frame["observed_at"].astype(object)
    frame.loc[frame.index[0], field] = value

    with pytest.raises(DataValidationError, match=message):
        validate_observations(frame)


def test_competitor_price_requires_source_attribution() -> None:
    frame = make_demo_observations(days=2, entities=1).iloc[[0]].copy()
    frame.loc[frame.index[0], "competitor_price"] = 100.0
    frame.loc[frame.index[0], "competitor_source"] = None

    with pytest.raises(DataValidationError, match="competitor_price.*competitor_source"):
        validate_observations(frame)


def test_duplicate_observation_key_is_rejected() -> None:
    frame = make_demo_observations(days=2, entities=1).iloc[[0]].copy()
    duplicate = pd.concat([frame, frame], ignore_index=True)

    with pytest.raises(DataValidationError, match="duplicate observation key"):
        validate_observations(duplicate)


def test_date_bucket_allows_same_day_observation_but_timestamp_does_not() -> None:
    frame = make_demo_observations(days=2, entities=1).iloc[[0]].copy()
    frame.loc[frame.index[0], "observed_at"] = "2025-01-01T18:00:00Z"
    frame.loc[frame.index[0], "target_at"] = "2025-01-01"
    validate_observations(frame)

    frame.loc[frame.index[0], "target_at"] = "2025-01-01T17:00:00Z"
    with pytest.raises(DataValidationError, match="observed_at.*after.*target_at"):
        validate_observations(frame)


def test_domain_mismatch_is_rejected() -> None:
    frame = make_demo_observations(domain="hotel", days=2, entities=1)

    with pytest.raises(DataValidationError, match="requested"):
        validate_observations(frame, expected_domain="ecommerce")


def test_missing_source_identifier_is_rejected() -> None:
    source = pd.DataFrame(
        {
            "product_id": [None],
            "observed_at": ["2025-01-01T00:00:00Z"],
            "target_at": ["2025-01-02"],
            "price": [100],
            "units_sold": [1],
            "inventory": [5],
            "currency": ["INR"],
            "source": ["example_source"],
        }
    )

    with pytest.raises(DataValidationError, match="product_id.*non-empty"):
        adapt_ecommerce_observations(source)


def test_invalid_demo_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="days must be at least 2"):
        make_demo_observations(days=1)
