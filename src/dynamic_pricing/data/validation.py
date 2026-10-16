"""Canonical observation schema and boundary validation."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import date, datetime

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "entity_id",
    "domain",
    "observed_at",
    "target_at",
    "price",
    "demand",
    "inventory",
    "currency",
    "source",
)
OPTIONAL_COLUMNS = (
    "competitor_price",
    "competitor_source",
    "event",
    "seasonality",
    "property_id",
    "room_type_id",
    "product_category",
)
CANONICAL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
ALLOWED_DOMAINS = {"hotel", "ecommerce"}
_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


class DataValidationError(ValueError):
    """Raised when observation rows do not satisfy the shared data contract."""

    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(issues)
        details = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Observation data is invalid:\n{details}")


def _number(value: object, *, row: object, column: str, issues: list[str]) -> float | None:
    if pd.isna(value):
        issues.append(f"row {row}: '{column}' is required")
        return None
    if isinstance(value, bool | np.bool_):
        issues.append(f"row {row}: '{column}' must be numeric")
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        issues.append(f"row {row}: '{column}' must be numeric")
        return None
    if not math.isfinite(result):
        issues.append(f"row {row}: '{column}' must be finite")
        return None
    return result


def _timestamp(
    value: object,
    *,
    row: object,
    column: str,
    issues: list[str],
    allow_date_only: bool,
) -> pd.Timestamp | None:
    if pd.isna(value):
        issues.append(f"row {row}: '{column}' is required")
        return None
    if not isinstance(value, str | date | datetime | pd.Timestamp):
        issues.append(f"row {row}: '{column}' must be an ISO-8601 date or timestamp")
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        issues.append(f"row {row}: '{column}' must be an ISO-8601 date or timestamp")
        return None
    if pd.isna(parsed):
        issues.append(f"row {row}: '{column}' must be a valid date or timestamp")
        return None
    if parsed.tzinfo is None:
        if column == "observed_at" or not allow_date_only:
            issues.append(f"row {row}: '{column}' must include a timezone, such as +00:00 or Z")
            return None
        # A date-only target is a calendar bucket. Normalize it to midnight UTC.
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def _is_date_only(value: object) -> bool:
    if isinstance(value, pd.Timestamp):
        return value == value.normalize()
    if isinstance(value, date) and not isinstance(value, datetime):
        return True
    if isinstance(value, str):
        return len(value.strip()) == 10 and value.strip()[4:5] == "-" and value.strip()[7:8] == "-"
    return False


def validate_observations(
    frame: pd.DataFrame,
    *,
    expected_domain: str | None = None,
) -> pd.DataFrame:
    """Validate and normalize canonical rows, returning a defensive copy.

    `observed_at` must be timezone-aware. Date-only `target_at` values are
    accepted as calendar buckets and normalized to midnight UTC. Full target
    timestamps must also include an offset. Date-only target buckets are
    ordered by UTC calendar date; timestamp targets are ordered by instant.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise DataValidationError(["dataset must contain at least one observation row"])
    if expected_domain is not None and expected_domain not in ALLOWED_DOMAINS:
        raise ValueError(f"expected_domain must be one of {sorted(ALLOWED_DOMAINS)}")

    issues: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError([f"missing required column(s): {', '.join(missing)}"])

    result = frame.copy(deep=True).reset_index(drop=True)
    # Normalize text fields without mutating the caller's DataFrame.
    for column in ("entity_id", "domain", "currency", "source"):
        result[column] = result[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    for column in ("competitor_source", "event", "seasonality"):
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    for index, row in result.iterrows():
        label = index
        for column in ("entity_id", "source"):
            value = row[column]
            if not isinstance(value, str) or not value:
                issues.append(f"row {label}: '{column}' must be a non-empty string")

        domain = row["domain"]
        if domain not in ALLOWED_DOMAINS:
            issues.append(f"row {label}: 'domain' must be 'hotel' or 'ecommerce'")
        elif expected_domain is not None and domain != expected_domain:
            issues.append(
                f"row {label}: domain is '{domain}', but '{expected_domain}' was requested"
            )

        currency = row["currency"]
        if not isinstance(currency, str) or not _CURRENCY_PATTERN.fullmatch(currency):
            issues.append(f"row {label}: 'currency' must be a three-letter uppercase code (for example INR)")

        observed_at = _timestamp(
            row["observed_at"],
            row=label,
            column="observed_at",
            issues=issues,
            allow_date_only=False,
        )
        target_at = _timestamp(
            row["target_at"],
            row=label,
            column="target_at",
            issues=issues,
            allow_date_only=True,
        )

        price = _number(row["price"], row=label, column="price", issues=issues)
        demand = _number(row["demand"], row=label, column="demand", issues=issues)
        inventory = _number(row["inventory"], row=label, column="inventory", issues=issues)
        competitor_value = row.get("competitor_price")
        competitor_price = None
        if "competitor_price" in result.columns and not pd.isna(competitor_value):
            competitor_price = _number(
                competitor_value, row=label, column="competitor_price", issues=issues
            )

        for column, value in (("price", price), ("competitor_price", competitor_price)):
            if value is not None and value <= 0:
                issues.append(f"row {label}: '{column}' must be greater than zero")
        for column, value in (("demand", demand), ("inventory", inventory)):
            if value is not None and value < 0:
                issues.append(f"row {label}: '{column}' cannot be negative")

        if observed_at is not None:
            result.at[index, "observed_at"] = observed_at
        if target_at is not None:
            result.at[index, "target_at"] = target_at
            is_date_bucket = _is_date_only(row["target_at"])
            is_after_target = False
            if observed_at is not None:
                is_after_target = (
                    observed_at.date() > target_at.date()
                    if is_date_bucket
                    else observed_at > target_at
                )
            if is_after_target:
                issues.append(f"row {label}: 'observed_at' cannot be after 'target_at'")
        for column, value in (
            ("price", price),
            ("demand", demand),
            ("inventory", inventory),
        ):
            if value is not None:
                result.at[index, column] = value
        if "competitor_price" in result.columns:
            result.at[index, "competitor_price"] = competitor_price

        for optional in ("competitor_source", "event", "seasonality"):
            if optional in result.columns and pd.notna(row.get(optional)):
                if not isinstance(row[optional], str) or not row[optional].strip():
                    issues.append(f"row {label}: '{optional}' must be a non-empty string when provided")
                elif optional == "competitor_source" and competitor_price is None:
                    issues.append(
                        f"row {label}: '{optional}' requires a non-empty 'competitor_price'"
                    )
        if competitor_price is not None and (
            "competitor_source" not in result.columns
            or pd.isna(row.get("competitor_source"))
            or not str(row.get("competitor_source")).strip()
        ):
            issues.append(
                f"row {label}: 'competitor_price' requires a 'competitor_source' attribution"
            )

    duplicate_key = ["entity_id", "observed_at", "target_at"]
    duplicate_mask = result.duplicated(subset=duplicate_key, keep=False)
    if duplicate_mask.any():
        duplicate_rows = ", ".join(str(index) for index in result.index[duplicate_mask].tolist())
        issues.append(f"duplicate observation key ({', '.join(duplicate_key)}) at row(s): {duplicate_rows}")

    if issues:
        raise DataValidationError(issues)

    result["domain"] = result["domain"].astype("string")
    result["entity_id"] = result["entity_id"].astype("string")
    result["currency"] = result["currency"].astype("string")
    result["source"] = result["source"].astype("string")
    result["price"] = result["price"].astype(float)
    result["demand"] = result["demand"].astype(float)
    result["inventory"] = result["inventory"].astype(float)
    result["observed_at"] = pd.to_datetime(result["observed_at"], utc=True)
    result["target_at"] = pd.to_datetime(result["target_at"], utc=True)
    return result
