"""Leakage-aware calendar, inventory, competition, and demand-history features."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .data.validation import DataValidationError, validate_observations

_OUTPUT_COLUMNS = (
    "entity_id",
    "domain",
    "currency",
    "target_day_of_week",
    "target_month",
    "target_day_of_year",
    "target_is_weekend",
    "target_week_sin",
    "target_week_cos",
    "target_year_sin",
    "target_year_cos",
    "price",
    "inventory",
    "inventory_pressure",
    "competitor_price_relative",
    "competitor_price_missing",
    "demand_lag_1",
    "demand_rolling_mean_7",
    "event",
    "seasonality",
    "product_category",
)


def _bucket_end(target_at: pd.Timestamp) -> pd.Timestamp:
    """Return when a target bucket's outcome can first be considered mature."""
    if target_at.time() == time.min:
        # Date-only buckets normalize to midnight UTC; the full calendar day
        # must elapse before its outcome is available as historical demand.
        return target_at + pd.Timedelta(days=1)
    return target_at


class DemandFeatureTransformer(TransformerMixin, BaseEstimator):
    """Build deterministic features from canonical observations.

    Fit this transformer on the training history, then transform subsequent
    observations with the same instance. Lag features only use outcomes from
    target buckets that had closed by each row's `observed_at`; future, current,
    and after-snapshot records cannot contribute. For inference, fit using the
    latest durable history before transforming new rows; this class does not
    persist or mutate history itself.

    Optional `event`, `seasonality`, and `product_category` values pass through
    as categorical features. Callers must supply these only when known at the
    observation time. Inventory pressure is a monotonic scarcity proxy because
    the data contract has no separate total-capacity field.
    """

    def __init__(self, domain: str | None = None) -> None:
        self.domain = domain

    def fit(self, X: pd.DataFrame, y: object = None) -> DemandFeatureTransformer:
        """Store validated training observations as the allowed feature history."""
        history = validate_observations(X, expected_domain=self.domain)
        self.history_ = history.copy(deep=True)
        self.feature_names_in_ = np.asarray(history.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return model-ready feature columns, excluding target labels/timestamps."""
        check_is_fitted(self, "history_")
        current = validate_observations(X, expected_domain=self.domain)
        # Include the transform batch as candidate context so later rows can
        # use earlier mature outcomes from that batch. Per-row checks below
        # prevent future or current target labels from entering a feature row.
        context = pd.concat([self.history_, current], ignore_index=True)
        history_by_entity = {
            entity_id: group
            for entity_id, group in context.groupby("entity_id", sort=False)
        }

        output_rows: list[dict[str, object]] = []
        for _, row in current.iterrows():
            target = row["target_at"]
            weekday = int(target.dayofweek)
            week_angle = 2 * np.pi * weekday / 7
            year_angle = 2 * np.pi * (int(target.dayofyear) - 1) / 365.2425
            competitor = row.get("competitor_price", np.nan)
            competitor_missing = pd.isna(competitor)
            competitor_relative = np.nan
            if not competitor_missing:
                price_value = float(row["price"])
                competitor_value = float(competitor)
                competitor_relative = (
                    1_000_000.0
                    if competitor_value < price_value / 1_000_001.0
                    else price_value / competitor_value - 1.0
                )

            entity_history = history_by_entity.get(row["entity_id"])
            available_demand = pd.DataFrame(columns=["target_at", "observed_at", "demand"])
            if entity_history is not None:
                observed_at = row["observed_at"]
                same_scope = entity_history[
                    (entity_history["domain"] == row["domain"])
                    & (entity_history["currency"] == row["currency"])
                    & (entity_history["target_at"] < target)
                    & (entity_history["observed_at"] <= observed_at)
                ].copy()
                if not same_scope.empty:
                    maturity = same_scope["target_at"].map(_bucket_end)
                    same_scope = same_scope.loc[maturity <= observed_at]
                    if not same_scope.empty:
                        # Repeated snapshots of a prior target bucket should
                        # contribute only the latest value known by this time.
                        available_demand = (
                            same_scope.sort_values(
                                ["target_at", "observed_at"], kind="mergesort"
                            )
                            .drop_duplicates("target_at", keep="last")
                            .sort_values("target_at", kind="mergesort")[
                                ["target_at", "observed_at", "demand"]
                            ]
                        )

            last_demand = (
                float(available_demand.iloc[-1]["demand"])
                if not available_demand.empty
                else np.nan
            )
            rolling_demand = (
                float(available_demand.tail(7)["demand"].mean())
                if not available_demand.empty
                else np.nan
            )
            output_rows.append(
                {
                    "entity_id": row["entity_id"],
                    "domain": row["domain"],
                    "currency": row["currency"],
                    "target_day_of_week": weekday,
                    "target_month": int(target.month),
                    "target_day_of_year": int(target.dayofyear),
                    "target_is_weekend": int(weekday >= 5),
                    "target_week_sin": float(np.sin(week_angle)),
                    "target_week_cos": float(np.cos(week_angle)),
                    "target_year_sin": float(np.sin(year_angle)),
                    "target_year_cos": float(np.cos(year_angle)),
                    "price": float(row["price"]),
                    "inventory": float(row["inventory"]),
                    "inventory_pressure": 1.0 / (float(row["inventory"]) + 1.0),
                    "competitor_price_relative": competitor_relative,
                    "competitor_price_missing": int(competitor_missing),
                    "demand_lag_1": last_demand,
                    "demand_rolling_mean_7": rolling_demand,
                    "event": row.get("event", pd.NA),
                    "seasonality": row.get("seasonality", pd.NA),
                    "product_category": row.get("product_category", pd.NA),
                }
            )

        features = pd.DataFrame(output_rows, columns=_OUTPUT_COLUMNS, index=current.index)
        assert_feature_output(features)
        return features

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return output column names for scikit-learn pipeline compatibility."""
        check_is_fitted(self, "history_")
        return np.asarray(_OUTPUT_COLUMNS, dtype=object)


def build_features(
    history: pd.DataFrame,
    observations: pd.DataFrame | None = None,
    *,
    domain: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper; `history` must contain only approved training data."""
    transformer = DemandFeatureTransformer(domain=domain).fit(history)
    return transformer.transform(history if observations is None else observations)


def assert_feature_output(features: pd.DataFrame) -> None:
    """Guard against accidentally passing labels or raw timestamps as features."""
    forbidden = {"demand", "target_at", "observed_at", "source", "competitor_source"}
    leaked = forbidden.intersection(features.columns)
    if leaked:
        raise DataValidationError(
            [f"feature output contains forbidden label/metadata column(s): {', '.join(sorted(leaked))}"]
        )
