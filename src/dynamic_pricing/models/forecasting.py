"""Chronological demand forecasting with baseline and tree model candidates."""

from __future__ import annotations

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

from ..data.validation import validate_observations
from ..features import _OUTPUT_COLUMNS, DemandFeatureTransformer

_CATEGORICAL_FEATURES = (
    "entity_id",
    "domain",
    "currency",
    "event",
    "seasonality",
    "product_category",
)
_ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ForecastTrainingResult:
    """Summary of one reproducible chronological training run."""

    artifact_dir: str
    model_version: str
    selected_model: str
    validation_metrics: dict[str, dict[str, float | None]]
    test_metrics: dict[str, float | None]
    split_dates: dict[str, str]
    row_counts: dict[str, int]
    warnings: tuple[str, ...]


class SeasonalNaiveRegressor:
    """Predict same-weekday demand, falling back to the latest mature demand."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        seasonal = pd.to_numeric(features["demand_seasonal_lag_7"], errors="coerce")
        latest = pd.to_numeric(features["demand_lag_1"], errors="coerce")
        return seasonal.fillna(latest).fillna(0.0).to_numpy(dtype=float).clip(min=0.0)


def _split_by_time(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    dates = np.array(sorted(data["target_at"].unique()))
    if len(dates) < 20:
        raise ValueError(
            "at least 20 distinct target dates are required for chronological "
            "train/validation/test evaluation"
        )
    train_end = max(1, int(len(dates) * 0.70))
    valid_end = max(train_end + 1, int(len(dates) * 0.85))
    valid_end = min(valid_end, len(dates) - 1)
    train_dates, valid_dates, test_dates = (
        dates[:train_end],
        dates[train_end:valid_end],
        dates[valid_end:],
    )
    train = data.loc[data["target_at"].isin(train_dates)].copy()
    valid = data.loc[data["target_at"].isin(valid_dates)].copy()
    test = data.loc[data["target_at"].isin(test_dates)].copy()
    return (
        train,
        valid,
        test,
        {
            "train_end_inclusive": str(pd.Timestamp(train_dates[-1]).date()),
            "validation_start": str(pd.Timestamp(valid_dates[0]).date()),
            "validation_end_inclusive": str(pd.Timestamp(valid_dates[-1]).date()),
            "test_start": str(pd.Timestamp(test_dates[0]).date()),
            "test_end_inclusive": str(pd.Timestamp(test_dates[-1]).date()),
        },
    )


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | None]:
    values = actual.to_numpy(dtype=float)
    denominator = float(np.abs(values).sum())
    return {
        "mae": float(mean_absolute_error(values, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(values, predicted))),
        "wape": float(np.abs(values - predicted).sum() / denominator) if denominator else None,
    }


def _candidate_pipeline(estimator: Any) -> Pipeline:
    numeric = [name for name in _OUTPUT_COLUMNS if name not in _CATEGORICAL_FEATURES]
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                                keep_empty_features=True,
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessing), ("regressor", estimator)])


def train_forecaster(
    observations: pd.DataFrame,
    *,
    domain: str,
    artifact_dir: str | Path = "artifacts",
    include_xgboost: bool = True,
) -> ForecastTrainingResult:
    """Train, compare, test, and persist one domain-specific demand forecaster.

    Splits are made on target dates in chronological order. Validation chooses
    between the seasonal naive baseline and candidate regressors; the final
    reported test score is computed once on later dates.
    """
    if domain not in {"hotel", "ecommerce"}:
        raise ValueError("domain must be 'hotel' or 'ecommerce'")
    data = validate_observations(observations, expected_domain=domain)
    data = data.sort_values(
        ["target_at", "entity_id", "observed_at"], kind="mergesort"
    ).reset_index(drop=True)
    train, validation, test, split_dates = _split_by_time(data)

    training_features = DemandFeatureTransformer(domain=domain).fit(train).transform(train)
    validation_features = DemandFeatureTransformer(domain=domain).fit(train).transform(validation)
    baseline = SeasonalNaiveRegressor()
    validation_predictions: dict[str, np.ndarray] = {
        "seasonal_naive": baseline.predict(validation_features)
    }
    candidates: dict[str, Pipeline] = {
        "scikit_hist_gradient_boosting": _candidate_pipeline(
            HistGradientBoostingRegressor(
                max_iter=100, learning_rate=0.08, l2_regularization=1.0, random_state=42
            )
        )
    }
    warnings: list[str] = []
    if include_xgboost:
        try:
            from xgboost import XGBRegressor

            candidates["xgboost"] = _candidate_pipeline(
                XGBRegressor(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                )
            )
        except ImportError:
            warnings.append(
                "XGBoost is unavailable; only the scikit-learn candidate was evaluated."
            )

    y_train = train["demand"].astype(float)
    y_validation = validation["demand"].astype(float)
    for name, model in candidates.items():
        model.fit(training_features, y_train)
        validation_predictions[name] = np.maximum(model.predict(validation_features), 0.0)
    validation_metrics = {
        name: _metrics(y_validation, predictions)
        for name, predictions in validation_predictions.items()
    }
    selected_model = min(validation_metrics, key=lambda name: validation_metrics[name]["mae"])

    fit_history = pd.concat([train, validation], ignore_index=True)
    final_transformer = DemandFeatureTransformer(domain=domain).fit(fit_history)
    final_training_features = final_transformer.transform(fit_history)
    test_features = final_transformer.transform(test)
    if selected_model == "seasonal_naive":
        final_model: Any = baseline
    else:
        final_model = candidates[selected_model]
        final_model.fit(final_training_features, fit_history["demand"].astype(float))
    test_predictions = np.maximum(final_model.predict(test_features), 0.0)
    test_metrics = _metrics(test["demand"].astype(float), test_predictions)

    baseline_test = _metrics(test["demand"].astype(float), baseline.predict(test_features))
    if selected_model != "seasonal_naive" and test_metrics["mae"] >= baseline_test["mae"]:
        warnings.append(
            "The selected ML candidate did not beat seasonal naive on the held-out test set; "
            "treat seasonal naive as the preferred fallback until more representative data "
            "is evaluated."
        )

    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / f"{domain}-demand-forecaster.joblib"
    metadata_path = directory / f"{domain}-demand-forecaster.json"
    bundle = ForecastBundle(domain=domain, model=final_model, transformer=final_transformer)
    joblib.dump(bundle, bundle_path)
    metadata = {
        "artifact_format_version": _ARTIFACT_FORMAT_VERSION,
        "model_version": version,
        "domain": domain,
        "selected_model": selected_model,
        "training_window": {
            "start": str(pd.Timestamp(fit_history["target_at"].min()).date()),
            "end": str(pd.Timestamp(fit_history["target_at"].max()).date()),
        },
        "split_dates": split_dates,
        "row_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "feature_schema": list(training_features.columns),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "seasonal_naive_test_metrics": baseline_test,
        "warnings": warnings,
        "synthetic_data_warning": bool(
            data["source"].astype(str).str.contains("synthetic", case=False).any()
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return ForecastTrainingResult(
        artifact_dir=str(directory),
        model_version=version,
        selected_model=selected_model,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        split_dates=split_dates,
        row_counts=metadata["row_counts"],
        warnings=tuple(warnings),
    )


@dataclass
class ForecastBundle:
    """Version-checked inference bundle for one domain."""

    domain: str
    model: Any
    transformer: DemandFeatureTransformer

    def predict(self, observations: pd.DataFrame) -> np.ndarray:
        features = self.transformer.transform(observations)
        return np.maximum(np.asarray(self.model.predict(features), dtype=float), 0.0)


def load_forecast(
    artifact_dir: str | Path, *, domain: str
) -> tuple[ForecastBundle, dict[str, Any]]:
    """Load a model bundle only after checking its domain and format metadata."""
    directory = Path(artifact_dir)
    bundle_path = directory / f"{domain}-demand-forecaster.joblib"
    metadata_path = directory / f"{domain}-demand-forecaster.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_format_version") != _ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported demand forecast artifact format version")
    if metadata.get("domain") != domain:
        raise ValueError("forecast artifact domain does not match the requested domain")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, ForecastBundle) or bundle.domain != domain:
        raise ValueError("forecast model bundle is invalid or has the wrong domain")
    expected_schema = list(bundle.transformer.get_feature_names_out())
    if metadata.get("feature_schema") != expected_schema:
        raise ValueError("forecast feature schema does not match the persisted model metadata")
    return bundle, metadata
