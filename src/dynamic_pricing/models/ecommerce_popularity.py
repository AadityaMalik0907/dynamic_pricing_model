"""Train a traditional ML regressor for product review-count popularity."""

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

_NUMERIC_FEATURES = ["discounted_price", "actual_price", "discount_percentage"]
_CATEGORICAL_FEATURES = ["category"]
_FEATURE_SCHEMA = [*_NUMERIC_FEATURES, *_CATEGORICAL_FEATURES]


@dataclass(frozen=True)
class EcommercePopularityTrainingResult:
    """Summary of the grouped product-popularity model evaluation."""

    artifact_path: str
    metadata_path: str
    row_counts: dict[str, int]
    metrics: dict[str, float | None]
    baseline_metrics: dict[str, float | None]
    model_beats_baseline: bool
    warnings: tuple[str, ...]


def prepare_amazon_product_data(products: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse prices and cumulative review counts without using review text/IDs."""
    required = {
        "product_id",
        "category",
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "rating_count",
    }
    missing = sorted(required.difference(products.columns))
    if missing:
        raise ValueError(f"Amazon product data is missing required column(s): {', '.join(missing)}")
    if products.empty:
        raise ValueError("Amazon product data contains no rows")

    frame = products.copy()
    input_rows = len(frame)
    frame["product_id"] = frame["product_id"].astype("string").str.strip()
    if frame["product_id"].isna().any() or frame["product_id"].eq("").any():
        raise ValueError("product_id must be present for every product row")
    for column in _NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(
            frame[column].astype("string").str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce",
        )
    frame["rating_count"] = pd.to_numeric(
        frame["rating_count"].astype("string").str.replace(",", "", regex=False),
        errors="coerce",
    )
    frame["category"] = (
        frame["category"]
        .astype("string")
        .str.split("|", regex=False)
        .str[0]
        .str.strip()
        .replace("", pd.NA)
    )
    missing_targets = int(frame["rating_count"].isna().sum())
    frame = frame.dropna(subset=["rating_count"]).copy()
    if frame.empty:
        raise ValueError("no numeric rating_count targets remain after parsing")
    if (frame["rating_count"] < 0).any():
        raise ValueError("rating_count cannot be negative")
    if frame[_NUMERIC_FEATURES].isna().any().any():
        bad = frame[_NUMERIC_FEATURES].isna().sum()
        raise ValueError(f"unparseable price/discount values: {bad[bad.gt(0)].to_dict()}")
    if (frame[["discounted_price", "actual_price"]] <= 0).any().any():
        raise ValueError("discounted_price and actual_price must be greater than zero")
    if not frame["discount_percentage"].between(0, 100).all():
        raise ValueError("discount_percentage must be between 0 and 100")
    if frame["category"].isna().any():
        frame["category"] = frame["category"].fillna("Unknown")

    counts = {
        "input_rows": input_rows,
        "rows_missing_rating_count_removed": missing_targets,
        "model_rows": len(frame),
        "unique_product_ids": int(frame["product_id"].nunique()),
        "repeated_product_id_rows": int(frame["product_id"].duplicated(keep=False).sum()),
        "top_level_categories": int(frame["category"].nunique()),
    }
    return frame.reset_index(drop=True), counts


def _pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), _NUMERIC_FEATURES),
            (
                "category",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
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
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=3,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | None]:
    truth = actual.to_numpy(dtype=float)
    total = float(np.abs(truth).sum())
    return {
        "mae_rating_count": float(mean_absolute_error(truth, predicted)),
        "rmse_rating_count": float(np.sqrt(mean_squared_error(truth, predicted))),
        "wape": float(np.abs(truth - predicted).sum() / total) if total else None,
        "r2": float(r2_score(truth, predicted)) if len(truth) > 1 else None,
    }


def split_products_by_id(products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic 80/20 sets with no product ID in both partitions."""
    if products["product_id"].nunique() < 10:
        raise ValueError(
            "at least 10 unique product IDs are required for a grouped train/test split"
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_indices, test_indices = next(splitter.split(products, groups=products["product_id"]))
    train = products.iloc[train_indices]
    test = products.iloc[test_indices]
    overlap = set(train["product_id"]).intersection(test["product_id"])
    if overlap:
        raise RuntimeError("product ID leakage detected between train and test partitions")
    return train, test


def train_ecommerce_popularity(
    csv_path: str | Path,
    *,
    artifact_dir: str | Path = "artifacts",
) -> EcommercePopularityTrainingResult:
    """Train/test one RandomForest model with product IDs held in one split only.

    `rating_count` is a cumulative review/popularity proxy, not units sold or
    time-indexed demand. This model must not be used as the pricing demand model.
    """
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"Amazon product CSV does not exist: {csv_file}")
    source_hash = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    products, data_counts = prepare_amazon_product_data(pd.read_csv(csv_file))
    train, test = split_products_by_id(products)
    model = _pipeline()
    model.fit(train[_FEATURE_SCHEMA], np.log1p(train["rating_count"].astype(float)))
    predictions = np.maximum(np.expm1(model.predict(test[_FEATURE_SCHEMA])), 0.0)
    baseline_value = float(train["rating_count"].median())
    baseline_predictions = np.full(len(test), baseline_value, dtype=float)
    metrics = _metrics(test["rating_count"], predictions)
    baseline_metrics = _metrics(test["rating_count"], baseline_predictions)
    beats_baseline = metrics["mae_rating_count"] < baseline_metrics["mae_rating_count"]
    warnings: list[str] = []
    if not beats_baseline:
        warnings.append(
            "The RandomForest did not beat the training-median baseline on test MAE; "
            "do not treat it as a useful popularity predictor without more data/features."
        )
    if metrics["wape"] is not None and metrics["wape"] > 0.50:
        warnings.append(
            "Test WAPE exceeds 50%; the popularity-count predictions have large relative "
            "errors and are only a pipeline demonstration."
        )
    if metrics["r2"] is not None and metrics["r2"] < 0.10:
        warnings.append(
            "Test R² is below 0.10; this model explains little product-to-product "
            "variation in cumulative rating counts."
        )

    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "ecommerce-review-popularity-random-forest.joblib"
    metadata_path = output / "ecommerce-review-popularity-random-forest.json"
    joblib.dump(model, model_path)
    metadata: dict[str, Any] = {
        "artifact_format_version": 1,
        "model_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "model": "scikit-learn RandomForestRegressor",
        "target": "cumulative product rating_count",
        "target_interpretation": "review-count popularity proxy; not sales units or demand",
        "features": _FEATURE_SCHEMA,
        "train_test_method": "80/20 GroupShuffleSplit by product_id; random_state=42",
        "row_counts": {"train": len(train), "test": len(test), **data_counts},
        "metrics": {"model": metrics, "training_median_baseline": baseline_metrics},
        "model_beats_baseline_test_mae": beats_baseline,
        "dataset_file": csv_file.name,
        "dataset_sha256": source_hash,
        "limitations": [
            "The file has no sales quantity, order date, or review-count observation date.",
            "Cumulative rating_count is not a demand target and cannot support temporal "
            "evaluation.",
            "The cross-sectional model does not estimate price elasticity or causal price effects.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return EcommercePopularityTrainingResult(
        artifact_path=str(model_path),
        metadata_path=str(metadata_path),
        row_counts={"train": len(train), "test": len(test), **data_counts},
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        model_beats_baseline=beats_baseline,
        warnings=tuple(warnings),
    )
