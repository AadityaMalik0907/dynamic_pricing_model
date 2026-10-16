"""Load notebook-trained model artifacts for API and dashboard inference."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .hotel_arrivals import _FEATURE_SCHEMA, build_arrival_features


@lru_cache(maxsize=4)
def _load_hotel_model(artifact_dir: str) -> tuple[Any, dict[str, Any]]:
    directory = Path(artifact_dir)
    model_path = directory / "hotel_arrivals_model.joblib"
    metadata_path = directory / "hotel_arrivals_model.json"
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Notebook-trained hotel model is missing in '{directory}'. "
            "Run notebooks/01_hotel_arrivals_training.ipynb first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_format_version") != 1:
        raise ValueError("unsupported hotel model artifact format")
    if metadata.get("feature_schema") != _FEATURE_SCHEMA:
        raise ValueError("hotel model feature schema is incompatible with this application")
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise ValueError("hotel model artifact does not expose predict()")
    return model, metadata


def load_hotel_model(artifact_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load the notebook's model and companion metadata."""
    return _load_hotel_model(str(Path(artifact_dir).resolve()))


def forecast_hotel_arrivals(
    *,
    hotel: str,
    target_date: date,
    history: pd.DataFrame,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Forecast one day from the exact 28 preceding daily arrival counts."""
    if hotel not in {"City Hotel", "Resort Hotel"}:
        raise ValueError("hotel must be 'City Hotel' or 'Resort Hotel'")
    required = {"arrival_date", "arrivals"}
    if not required.issubset(history.columns):
        raise ValueError("history must contain arrival_date and arrivals columns")
    if len(history) != 28:
        raise ValueError("exactly 28 consecutive history days are required")
    recent = history.loc[:, ["arrival_date", "arrivals"]].copy()
    recent["arrival_at"] = pd.to_datetime(recent.pop("arrival_date"), errors="coerce")
    if recent["arrival_at"].isna().any():
        raise ValueError("history arrival_date values must be valid dates")
    recent["arrivals"] = pd.to_numeric(recent["arrivals"], errors="coerce")
    if recent["arrivals"].isna().any() or (recent["arrivals"] < 0).any():
        raise ValueError("history arrivals must be non-negative numeric values")
    recent = recent.sort_values("arrival_at", kind="mergesort").reset_index(drop=True)
    if recent["arrival_at"].duplicated().any():
        raise ValueError("history must contain one row per date")
    target = pd.Timestamp(target_date)
    expected_dates = pd.date_range(target - pd.Timedelta(days=28), periods=28, freq="D")
    actual_dates = recent["arrival_at"].to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(actual_dates, expected_dates.to_numpy(dtype="datetime64[ns]")):
        raise ValueError(
            "history must cover every day from target_date - 28 through target_date - 1"
        )

    context = recent.assign(hotel=hotel)
    future = pd.DataFrame([{"hotel": hotel, "arrival_at": target, "arrivals": np.nan}])
    features = build_arrival_features(pd.concat([context, future], ignore_index=True))
    target_features = features.loc[features["arrival_at"].eq(target), _FEATURE_SCHEMA]
    model, metadata = load_hotel_model(artifact_dir)
    prediction = float(np.maximum(model.predict(target_features)[0], 0.0))
    return {
        "hotel": hotel,
        "target_date": target.date().isoformat(),
        "predicted_arrivals": prediction,
        "target_unit": metadata.get("target_unit", "reservation arrivals"),
        "model_name": metadata.get("selected_model", "notebook-trained model"),
        "model_version": metadata.get("model_version"),
    }
