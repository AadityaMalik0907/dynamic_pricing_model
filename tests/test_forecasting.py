import json

import numpy as np
import pytest

from dynamic_pricing.data import make_demo_observations
from dynamic_pricing.models import load_forecast, train_forecaster


def test_forecaster_uses_chronological_splits_and_persists_checked_bundle(tmp_path) -> None:
    observations = make_demo_observations(domain="hotel", days=60, entities=2)
    result = train_forecaster(
        observations,
        domain="hotel",
        artifact_dir=tmp_path,
    )

    assert result.row_counts == {"train": 84, "validation": 18, "test": 18}
    assert result.split_dates["train_end_inclusive"] < result.split_dates["validation_start"]
    assert result.split_dates["validation_end_inclusive"] < result.split_dates["test_start"]
    assert set(result.test_metrics) == {"mae", "rmse", "wape"}
    assert result.test_metrics["mae"] >= 0
    assert "xgboost" in result.validation_metrics

    bundle, metadata = load_forecast(tmp_path, domain="hotel")
    predictions = bundle.predict(observations.iloc[-4:])
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert metadata["selected_model"] == result.selected_model
    assert metadata["synthetic_data_warning"] is True
    assert json.loads((tmp_path / "hotel-demand-forecaster.json").read_text())["feature_schema"]


def test_forecaster_rejects_too_few_target_dates() -> None:
    observations = make_demo_observations(domain="hotel", days=19, entities=1)

    with pytest.raises(ValueError, match="20 distinct target dates"):
        train_forecaster(observations, domain="hotel", include_xgboost=False)


def test_forecast_loader_rejects_wrong_domain(tmp_path) -> None:
    observations = make_demo_observations(domain="ecommerce", days=30, entities=1)
    train_forecaster(
        observations,
        domain="ecommerce",
        artifact_dir=tmp_path,
        include_xgboost=False,
    )

    with pytest.raises(FileNotFoundError):
        load_forecast(tmp_path, domain="hotel")
