import numpy as np
import pandas as pd

from dynamic_pricing.analysis import profile_observations, render_profile_markdown
from dynamic_pricing.data import make_demo_observations
from dynamic_pricing.features import DemandFeatureTransformer, assert_feature_output


def test_calendar_inventory_and_lag_features_are_built_without_label_columns() -> None:
    history = make_demo_observations(domain="hotel", days=8, entities=1)
    features = DemandFeatureTransformer(domain="hotel").fit_transform(history)

    assert len(features) == len(history)
    assert features.loc[0, "demand_lag_1"] != features.loc[0, "demand_lag_1"]
    assert features.loc[1, "demand_lag_1"] == history.loc[0, "demand"]
    assert features.loc[7, "demand_rolling_mean_7"] == history.loc[:6, "demand"].mean()
    assert features.loc[0, "inventory_pressure"] == 1 / (history.loc[0, "inventory"] + 1)
    assert "target_is_weekend" in features
    assert_feature_output(features)


def test_future_outcomes_do_not_change_earlier_features() -> None:
    original = make_demo_observations(domain="ecommerce", days=12, entities=1)
    changed_future = original.copy(deep=True)
    changed_future.loc[11, "demand"] = changed_future.loc[11, "demand"] + 100

    first_features = DemandFeatureTransformer().fit_transform(original)
    changed_features = DemandFeatureTransformer().fit_transform(changed_future)

    pd.testing.assert_frame_equal(
        first_features.iloc[:11].reset_index(drop=True),
        changed_features.iloc[:11].reset_index(drop=True),
    )


def test_competitor_features_distinguish_present_and_missing_quotes() -> None:
    history = make_demo_observations(domain="hotel", days=3, entities=1)
    history.loc[0, "competitor_price"] = 5000.0
    history.loc[0, "competitor_source"] = "fixture_source"
    features = DemandFeatureTransformer().fit_transform(history)

    assert features.loc[0, "competitor_price_relative"] == history.loc[0, "price"] / 5000.0 - 1
    assert features.loc[0, "competitor_price_missing"] == 0
    assert features.loc[1, "competitor_price_missing"] == 1
    assert np.isnan(features.loc[1, "competitor_price_relative"])


def test_profile_reports_missingness_saturation_and_synthetic_data_caveat() -> None:
    observations = make_demo_observations(domain="hotel", days=4, entities=1)
    observations.loc[0, "demand"] = observations.loc[0, "inventory"]
    observations.loc[0, "competitor_price"] = np.nan
    observations.loc[0, "competitor_source"] = None

    profile = profile_observations(observations)
    report = render_profile_markdown(profile)

    hotel = profile["by_domain"]["hotel"]
    assert profile["row_count"] == 4
    assert profile["missingness"]["competitor_price"]["missing_count"] >= 1
    assert hotel["inventory"]["demand_at_or_above_inventory_count"] >= 1
    assert any("synthetic demo" in warning for warning in profile["warnings"])
    assert "not a causal" in report


def test_profile_supports_ecommerce_domain_separately() -> None:
    observations = make_demo_observations(domain="ecommerce", days=3, entities=2)
    profile = profile_observations(observations)

    assert profile["domain_counts"] == {"ecommerce": 6}
    assert profile["by_domain"]["ecommerce"]["entity_count"] == 2


def test_serving_transformer_uses_training_history_for_lags() -> None:
    all_rows = make_demo_observations(domain="hotel", days=6, entities=1)
    train = all_rows.iloc[:4]
    validation = all_rows.iloc[4:]
    transformer = DemandFeatureTransformer(domain="hotel").fit(train)

    features = transformer.transform(validation)

    assert features.iloc[0]["demand_lag_1"] == train.iloc[-1]["demand"]
    assert features.iloc[1]["demand_lag_1"] == validation.iloc[0]["demand"]
    assert "demand" not in features.columns
