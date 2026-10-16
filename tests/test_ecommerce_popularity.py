import json

import pandas as pd
import pytest

from dynamic_pricing.models import (
    prepare_amazon_product_data,
    split_products_by_id,
    train_ecommerce_popularity,
)


def _products(count: int = 60) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "product_id": f"SKU-{index // 2:03d}",
                "category": "Computers|Accessories|Cables" if index % 2 else "Home|Lights",
                "discounted_price": f"₹{100 + index}",
                "actual_price": f"₹{200 + index}",
                "discount_percentage": f"{50 - (index % 10)}%",
                "rating_count": f"{1_000 + index * 200:,}",
            }
        )
    return pd.DataFrame(rows)


def test_amazon_parser_cleans_rupee_prices_indian_count_commas_and_category() -> None:
    parsed, counts = prepare_amazon_product_data(_products(20))

    assert parsed.loc[0, "discounted_price"] == 100
    assert parsed.loc[0, "rating_count"] == 1000
    assert parsed.loc[0, "category"] == "Home"
    assert counts["unique_product_ids"] == 10
    assert counts["repeated_product_id_rows"] == 20


def test_product_ids_do_not_leak_between_train_and_test() -> None:
    parsed, _ = prepare_amazon_product_data(_products())
    train, test = split_products_by_id(parsed)

    assert set(train.product_id).isdisjoint(set(test.product_id))
    assert abs(len(test) / len(parsed) - 0.2) < 0.05


def test_amazon_popularity_model_saves_test_metrics_and_scope_limits(tmp_path) -> None:
    source = tmp_path / "amazon.csv"
    _products(100).to_csv(source, index=False)

    result = train_ecommerce_popularity(source, artifact_dir=tmp_path / "artifacts")

    assert result.row_counts["train"] > result.row_counts["test"]
    assert result.metrics["mae_rating_count"] >= 0
    assert (tmp_path / "artifacts" / "ecommerce-review-popularity-random-forest.joblib").is_file()
    metadata = json.loads(
        (tmp_path / "artifacts" / "ecommerce-review-popularity-random-forest.json").read_text()
    )
    assert (
        metadata["target_interpretation"]
        == "review-count popularity proxy; not sales units or demand"
    )
    assert metadata["train_test_method"].startswith("80/20 GroupShuffleSplit")


def test_amazon_parser_rejects_invalid_prices() -> None:
    products = _products(20)
    products.loc[0, "discounted_price"] = "not a price"

    with pytest.raises(ValueError, match="unparseable price"):
        prepare_amazon_product_data(products)
