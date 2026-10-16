"""Demand model training, evaluation, and artifact loading."""

from .ecommerce_popularity import (
    EcommercePopularityTrainingResult,
    prepare_amazon_product_data,
    split_products_by_id,
    train_ecommerce_popularity,
)
from .forecasting import ForecastBundle, ForecastTrainingResult, load_forecast, train_forecaster
from .hotel_arrivals import (
    HotelArrivalsTrainingResult,
    build_arrival_features,
    build_daily_hotel_arrivals,
    train_hotel_arrivals,
)

__all__ = [
    "ForecastBundle",
    "ForecastTrainingResult",
    "EcommercePopularityTrainingResult",
    "HotelArrivalsTrainingResult",
    "build_arrival_features",
    "build_daily_hotel_arrivals",
    "load_forecast",
    "prepare_amazon_product_data",
    "split_products_by_id",
    "train_forecaster",
    "train_ecommerce_popularity",
    "train_hotel_arrivals",
]
