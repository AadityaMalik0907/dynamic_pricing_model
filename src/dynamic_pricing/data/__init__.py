"""Data contracts, domain adapters, and demo data generation."""

from .adapters import adapt_ecommerce_observations, adapt_hotel_observations
from .demo import make_demo_observations
from .io import load_observations_csv
from .validation import DataValidationError, validate_observations

__all__ = [
    "DataValidationError",
    "adapt_ecommerce_observations",
    "adapt_hotel_observations",
    "load_observations_csv",
    "make_demo_observations",
    "validate_observations",
]
