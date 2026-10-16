"""Environment-backed application settings."""

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration shared by the local API and dashboard."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    pricing_domain: str = "hotel"
    price_currency: str = "INR"
    price_floor: float = Field(default=2500.0, gt=0)
    price_ceiling: float = Field(default=25000.0, gt=0)
    max_price_change_pct: float = Field(default=0.15, ge=0, le=1)
    data_refresh_hours: int = Field(default=24, gt=0)
    model_refresh_days: int = Field(default=7, gt=0)

    @model_validator(mode="after")
    def validate_price_bounds(self) -> Settings:
        if self.price_floor >= self.price_ceiling:
            raise ValueError("PRICE_FLOOR must be less than PRICE_CEILING")
        if self.pricing_domain not in {"hotel", "ecommerce"}:
            raise ValueError("PRICING_DOMAIN must be 'hotel' or 'ecommerce'")
        return self


settings = Settings()
