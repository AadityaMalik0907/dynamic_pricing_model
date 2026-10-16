"""Deterministic, explicitly synthetic observations for software demos."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .adapters import adapt_ecommerce_observations, adapt_hotel_observations


def make_demo_observations(
    *,
    domain: str = "hotel",
    days: int = 120,
    entities: int = 3,
    seed: int = 42,
    start_date: date = date(2025, 1, 1),
) -> pd.DataFrame:
    """Create reproducible synthetic historical data, validated by its adapter.

    The generated values are illustrative and are not observed market data or
    suitable evidence of real demand, model quality, or revenue lift.
    """
    if domain not in {"hotel", "ecommerce"}:
        raise ValueError("domain must be 'hotel' or 'ecommerce'")
    if days < 2:
        raise ValueError("days must be at least 2")
    if entities < 1:
        raise ValueError("entities must be at least 1")

    rng = np.random.default_rng(seed)
    target_days = pd.date_range(start=start_date, periods=days, freq="D")
    rows: list[dict[str, object]] = []
    for entity_number in range(entities):
        entity = f"demo-{entity_number + 1:03d}"
        base_price = (7200.0 if domain == "hotel" else 1800.0) * (1 + 0.12 * entity_number)
        capacity = 24 + 4 * entity_number
        for day_number, target in enumerate(target_days):
            weekly_factor = 1.0 + 0.22 * (target.dayofweek in (4, 5))
            annual_proxy = 1.0 + 0.12 * np.sin(2 * np.pi * day_number / 90)
            price = base_price * (1 + 0.05 * np.sin(2 * np.pi * day_number / 30))
            demand_mean = max(1.0, capacity * 0.55 * weekly_factor * annual_proxy)
            simulated_demand = min(capacity, int(rng.poisson(demand_mean)))
            # Snapshot at the start of the target date. Prior calendar buckets
            # have closed, while the current bucket's outcome is still future.
            observed = target.tz_localize("UTC")
            row = {
                "observed_at": observed.isoformat(),
                "target_at": target.date().isoformat(),
                "price": round(price, 2),
                "currency": "INR",
                "source": "synthetic_demo",
                "competitor_source": "synthetic_demo" if day_number % 3 else None,
                "event": "synthetic_weekend" if target.dayofweek in (4, 5) else None,
                "seasonality": "synthetic_calendar_proxy",
            }
            if domain == "hotel":
                row.update(
                    {
                        "property_id": "demo-property",
                        "room_type_id": entity,
                        "stay_date": target.date().isoformat(),
                        "bookings": simulated_demand,
                        "rooms_available": capacity,
                        "competitor_rate": round(max(0.01, price * (1 + rng.normal(0, 0.08))), 2)
                        if day_number % 3
                        else None,
                    }
                )
            else:
                row.update(
                    {
                        "product_id": entity,
                        "units_sold": simulated_demand,
                        "inventory": capacity,
                        "category": "synthetic_demo_category",
                        "competitor_price": round(max(0.01, price * (1 + rng.normal(0, 0.08))), 2)
                        if day_number % 3
                        else None,
                    }
                )
            rows.append(row)

    raw = pd.DataFrame(rows)
    if domain == "hotel":
        return adapt_hotel_observations(raw)
    return adapt_ecommerce_observations(raw)
