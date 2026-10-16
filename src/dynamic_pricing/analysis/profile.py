"""Descriptive, reproducible profiling for canonical observations."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..data.validation import CANONICAL_COLUMNS, validate_observations


def _correlation(frame: pd.DataFrame, method: str) -> float | None:
    if len(frame) < 2 or frame["price"].nunique() < 2 or frame["demand"].nunique() < 2:
        return None
    value = frame["price"].corr(frame["demand"], method=method)
    return float(value) if pd.notna(value) else None


def profile_observations(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize coverage, missingness, numeric ranges, and censoring indicators."""
    data = validate_observations(frame)
    missingness = {}
    for column in CANONICAL_COLUMNS:
        missing_count = int(data[column].isna().sum()) if column in data.columns else len(data)
        missingness[column] = {
            "missing_count": missing_count,
            "missing_rate": missing_count / len(data),
        }

    by_domain: dict[str, object] = {}
    for domain, group in data.groupby("domain", sort=True):
        positive_inventory = group["inventory"] > 0
        saturated = positive_inventory & (group["demand"] >= group["inventory"])
        numeric_summary = {
            column: {
                "min": float(group[column].min()),
                "median": float(group[column].median()),
                "mean": float(group[column].mean()),
                "max": float(group[column].max()),
            }
            for column in ("price", "demand", "inventory")
        }
        weekday_summary = (
            group.assign(target_weekday=group["target_at"].dt.day_name().str[:3])
            .groupby("target_weekday", sort=True)["demand"]
            .agg(["count", "mean"])
        )
        month_summary = (
            group.assign(target_month=group["target_at"].dt.month)
            .groupby("target_month", sort=True)["demand"]
            .agg(["count", "mean"])
        )
        by_domain[str(domain)] = {
            "row_count": int(len(group)),
            "entity_count": int(group["entity_id"].nunique()),
            "currency_counts": {
                str(currency): int(count)
                for currency, count in group["currency"].value_counts().sort_index().items()
            },
            "observed_at_range_utc": [
                group["observed_at"].min().isoformat(),
                group["observed_at"].max().isoformat(),
            ],
            "target_at_range_utc": [
                group["target_at"].min().isoformat(),
                group["target_at"].max().isoformat(),
            ],
            "numeric_summary": numeric_summary,
            "seasonality": {
                "mean_demand_by_weekday": {
                    str(weekday): {"count": int(values["count"]), "mean": float(values["mean"])}
                    for weekday, values in weekday_summary.iterrows()
                },
                "mean_demand_by_month": {
                    str(month): {"count": int(values["count"]), "mean": float(values["mean"])}
                    for month, values in month_summary.iterrows()
                },
            },
            "inventory": {
                "zero_inventory_count": int((group["inventory"] == 0).sum()),
                "demand_at_or_above_inventory_count": int(saturated.sum()),
                "demand_at_or_above_inventory_rate_when_positive": (
                    float(saturated.sum() / positive_inventory.sum())
                    if positive_inventory.any()
                    else None
                ),
            },
            "price_demand_relationship": {
                "pearson_correlation": _correlation(group, "pearson"),
                "spearman_correlation": _correlation(group, "spearman"),
                "interpretation": "descriptive association only; not a causal price effect",
            },
            "source_counts": {
                str(source): int(count)
                for source, count in group["source"].value_counts().sort_index().items()
            },
        }

    warnings = [
        "Price-demand correlations are descriptive associations, not a causal price effect."
    ]
    if data["source"].eq("synthetic_demo").any():
        warnings.append(
            "This profile includes synthetic demo observations, not observed market data."
        )
    if any(
        values["inventory"]["demand_at_or_above_inventory_count"] > 0
        for values in by_domain.values()
    ):
        warnings.append(
            "Demand reaches or exceeds recorded inventory for some rows; "
            "observed demand may be censored."
        )

    return {
        "row_count": int(len(data)),
        "entity_count": int(data["entity_id"].nunique()),
        "domain_counts": {
            str(domain): int(count)
            for domain, count in data["domain"].value_counts().sort_index().items()
        },
        "missingness": missingness,
        "by_domain": by_domain,
        "forecast_baseline_plan": [
            "Persistence baseline: latest mature prior demand for the same entity.",
            "Seasonal baseline: mature demand from the same weekday one week earlier "
            "when available.",
            "Score baselines on chronological holdouts in Phase 4; this profile is "
            "descriptive only.",
        ],
        "warnings": warnings,
    }


def render_profile_markdown(profile: Mapping[str, object]) -> str:
    """Render a profile dictionary as a shareable Markdown report string."""
    lines = [
        "# Observation Data Profile",
        "",
        f"- Rows: {profile['row_count']}",
        f"- Entities: {profile['entity_count']}",
        "",
        "## Domain coverage",
        "",
        "| Domain | Rows | Entities | Target range (UTC) | Currencies |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for domain, raw_values in profile["by_domain"].items():
        values = raw_values
        target_range = values["target_at_range_utc"]
        currencies = ", ".join(values["currency_counts"].keys())
        lines.append(
            f"| {domain} | {values['row_count']} | {values['entity_count']} | "
            f"{target_range[0]} to {target_range[1]} | {currencies} |"
        )

    lines.extend(
        [
            "",
            "## Missingness",
            "",
            "| Field | Missing rows | Rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for column, values in profile["missingness"].items():
        lines.append(f"| {column} | {values['missing_count']} | {values['missing_rate']:.1%} |")

    lines.extend(["", "## Inventory and price-demand checks", ""])
    for domain, raw_values in profile["by_domain"].items():
        values = raw_values
        inventory = values["inventory"]
        relationship = values["price_demand_relationship"]
        saturated_rate = inventory["demand_at_or_above_inventory_rate_when_positive"]
        saturation_text = "n/a" if saturated_rate is None else f"{saturated_rate:.1%}"
        lines.append(
            f"- **{domain}:** {inventory['zero_inventory_count']} zero-inventory rows; "
            f"demand at or above positive inventory: {saturation_text}; "
            f"price-demand Pearson correlation: {relationship['pearson_correlation']}"
        )

    lines.extend(["", "## Seasonality preview", ""])
    for domain, raw_values in profile["by_domain"].items():
        values = raw_values
        weekday_means = ", ".join(
            f"{weekday}: {summary['mean']:.2f} (n={summary['count']})"
            for weekday, summary in values["seasonality"]["mean_demand_by_weekday"].items()
        )
        lines.append(f"- **{domain} weekday mean demand:** {weekday_means}")

    lines.extend(["", "## Forecast baselines for Phase 4", ""])
    lines.extend(f"- {baseline}" for baseline in profile["forecast_baseline_plan"])

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {warning}" for warning in profile["warnings"])
    lines.append("")
    return "\n".join(lines)
