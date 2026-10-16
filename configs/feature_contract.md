# Phase 3 — Feature and analysis contract

## Shared feature transformer

`dynamic_pricing.features.DemandFeatureTransformer` accepts validated canonical observations and emits a fixed feature frame for model training and serving. It can be fitted on training history and reused to transform later observations. When transforming a batch, earlier mature outcomes within that batch can contribute to later rows only after the per-row availability checks pass. The transformer is stateless at inference: a service should reload/update its history from durable storage as new realized outcomes become available. Its inputs are canonical rows; its outputs intentionally omit the target label, raw target timestamp, raw observation timestamp, source identifier, and competitor attribution.

Features include:

- Target weekday, month, day of year, weekend flag, and cyclical weekly/yearly encodings.
- Offered price and recorded available inventory.
- `inventory_pressure = 1 / (inventory + 1)`, a monotonic scarcity proxy. The data contract does not include total capacity, so this is not a true occupancy or capacity ratio.
- Competitor price relative to the offered price, clipped at a relative premium of 1,000,000 to avoid overflow, plus a missing-quote indicator. Competitor price remains optional.
- Latest mature prior demand and the mean of up to seven mature prior target buckets for the same entity, domain, and currency.
- Optional event, seasonality, and product category columns, passed through as categoricals only when those values were known at `observed_at`.

## Leakage controls

- A target bucket contributes demand history only after it has closed and its row was observed no later than the current row's observation time.
- A date-only target bucket closes at the next midnight UTC. Non-midnight timestamp targets close at their timestamp. A canonical midnight UTC target is treated as a date bucket, matching the Phase 2 contract.
- Current and future target buckets cannot contribute to a row's lag or rolling features. Repeated observations of an earlier target bucket contribute only the latest record available by that row's observation time.
- Features are generated per row using its own `observed_at` and only time-eligible outcomes; changing a future row's demand must not change earlier rows' features.
- Fit the transformer on training history, then reuse it for validation/test/serving observations. Do not fit any learned preprocessing on validation or test data.

The demand field must represent a stable realized outcome for each target bucket if it is to be used as historical lag data. If source data records booking pace or inventory snapshots instead of final units sold, Phase 4 must define and construct a separate realized-demand label before model training.

## Reproducible data profile

`dynamic_pricing.analysis.profile_observations` returns a JSON-compatible summary of row/entity/domain coverage, time ranges, currencies, missingness, numeric ranges, demand means by weekday/month, inventory saturation indicators, and descriptive price-demand correlations. Its Markdown view also documents the planned persistence and same-weekday baselines; Phase 4 will score them on chronological holdouts. `render_profile_markdown` renders that profile without writing to disk, so a UI or deployment can decide how to display or persist it.

Inventory saturation (`demand >= inventory`) is a warning signal for potentially censored demand, not a correction. Price-demand correlations are descriptive associations and do not measure a causal price effect. Synthetic demo inputs are explicitly flagged in the report.
