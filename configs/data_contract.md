# Phase 2 — Observation data contract

Both domains are adapted into one row-per-entity, observation-time, target-time contract. Hotel and e-commerce source columns stay explicit in their adapters, and every row retains a `domain` value. Do not train a shared model across domains unless a later experiment supports that choice.

## Canonical columns

| Field | Required | Meaning and validation |
| --- | --- | --- |
| `entity_id` | Yes | Stable hotel room-type or product identifier. Hotel identifiers combine `property_id` and `room_type_id` as `property::room_type`. |
| `domain` | Yes | `hotel` or `ecommerce`. |
| `observed_at` | Yes | When the input snapshot was observed; timezone-aware ISO-8601 timestamp. |
| `target_at` | Yes | Stay night or product demand bucket; ISO date or timezone-aware ISO-8601 timestamp. Date-only buckets normalize to midnight UTC. |
| `price` | Yes | Positive finite offered price in the row's currency. |
| `demand` | Yes | Nonnegative units/bookings recorded for the target bucket. This field's exact business meaning must be consistent within a source. |
| `inventory` | Yes | Nonnegative available quantity/capacity represented by the source snapshot. |
| `currency` | Yes | Three uppercase letters (for example `INR`); the validator checks shape, not ISO currency membership, and conversion is not performed. |
| `source` | Yes | Dataset/source identifier; synthetic examples use `synthetic_demo`. |
| `competitor_price` | No | Positive finite competitor quote when available. Requires `competitor_source`. |
| `competitor_source` | Conditional | Attribution for the competitor quote. Must accompany `competitor_price`. |
| `event` | No | Optional event annotation. |
| `seasonality` | No | Optional source-provided seasonality annotation. |
| `property_id`, `room_type_id`, `product_category` | No | Domain attributes preserved when supplied by the adapter. |

For timestamp targets, `observed_at` must not be later than `target_at` as an instant. For date-only target buckets, the UTC calendar date of `observed_at` may equal the target date (the snapshot can be taken later that day), but cannot be later. The unique row key is (`entity_id`, `observed_at`, `target_at`). Duplicate rows are rejected rather than silently aggregated. Numeric fields reject nulls (except optional competitor price), infinities, and invalid signs. Validation returns a normalized copy and does not mutate the source frame.

## Source layouts

Hotel CSV required columns:

```text
property_id, room_type_id, observed_at, stay_date, price, bookings,
rooms_available, currency, source
```

Optional hotel columns: `competitor_rate`, `competitor_source`, `event`, `seasonality`.

E-commerce CSV required columns:

```text
product_id, observed_at, target_at, price, units_sold, inventory,
currency, source
```

Optional e-commerce columns: `competitor_price`, `competitor_source`, `event`, `seasonality`, `category`.

Timestamps for `observed_at` must include an offset (`2025-01-01T12:00:00Z`). Date-only `stay_date` and `target_at` are allowed and treated as calendar dates, not local-time instants. A canonical target timestamp at exactly UTC midnight is also treated as a date bucket because that is how date-only values normalize. Clarify timezone and daylight-saving conventions for the real data source before model features are built.

## Using the loader

```python
from dynamic_pricing.data import load_observations_csv

canonical = load_observations_csv("data/my_hotel.csv", domain="hotel")
```

`source` may also be a file-like object, including a Streamlit file upload. The loader does not write files or depend on a specific current working directory. Hosting applications should place any persistent raw data in storage configured for that platform; process memory and a deployment's local filesystem are not durable storage.

## Synthetic demo data

```python
from dynamic_pricing.data import make_demo_observations

demo = make_demo_observations(domain="hotel", days=120, entities=3, seed=42)
```

The generator is deterministic for a fixed set of arguments and marks every row with `source="synthetic_demo"`. The values are artificial and must not be represented as observed market data, model validation evidence, or revenue lift.
