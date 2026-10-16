# Phase 0 — Demo requirements and assumptions

## First use case

- **Domain:** hotel room-night pricing. E-commerce is a separate later phase and must have its own data adapter and model evaluation.
- **Primary user:** a hotel revenue/pricing analyst reviewing suggested prices.
- **Action:** recommendations are advisory only. No price is automatically published or applied.
- **Price unit/currency:** INR per room-night. Currency and unit must be changed with the data source; this is a demo default, not a claim about a real property.
- **Decision cadence:** one recommendation per property/room type/target night, refreshed daily when fresh data is available.
- **Data refresh assumption:** daily input refresh. **Model refresh assumption:** weekly, only if sufficient validated data is present. These are configurable starting assumptions, not operational guarantees.

## Demo guardrails

- Example floor: INR 2,500 per room-night.
- Example ceiling: INR 25,000 per room-night.
- Example maximum movement: 15% from the supplied current price in one recommendation.
- These values are placeholders for a local demonstration. A property operator must set realistic bounds before using recommendations for business decisions.
- Recommendations should expose the current price, suggested price, forecast demand, estimated revenue, constraints applied, data timestamp, and model version when those components are implemented.
- If required data or a valid model is missing or stale, do not silently invent a recommendation; return a clear unavailable/error state.

## Data and rights assumptions

- No customer or hotel data was supplied with the repository.
- Phase 2 may use clearly labeled synthetic data for software demonstrations, or real data the project owner has permission to use.
- Competitor prices are optional. Any real competitor data must include source and observation time and must be obtained and used under its applicable terms.
- Synthetic data is not evidence of real-world demand, forecast accuracy, or revenue lift.
- Do not commit private/raw datasets, credentials, or generated model artifacts.

## Evaluation and limitations

- Demand forecasting will be evaluated with chronological train/validation/test windows and compared with a simple baseline.
- Offline historical replay can estimate how a pricing rule behaves under assumptions; it does not establish causal revenue lift.
- Inventory availability can censor observed bookings, and historical prices may reflect prior business decisions. These limitations must be reported with model results.
- Automatic price changes and production use are out of scope for the initial demo.

## Decisions still needed before real-world use

Choose the property or dataset, room-type/entity identifiers, local timezone and booking-window convention, data source and usage rights, realistic currency/bounds, pricing objective (revenue or margin), and actual refresh/approval process. E-commerce requires a separate requirements and data review.
