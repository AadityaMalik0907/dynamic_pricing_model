"""Streamlit UI that consumes model artifacts produced by training notebooks."""

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from .. import __version__
from ..models.serving import forecast_hotel_arrivals
from ..settings import settings

st.set_page_config(page_title="Dynamic Pricing Demo", page_icon="🏨", layout="wide")
st.title("Dynamic Pricing Optimization")
st.caption("Advisory forecasts only. Train and evaluate models in notebooks before serving them.")
st.caption(
    f"Domain: {settings.pricing_domain} · Currency: {settings.price_currency} · "
    f"Version: {__version__} · Advisory demo only"
)

artifact_dir = Path(settings.model_artifact_dir)
metadata_path = artifact_dir / "hotel_arrivals_model.json"
daily_path = artifact_dir / "hotel_daily_arrivals.csv"
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    st.subheader("Loaded notebook model")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Model", metadata.get("selected_model", "unknown"))
    col_b.metric(
        "Test MAE",
        f"{metadata.get('test_metrics', {}).get('model', {}).get('mae', float('nan')):.2f}",
    )
    col_c.metric(
        "Test WAPE",
        f"{metadata.get('test_metrics', {}).get('model', {}).get('wape', float('nan')):.1%}",
    )
    st.caption(
        f"Version {metadata.get('model_version', 'unknown')} · "
        f"Test period {metadata.get('split_dates', {}).get('test_start', '?')} to "
        f"{metadata.get('split_dates', {}).get('test_end', '?')}"
    )
else:
    st.warning(
        "No notebook-trained model is available. Run "
        "notebooks/01_hotel_arrivals_training.ipynb to create artifacts/models."
    )

if daily_path.is_file():
    st.caption(f"Using processed hotel history saved by the notebook: {daily_path}")
    try:
        daily = pd.read_csv(daily_path)
    except (OSError, pd.errors.ParserError) as error:
        daily = None
        st.error(f"Could not read the processed daily history: {error}")
else:
    upload = st.file_uploader(
        "Upload daily hotel history CSV (hotel, arrival_at, arrivals)", type=["csv"]
    )
    daily = None
    if upload is not None:
        try:
            daily = pd.read_csv(upload)
        except pd.errors.ParserError as error:
            st.error(f"Could not parse the uploaded history CSV: {error}")

if daily is not None:
    required_columns = {"hotel", "arrival_at", "arrivals"}
    if not required_columns.issubset(daily.columns):
        st.error("Daily history must contain hotel, arrival_at, and arrivals columns.")
    else:
        daily["arrival_at"] = pd.to_datetime(daily["arrival_at"], errors="coerce")
        if daily["arrival_at"].isna().any():
            st.error("History includes invalid arrival dates.")
        else:
            hotel = st.selectbox("Hotel", sorted(daily["hotel"].dropna().unique().tolist()))
            hotel_history = daily.loc[daily["hotel"].eq(hotel)].sort_values("arrival_at")
            if len(hotel_history) < 28:
                st.info("At least 28 daily history rows are needed to make a forecast.")
            else:
                latest_date = hotel_history["arrival_at"].max().date()
                target_date = latest_date + timedelta(days=1)
                st.write(f"Forecast date: **{target_date.isoformat()}**")
                history = hotel_history.tail(28).rename(columns={"arrival_at": "arrival_date"})
                try:
                    prediction = forecast_hotel_arrivals(
                        hotel=hotel,
                        target_date=target_date,
                        history=history,
                        artifact_dir=artifact_dir,
                    )
                    st.metric(
                        "Predicted completed reservation arrivals",
                        f"{prediction['predicted_arrivals']:.1f}",
                    )
                except (FileNotFoundError, ValueError) as error:
                    st.error(str(error))
