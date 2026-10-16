"""Initial dashboard shell; KPI and model views arrive in later phases."""

import streamlit as st

from .. import __version__
from ..settings import settings

st.set_page_config(page_title="Dynamic Pricing Demo", page_icon="🏨", layout="wide")
st.title("Dynamic Pricing Optimization")
st.info("Project foundation is ready. Forecasts and recommendations will be added in later phases.")
st.caption(
    f"Domain: {settings.pricing_domain} · Currency: {settings.price_currency} · "
    f"Version: {__version__} · Advisory demo only"
)
