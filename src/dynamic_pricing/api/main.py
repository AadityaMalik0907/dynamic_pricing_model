"""API shell and inference endpoints for notebook-trained model artifacts."""

from datetime import date
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..logging_config import configure_logging
from ..models.serving import forecast_hotel_arrivals, load_hotel_model
from ..settings import settings

configure_logging()

app = FastAPI(
    title="Dynamic Pricing Optimization API",
    version=__version__,
    description=(
        "Advisory forecasting service. Training happens in notebooks; "
        "endpoints load saved artifacts."
    ),
)


class ArrivalHistoryPoint(BaseModel):
    """One observed daily completed-arrival count."""

    arrival_date: date
    arrivals: int = Field(ge=0)


class HotelArrivalForecastRequest(BaseModel):
    """Request a one-day hotel arrival forecast from recent history."""

    hotel: Literal["City Hotel", "Resort Hotel"]
    target_date: date
    history: list[ArrivalHistoryPoint] = Field(min_length=28, max_length=28)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Return process health without implying model readiness."""
    return {"status": "ok", "environment": settings.app_env, "version": __version__}


@app.get("/ready", tags=["operations"])
def readiness() -> dict[str, str]:
    """Report whether a notebook-trained hotel artifact can be loaded."""
    try:
        _, metadata = load_hotel_model(settings.model_artifact_dir)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ready", "model_version": str(metadata.get("model_version", "unknown"))}


@app.post("/forecast/hotel-arrivals", tags=["forecasting"])
def hotel_arrivals_forecast(request: HotelArrivalForecastRequest) -> dict[str, object]:
    """Load the notebook artifact and forecast the next day of hotel arrivals."""
    try:
        result = forecast_hotel_arrivals(
            hotel=request.hotel,
            target_date=request.target_date,
            history=pd.DataFrame([point.model_dump() for point in request.history]),
            artifact_dir=settings.model_artifact_dir,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return result
