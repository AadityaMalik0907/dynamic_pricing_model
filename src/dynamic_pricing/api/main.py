"""Minimal API shell for the first project phase."""

from fastapi import FastAPI

from .. import __version__
from ..logging_config import configure_logging
from ..settings import settings

configure_logging()

app = FastAPI(
    title="Dynamic Pricing Optimization API",
    version=__version__,
    description="Advisory demo service; pricing recommendations are added in later phases.",
)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Return process health without implying model readiness."""
    return {"status": "ok", "environment": settings.app_env, "version": __version__}
