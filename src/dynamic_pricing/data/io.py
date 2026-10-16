"""Platform-neutral CSV loading and canonical validation helpers."""

from __future__ import annotations

from os import PathLike
from typing import BinaryIO

import pandas as pd

from .adapters import adapt_ecommerce_observations, adapt_hotel_observations
from .validation import DataValidationError


def load_observations_csv(
    source: str | PathLike[str] | BinaryIO,
    *,
    domain: str,
) -> pd.DataFrame:
    """Load and adapt a CSV from a path or uploaded binary file object.

    The function does not write files, depend on a working directory, or assume
    a hosting platform. `source` may be a local path, Streamlit UploadedFile,
    or another file-like object accepted by pandas.
    """
    if domain == "hotel":
        adapter = adapt_hotel_observations
    elif domain == "ecommerce":
        adapter = adapt_ecommerce_observations
    else:
        raise ValueError("domain must be 'hotel' or 'ecommerce'")
    try:
        frame = pd.read_csv(source)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        raise DataValidationError([f"could not read CSV input: {exc}"]) from exc
    return adapter(frame)
