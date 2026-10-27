"""Forecaster protocol and the forecast-frame schema."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

FORECAST_COLUMNS = [
    "borocd",
    "target",
    "origin",
    "horizon",
    "week_start",
    "model",
    "yhat",
    "yhat_p10",
    "yhat_p90",
]


class Forecaster(Protocol):
    name: str

    def forecast(self, panel: pd.DataFrame, origin: date, horizons: list[int]) -> pd.DataFrame:
        """Forecast frame for origin; must use only rows with week_start <= origin."""
        ...


def forecast_frame(
    borocd, target, origin, horizon, week_start, model, yhat, yhat_p10=np.nan, yhat_p90=np.nan
) -> pd.DataFrame:
    """Build a forecast frame; scalars broadcast against array arguments."""
    cols = np.broadcast_arrays(
        *[
            np.asarray(v, dtype=object) if not isinstance(v, (int, float, np.generic)) else v
            for v in (borocd, target, origin, horizon, week_start, model, yhat, yhat_p10, yhat_p90)
        ]
    )
    df = pd.DataFrame(dict(zip(FORECAST_COLUMNS, cols, strict=True)))
    df["borocd"] = df["borocd"].astype("int16")
    df["horizon"] = df["horizon"].astype("int64")
    for c in ("yhat", "yhat_p10", "yhat_p90"):
        df[c] = df[c].astype(float)
    return df[FORECAST_COLUMNS]
