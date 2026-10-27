"""Statistical forecasters via statsforecast: AutoETS and MSTL (spec section 3)."""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Literal

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS

from nyc311.models.base import FORECAST_COLUMNS, forecast_frame

MIN_MSTL_WEEKS = 104
SEASON = 52


def _long(panel: pd.DataFrame, origin: date) -> pd.DataFrame:
    p = panel[panel["week_start"] <= origin]
    return pd.DataFrame(
        {
            "unique_id": p["borocd"].astype(str) + "|" + p["target"].astype(str),
            "ds": pd.to_datetime(p["week_start"]),
            "y": p["count"].astype(float),
        }
    ).sort_values(["unique_id", "ds"], ignore_index=True)


def _run(model, df: pd.DataFrame, h: int) -> pd.DataFrame:
    alias = "m"
    model.alias = alias
    sf = StatsForecast(models=[model], freq="W-MON", n_jobs=-1)
    with warnings.catch_warnings():
        # constant/degenerate series trigger benign numpy/statsforecast warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="statsforecast")
        fc = sf.forecast(df=df, h=h, level=[80])
    return fc.rename(columns={alias: "yhat", f"{alias}-lo-80": "p10", f"{alias}-hi-80": "p90"})


class StatsForecaster:
    def __init__(self, kind: Literal["ets", "mstl"]):
        if kind not in ("ets", "mstl"):
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind
        self.name = "auto_ets" if kind == "ets" else "mstl"

    def forecast(self, panel: pd.DataFrame, origin: date, horizons: list[int]) -> pd.DataFrame:
        long = _long(panel, origin)
        if long.empty or not horizons:
            return pd.DataFrame(columns=FORECAST_COLUMNS)
        hmax = max(horizons)
        if self.kind == "ets":
            parts = [_run(AutoETS(season_length=1), long, hmax)]
        else:
            lengths = long.groupby("unique_id")["y"].transform("size")
            long_ids = long[lengths >= MIN_MSTL_WEEKS]
            short_ids = long[lengths < MIN_MSTL_WEEKS]
            parts = []
            if not long_ids.empty:
                mstl = MSTL(season_length=SEASON, trend_forecaster=AutoETS(model="ZZN"))
                parts.append(_run(mstl, long_ids, hmax))
            if not short_ids.empty:
                parts.append(_run(AutoETS(season_length=1), short_ids, hmax))
        fc = pd.concat(parts, ignore_index=True)
        fc["step"] = fc.groupby("unique_id").cumcount() + 1
        fc = fc[fc["step"].isin(horizons)]
        ids = fc["unique_id"].str.split("|", n=1, expand=True)
        out = forecast_frame(
            ids[0].astype(int).to_numpy(),
            ids[1].to_numpy(),
            origin,
            fc["step"].to_numpy(),
            [origin + timedelta(weeks=int(s)) for s in fc["step"]],
            self.name,
            fc["yhat"].clip(lower=0).to_numpy(),
            fc["p10"].clip(lower=0).to_numpy(),
            fc["p90"].clip(lower=0).to_numpy(),
        )
        return out.sort_values(["borocd", "target", "horizon"], ignore_index=True)
