"""Baseline forecasters (spec section 3)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from nyc311.models.base import FORECAST_COLUMNS, forecast_frame


def _wide(panel: pd.DataFrame, origin: date) -> pd.DataFrame:
    p = panel[panel["week_start"] <= origin]
    return p.pivot(index="week_start", columns=["borocd", "target"], values="count").astype(float)


def _build(name, wide, origin, horizons, value_fn) -> pd.DataFrame:
    frames = []
    for h in horizons:
        yhat = value_fn(wide, h)
        frames.append(
            forecast_frame(
                wide.columns.get_level_values(0).to_numpy(),
                wide.columns.get_level_values(1).to_numpy(),
                origin,
                h,
                origin + timedelta(weeks=h),
                name,
                yhat.to_numpy(),
            )
        )
    if not frames:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    return pd.concat(frames, ignore_index=True)


class SeasonalNaive:
    name = "seasonal_naive"

    def forecast(self, panel: pd.DataFrame, origin: date, horizons: list[int]) -> pd.DataFrame:
        wide = _wide(panel, origin)

        def value(w: pd.DataFrame, h: int) -> pd.Series:
            src = origin + timedelta(weeks=h - 52)
            if src in w.index:
                return w.loc[src]
            return pd.Series(float("nan"), index=w.columns)

        return _build(self.name, wide, origin, horizons, value)


class RecentAverage:
    name = "recent_average"

    def forecast(self, panel: pd.DataFrame, origin: date, horizons: list[int]) -> pd.DataFrame:
        wide = _wide(panel, origin).sort_index()
        avg = wide.iloc[-4:].mean()
        return _build(self.name, wide, origin, horizons, lambda w, h: avg)
