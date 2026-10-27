"""Seasonal norm and surge flags (spec section 4)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

LEVEL_WEEKS = 13
FACTOR_RANGE = (0.5, 2.0)


def flag_surge(
    values: pd.Series, norm: pd.Series, threshold: float = 1.25, min_norm: float = 5
) -> pd.Series:
    """True where value >= threshold * norm and norm >= min_norm; NaN norm -> False."""
    flag = (values >= threshold * norm) & (norm >= min_norm)
    return flag.fillna(False).astype(bool)


def _woy(idx: pd.DatetimeIndex) -> pd.DataFrame:
    iso = idx.isocalendar()
    return pd.DataFrame(
        {"year": iso["year"].to_numpy(), "week": np.minimum(iso["week"].to_numpy(), 52)},
        index=idx,
    )


def seasonal_norm(panel: pd.DataFrame, origin: date, target_weeks: list[date]) -> pd.DataFrame:
    """Per-series seasonal norm for each target week, using only data with week_start <= origin.

    norm = mean of same-ISO-week-of-year counts over prior ISO years (week 53 -> 52)
           x level factor (13-week mean ending at origin / same weeks 52 weeks earlier,
           clipped to [0.5, 2.0]; 1.0 if undefined).
    """
    origin_ts = pd.Timestamp(origin)
    p = panel[["week_start", "borocd", "target", "count"]].copy()
    p["week_start"] = pd.to_datetime(p["week_start"])
    p = p[p["week_start"] <= origin_ts]
    wide = p.pivot(index="week_start", columns=["borocd", "target"], values="count").astype(float)
    wide = wide.sort_index()
    cols = wide.columns

    # level factor
    recent_idx = origin_ts - pd.to_timedelta(np.arange(LEVEL_WEEKS) * 7, unit="D")
    prior_idx = recent_idx - pd.Timedelta(weeks=52)
    recent = wide.reindex(recent_idx)
    prior = wide.reindex(prior_idx)
    complete = ~(recent.isna().any() | prior.isna().any())
    num, den = recent.mean(), prior.mean()
    factor = (num / den.where(den > 0)).clip(*FACTOR_RANGE)
    factor = factor.where(complete & (den > 0), 1.0).fillna(1.0)

    meta = _woy(wide.index)
    out = []
    for w in target_weeks:
        wts = pd.Timestamp(w)
        iso = wts.isocalendar()
        k = min(iso.week, 52)
        mask = ((meta["year"] < iso.year) & (meta["week"] == k)).to_numpy()
        seasonal = wide[mask].mean() if mask.any() else pd.Series(np.nan, index=cols)
        df = pd.DataFrame(
            {
                "borocd": cols.get_level_values(0).astype("int16"),
                "target": cols.get_level_values(1),
                "week_start": w,
                "norm": (seasonal * factor).to_numpy(),
            }
        )
        out.append(df)
    if not out:
        return pd.DataFrame(columns=["borocd", "target", "week_start", "norm"])
    return pd.concat(out, ignore_index=True)
