"""Leakage-safe feature builder for the global LightGBM model (spec section 3)."""

from __future__ import annotations

from datetime import date, timedelta

import holidays
import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["borocd", "borough", "target", "is_special_area"]
FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "roll_mean_4",
    "roll_mean_13",
    "roll_mean_52",
    "roll_std_13",
    "same_week_last_year",
    "woy_sin",
    "woy_cos",
    "month",
    "n_holidays",
    "covid",
    "borocd",
    "borough",
    "target",
    "is_special_area",
    "horizon",
]

_WINDOW = 52
_COVID_START = date(2020, 3, 1)
_COVID_END = date(2021, 6, 30)


def _calendar(week_starts: list[date]) -> pd.DataFrame:
    """Calendar features for each (unique) target-week Monday."""
    years = range(min(week_starts).year, max(week_starts).year + 1)
    hol = holidays.country_holidays("US", subdiv="NY", years=years)
    hol_dates = set(hol.keys())
    rows = []
    for w in week_starts:
        k = min(w.isocalendar()[1], 52)
        rows.append(
            (
                w,
                np.sin(2 * np.pi * k / 52),
                np.cos(2 * np.pi * k / 52),
                w.month,
                sum((w + timedelta(days=d)) in hol_dates for d in range(7)),
                int(_COVID_START <= w <= _COVID_END),
            )
        )
    return pd.DataFrame(
        rows, columns=["week_start", "woy_sin", "woy_cos", "month", "n_holidays", "covid"]
    )


def build_features(
    panel: pd.DataFrame, origins: list[date], horizons: list[int], with_label: bool
) -> pd.DataFrame:
    """One row per (series, origin, horizon); all features use data up to the origin only."""
    wide = panel.pivot(index="week_start", columns=["borocd", "target"], values="count")
    wide = wide.astype(float)
    weeks = list(wide.index)
    pos = {w: i for i, w in enumerate(weeks)}
    y = wide.to_numpy()

    roll_mean = {w: wide.rolling(w, min_periods=w).mean().to_numpy() for w in (4, 13, 52)}
    roll_std13 = wide.rolling(13, min_periods=13).std(ddof=0).to_numpy()

    meta = (
        panel.drop_duplicates(["borocd", "target"])
        .set_index(["borocd", "target"])
        .loc[wide.columns, ["borough", "is_special_area"]]
    )
    borocd = wide.columns.get_level_values(0).to_numpy()
    target = wide.columns.get_level_values(1).to_numpy()
    borough = meta["borough"].to_numpy()
    special = meta["is_special_area"].to_numpy()

    valid = sorted(o for o in set(origins) if o in pos and pos[o] >= _WINDOW - 1)
    out_cols = FEATURE_COLUMNS + ["origin", "week_start", "scale"]
    if with_label:
        out_cols = out_cols + ["label"]
    if not valid:
        return pd.DataFrame(columns=out_cols)

    last_week = weeks[-1]
    all_targets = sorted({o + timedelta(weeks=h) for o in valid for h in horizons})
    cal = _calendar(all_targets).set_index("week_start")

    frames = []
    for o in valid:
        t = pos[o]
        scale = np.maximum(1.0, roll_mean[52][t])
        base = {
            "lag_1": y[t] / scale,
            "lag_2": y[t - 1] / scale,
            "lag_3": y[t - 2] / scale,
            "lag_4": y[t - 3] / scale,
            "roll_mean_4": roll_mean[4][t] / scale,
            "roll_mean_13": roll_mean[13][t] / scale,
            "roll_mean_52": roll_mean[52][t] / scale,
            "roll_std_13": roll_std13[t] / scale,
            "borocd": borocd,
            "borough": borough,
            "target": target,
            "is_special_area": special,
            "scale": scale,
            "origin": o,
        }
        for h in horizons:
            tw = o + timedelta(weeks=h)
            if with_label and tw > last_week:
                continue
            d = dict(base)
            d["same_week_last_year"] = y[t + h - _WINDOW] / scale
            c = cal.loc[tw]
            for name in ("woy_sin", "woy_cos", "month", "n_holidays", "covid"):
                d[name] = c[name]
            d["horizon"] = h
            d["week_start"] = tw
            if with_label:
                d["label"] = y[t + h] / scale
            frames.append(pd.DataFrame(d))
    if not frames:
        return pd.DataFrame(columns=out_cols)
    res = pd.concat(frames, ignore_index=True)
    for col in ("month", "n_holidays", "covid", "horizon"):
        res[col] = res[col].astype("int16")
    res["borocd"] = res["borocd"].astype("int16")
    return res[out_cols]
