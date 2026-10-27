"""Weekly panel: 71 areas x 6 targets x weeks, zero-filled, plus validation."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from nyc311.clean import (
    REGULAR_DISTRICTS,
    SPECIAL_AREAS,
    borough_of,
    map_category,
    parse_area,
)
from nyc311.config import Categories

PANEL_START = date(2020, 1, 1)
RECON_TOLERANCE = 0.01
_AREAS = sorted(REGULAR_DISTRICTS | SPECIAL_AREAS)


class PanelValidationError(ValueError):
    pass


def last_complete_week(max_day: date) -> date:
    """Monday of the latest week whose Sunday is strictly before `max_day`."""
    monday = max_day - timedelta(days=max_day.weekday())
    return monday - timedelta(days=7)


def _none(value):
    return None if pd.isna(value) else value


def _glob(raw_dir: Path, prefix: str) -> str:
    files = sorted(Path(raw_dir).glob(f"{prefix}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {prefix}_*.parquet files in {raw_dir}")
    return str(Path(raw_dir) / f"{prefix}_*.parquet")


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.date


def build_panel(raw_dir: Path, cats: Categories) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(panel, unmapped)`; unmapped has columns complaint_type, n (descending)."""
    raw_dir = Path(raw_dir)
    daily = _glob(raw_dir, "daily")
    con = duckdb.connect()
    pairs = con.execute(
        f"SELECT DISTINCT community_board, complaint_type FROM read_parquet('{daily}')"
    ).df()
    pairs["borocd"] = [parse_area(_none(b)) for b in pairs["community_board"]]
    pairs["category"] = [
        None if pd.isna(t) else map_category(t, cats) for t in pairs["complaint_type"]
    ]
    pairs["borocd"] = pairs["borocd"].astype("Int64")
    con.register("pairs", pairs)

    min_day, max_day = con.execute(
        f"SELECT min(day), max(day) FROM read_parquet('{daily}')"
    ).fetchone()
    last_week = last_complete_week(max_day)
    start = min_day + timedelta(days=(-min_day.weekday()) % 7)

    agg = con.execute(
        f"""
        SELECT date_trunc('week', d.day)::DATE AS week_start, p.borocd, p.category,
               sum(d.n)::BIGINT AS n
        FROM read_parquet('{daily}') d
        JOIN pairs p
          ON d.community_board IS NOT DISTINCT FROM p.community_board
         AND d.complaint_type IS NOT DISTINCT FROM p.complaint_type
        WHERE p.borocd IS NOT NULL AND date_trunc('week', d.day)::DATE <= ?
        GROUP BY ALL
        """,
        [last_week],
    ).df()
    unmapped = con.execute(
        f"""
        SELECT d.complaint_type, sum(d.n)::BIGINT AS n
        FROM read_parquet('{daily}') d
        JOIN pairs p
          ON d.community_board IS NOT DISTINCT FROM p.community_board
         AND d.complaint_type IS NOT DISTINCT FROM p.complaint_type
        WHERE p.category IS NULL AND d.complaint_type IS NOT NULL
        GROUP BY d.complaint_type ORDER BY n DESC, d.complaint_type
        """
    ).df()
    con.close()

    agg["week_start"] = _to_date(agg["week_start"])
    agg["borocd"] = agg["borocd"].astype("int64")
    total = agg.groupby(["week_start", "borocd"], as_index=False)["n"].sum()
    total["target"] = "total"
    per_cat = agg.dropna(subset=["category"]).rename(columns={"category": "target"})
    long = pd.concat([total, per_cat[["week_start", "borocd", "target", "n"]]])
    long = long.groupby(["week_start", "borocd", "target"], as_index=False)["n"].sum()

    weeks = pd.date_range(start, last_week, freq="7D").date
    grid = pd.MultiIndex.from_product(
        [weeks, _AREAS, cats.targets], names=["week_start", "borocd", "target"]
    )
    panel = (
        long.set_index(["week_start", "borocd", "target"])
        .reindex(grid, fill_value=0)
        .rename(columns={"n": "count"})
        .reset_index()
    )
    panel["is_special_area"] = panel["borocd"].isin(SPECIAL_AREAS)
    panel["borough"] = panel["borocd"].map({a: borough_of(a) for a in _AREAS})
    panel["borocd"] = panel["borocd"].astype("int16")
    panel["count"] = panel["count"].astype("int32")
    return panel[
        ["week_start", "borocd", "target", "count", "is_special_area", "borough"]
    ], unmapped


def _dropped_by_week(raw_dir: Path) -> pd.Series:
    daily = _glob(raw_dir, "daily")
    df = duckdb.sql(
        f"""SELECT date_trunc('week', day)::DATE AS week_start, community_board,
                   sum(n)::BIGINT AS n
            FROM read_parquet('{daily}') GROUP BY ALL"""
    ).df()
    df["week_start"] = _to_date(df["week_start"])
    df["valid"] = [parse_area(_none(b)) is not None for b in df["community_board"]]
    return df[~df["valid"]].groupby("week_start")["n"].sum()


def _citywide_by_week(raw_dir: Path) -> pd.Series:
    city = _glob(raw_dir, "citywide")
    df = duckdb.sql(
        f"""SELECT date_trunc('week', day)::DATE AS week_start, sum(n)::BIGINT AS n
            FROM read_parquet('{city}') GROUP BY ALL"""
    ).df()
    df["week_start"] = _to_date(df["week_start"])
    return df.set_index("week_start")["n"]


def validate_panel(panel: pd.DataFrame, raw_dir: Path) -> None:
    """Raise PanelValidationError listing every failed check."""
    errors: list[str] = []

    areas = panel["borocd"].unique()
    n_regular = sum(a in REGULAR_DISTRICTS for a in areas)
    if len(areas) != 71 or n_regular != 59:
        errors.append(f"areas: expected 71 (59 regular), got {len(areas)} ({n_regular} regular)")

    weeks = sorted(panel["week_start"].unique())
    expected = list(pd.date_range(weeks[0], weeks[-1], freq="7D").date) if weeks else []
    if weeks != expected:
        missing = sorted(set(expected) - set(weeks))
        errors.append(f"gap in weeks: missing {missing[:5]}")

    if (panel["count"] < 0).any():
        errors.append(f"negative counts: {int((panel['count'] < 0).sum())} rows")

    n_dup = int(panel.duplicated(["week_start", "borocd", "target"]).sum())
    if n_dup:
        errors.append(f"duplicate keys: {n_dup} rows")

    totals = (
        panel[panel["target"] == "total"].groupby("week_start")["count"].sum().astype("int64")
    )
    dropped = _dropped_by_week(raw_dir).reindex(totals.index, fill_value=0)
    city = _citywide_by_week(raw_dir).reindex(totals.index)
    mine = totals + dropped
    rel = (mine - city).abs() / city.replace(0, np.nan)
    bad = rel[(rel > RECON_TOLERANCE) | city.isna() | (city.eq(0) & mine.ne(0))]
    if len(bad):
        errors.append(
            f"reconciliation vs citywide off by >1% in {len(bad)} weeks, first: {list(bad.index[:5])}"
        )

    if errors:
        raise PanelValidationError("; ".join(errors))
