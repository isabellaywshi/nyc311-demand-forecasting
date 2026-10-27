"""Shared data loading and helpers for the dashboard. Reads only app/data/ (no package imports)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

TARGETS = ["total", "noise", "heat_hot_water", "illegal_parking", "sanitation", "street_conditions"]
TARGET_LABELS = {
    "total": "All 311 requests",
    "noise": "Noise",
    "heat_hot_water": "Heat / hot water",
    "illegal_parking": "Illegal parking",
    "sanitation": "Sanitation",
    "street_conditions": "Street conditions",
}
TARGET_COLORS = {
    "total": "#4C78A8",
    "noise": "#F58518",
    "heat_hot_water": "#E45756",
    "illegal_parking": "#72B7B2",
    "sanitation": "#54A24B",
    "street_conditions": "#B279A2",
}
BOROUGHS = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
SPECIAL_AREAS = {
    164: "Central Park",
    226: "Van Cortlandt Park",
    227: "Bronx Park",
    228: "Pelham Bay Park",
    355: "Prospect Park",
    356: "Marine Park",
    480: "LaGuardia Airport",
    481: "Flushing Meadows–Corona Park",
    482: "Forest Park",
    483: "JFK Airport",
    484: "Gateway / Jamaica Bay",
    595: "Great Kills / Gateway",
}
FEATURE_LABELS = {
    "lag_1": "Last week's count",
    "lag_2": "Count two weeks ago",
    "lag_4": "Count four weeks ago",
    "lag_52": "Same week last year",
    "same_week_last_year": "Same week last year",
    "roll_mean_4": "4-week average",
    "roll_mean_8": "8-week average",
    "roll_mean_13": "13-week average",
    "roll_mean_26": "26-week average",
    "roll_mean_52": "52-week average",
    "woy_sin": "Time of year",
    "woy_cos": "Time of year",
    "n_holidays": "Holidays in week",
    "covid": "COVID period",
    "borocd": "District",
    "target": "Category",
}
CHART_HEIGHT = 420


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def data_dir() -> Path:
    env = os.environ.get("NYC311_APP_DATA")
    return Path(env) if env else Path(__file__).resolve().parent / "data"


def area_label(borocd) -> str:
    code = int(borocd)
    if code in SPECIAL_AREAS:
        return f"{SPECIAL_AREAS[code]} (special area)"
    return f"{BOROUGHS.get(code // 100, 'Unknown')} {code % 100:02d}"


def borough_name(borocd) -> str:
    code = int(borocd)
    if code in SPECIAL_AREAS:
        return BOROUGHS.get(code // 100, "")
    return BOROUGHS.get(code // 100, "")


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())


def target_label(t: str) -> str:
    return TARGET_LABELS.get(t, t)


def fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    v = round(float(x) * 100)
    return f"{v:+d}%" if v else "0%"


def fmt_int(x) -> str:
    if x is None or pd.isna(x) or math.isinf(x):
        return "n/a"
    return f"{round(float(x)):,}"


def fmt_ratio(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{float(x) * 100:.1f}%"


@dataclass
class AppData:
    forecasts: pd.DataFrame
    history: pd.DataFrame
    metrics: pd.DataFrame
    shap: pd.DataFrame
    geojson: dict
    meta: dict

    @property
    def data_through(self) -> str:
        return str(self.meta.get("data_through", "unknown"))


@st.cache_data(show_spinner=False)
def load_data(data_dir: Path) -> AppData:
    d = Path(data_dir)
    return AppData(
        forecasts=pd.read_parquet(d / "forecasts.parquet"),
        history=pd.read_parquet(d / "history.parquet"),
        metrics=pd.read_parquet(d / "metrics.parquet"),
        shap=pd.read_parquet(d / "shap_summary.parquet"),
        geojson=json.loads((d / "districts.geojson").read_text()),
        meta=json.loads((d / "meta.json").read_text()),
    )


def get_data() -> AppData:
    return load_data(data_dir())


def metrics_table(metrics: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of one metrics table; accepts 'overall' or 'test_overall'."""
    tables = metrics["table"].astype(str)
    return metrics[(tables == name) | (tables == f"test_{name}")].copy()


def surge_watch(forecasts: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Alerting areas for one target/horizon, biggest surge first."""
    f = forecasts[
        (forecasts["target"] == target)
        & (forecasts["horizon"] == horizon)
        & forecasts["surge_alert"]
    ].sort_values("pct_vs_norm", ascending=False)
    out = pd.DataFrame(
        {
            "area": f["borocd"].map(area_label),
            "borough": f["borocd"].map(borough_name),
            "target": f["target"],
            "week_start": f["week_start"],
            "yhat": f["yhat"],
            "norm": f["norm"],
            "pct_vs_norm": f["pct_vs_norm"],
            "agency": f["agency"],
        }
    )
    return out.reset_index(drop=True)


def area_options(forecasts: pd.DataFrame) -> list[int]:
    return sorted(forecasts["borocd"].astype(int).unique().tolist())


def setup_page(title: str, data: AppData | None = None, icon: str = "\U0001f4de") -> AppData:
    """Common page chrome: config, title and the data-through caption."""
    st.set_page_config(
        layout="wide", page_title=f"{title} · NYC 311 Demand Forecast", page_icon=icon
    )
    data = data or get_data()
    st.title(title)
    st.caption(f"Data through week of {data.data_through}")
    return data
