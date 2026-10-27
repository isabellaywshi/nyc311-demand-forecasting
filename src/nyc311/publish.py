"""Final training, latest forecast and publishing of dashboard data (spec sections 7-8)."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
import requests

from nyc311.clean import REGULAR_DISTRICTS, SPECIAL_AREAS
from nyc311.config import Categories, Settings
from nyc311.explain import shap_summary
from nyc311.models.lgbm import LGBMForecaster
from nyc311.surge import flag_surge, seasonal_norm

HISTORY_WEEKS = 104
SIMPLIFY_TOLERANCE = 0.0005
GEOJSON_CACHE = "districts_raw.geojson"
ALL_AGENCIES = "All agencies"


def _model_path(settings: Settings) -> Path:
    return Path(settings.models_dir) / "lgbm.pkl"


def _forecast_path(settings: Settings) -> Path:
    return Path(settings.processed_dir) / "forecast_latest.parquet"


def _load_panel(settings: Settings) -> pd.DataFrame:
    panel = pd.read_parquet(Path(settings.processed_dir) / "weekly_panel.parquet")
    panel["week_start"] = pd.to_datetime(panel["week_start"]).dt.date
    return panel


def train(settings: Settings) -> Path:
    """Fit the final LightGBM on the whole panel and save it with joblib."""
    panel = _load_panel(settings)
    model = LGBMForecaster(settings.lgbm_params)
    model.fit(panel, panel["week_start"].max(), settings.horizons)
    path = _model_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def pct_vs_norm(yhat: pd.Series, norm: pd.Series) -> pd.Series:
    """yhat / norm - 1; NaN wherever norm is NaN or not positive."""
    return yhat / norm.where(norm > 0) - 1


def forecast(settings: Settings, cats: Categories) -> pd.DataFrame:
    """Forecast every series at the panel's last week; add norm, % vs norm, surge, agency."""
    panel = _load_panel(settings)
    model = joblib.load(_model_path(settings))
    origin = panel["week_start"].max()
    fc = model.predict(panel, origin)
    targets = [origin + timedelta(weeks=h) for h in settings.horizons]
    norm = seasonal_norm(panel, origin, targets)
    fc = fc.merge(norm, on=["borocd", "target", "week_start"], how="left")
    fc["pct_vs_norm"] = pct_vs_norm(fc["yhat"], fc["norm"])
    fc["surge_alert"] = flag_surge(
        fc["yhat"], fc["norm"], settings.surge_threshold, settings.surge_min_norm
    )
    fc["agency"] = fc["target"].map(cats.agencies).fillna(ALL_AGENCIES)
    path = _forecast_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(path, index=False)
    return fc


def _geojson(settings: Settings, out: Path) -> None:
    """Download (once) and cache the community-district boundaries; write the simplified file."""
    cache = Path(settings.raw_dir) / GEOJSON_CACHE
    if not cache.exists():
        resp = requests.get(settings.geojson_url, timeout=120)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(cache)
    gdf = gpd.read_file(cache)
    col = next((c for c in gdf.columns if c.lower() in ("borocd", "boro_cd")), None)
    if col is None:
        raise ValueError(f"no BoroCD column in {cache}: {list(gdf.columns)}")
    gdf = gdf.to_crs(4326)
    gdf = gpd.GeoDataFrame(
        {"BoroCD": gdf[col].astype(int)}, geometry=gdf.geometry.simplify(SIMPLIFY_TOLERANCE)
    )
    expected = REGULAR_DISTRICTS | SPECIAL_AREAS
    if set(gdf["BoroCD"]) != expected or len(gdf) != len(expected):
        raise ValueError(
            f"GeoJSON has {len(gdf)} features; BoroCD mismatch vs the 71 known areas: "
            f"missing={sorted(expected - set(gdf['BoroCD']))} "
            f"extra={sorted(set(gdf['BoroCD']) - expected)}"
        )
    gdf.to_file(out, driver="GeoJSON")


def _metrics(settings: Settings) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(settings.reports_dir).glob("test_*.csv")):
        frames.append(pd.read_csv(path).assign(table=path.stem))
    if not frames:
        raise FileNotFoundError(f"no test_*.csv reports in {settings.reports_dir}")
    return pd.concat(frames, ignore_index=True)


def publish(settings: Settings, cats: Categories) -> None:
    """Write the dashboard data files to app/data via a temp dir swapped in at the end."""
    final = Path(settings.app_data_dir)
    tmp = final.with_name(final.name + ".tmp")
    old = final.with_name(final.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    if old.exists():
        if final.exists():
            shutil.rmtree(old)
        else:  # a previous swap died between its two renames: old is the only good copy
            old.rename(final)
    panel = _load_panel(settings)
    fc = pd.read_parquet(_forecast_path(settings))
    if pd.to_datetime(fc["origin"]).dt.date.max() != max(panel["week_start"]):
        raise ValueError(
            f"stale forecast: origin {fc['origin'].max()} != panel last week "
            f"{max(panel['week_start'])}; run `nyc311 forecast` first"
        )
    model = joblib.load(_model_path(settings))
    try:
        tmp.mkdir(parents=True)
        fc.to_parquet(tmp / "forecasts.parquet", index=False)
        weeks = sorted(panel["week_start"].unique())[-HISTORY_WEEKS:]
        panel[panel["week_start"].isin(weeks)].to_parquet(tmp / "history.parquet", index=False)
        _metrics(settings).to_parquet(tmp / "metrics.parquet", index=False)
        shap_summary(model.models_[(1, "point")], model.train_frame_).to_parquet(
            tmp / "shap_summary.parquet", index=False
        )
        _geojson(settings, tmp / "districts.geojson")
        meta = {
            "data_through": max(panel["week_start"]).isoformat(),
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        (tmp / ".gitkeep").touch()
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    final.parent.mkdir(parents=True, exist_ok=True)
    had_old = final.exists()
    if had_old:
        final.rename(old)
    try:
        tmp.rename(final)
    except BaseException:
        if had_old:
            old.rename(final)
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)
