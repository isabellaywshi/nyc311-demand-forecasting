"""Rolling-origin backtest engine (spec section 5)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from nyc311.metrics import mase, wape
from nyc311.models.base import Forecaster
from nyc311.models.baselines import SeasonalNaive
from nyc311.surge import flag_surge, seasonal_norm


def make_origins(start: date, n: int, step_weeks: int, last_week: date, max_h: int) -> list[date]:
    """n origins every step_weeks from start, dropping those with origin + max_h > last_week."""
    origins = [start + timedelta(weeks=step_weeks * i) for i in range(n)]
    return [o for o in origins if o + timedelta(weeks=max_h) <= last_week]


def run_backtest(
    models: list[Forecaster],
    panel: pd.DataFrame,
    origins: list[date],
    horizons: list[int],
    period: str,
    log_mlflow: bool = True,
    reports_dir: Path | None = None,
    surge_threshold: float = 1.25,
    surge_min_norm: float = 5,
    tags: dict[str, str] | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    actuals = panel[["week_start", "borocd", "target", "count"]].rename(columns={"count": "y"})
    naive = SeasonalNaive()
    parts = []
    for origin in origins:
        hist = panel[panel["week_start"] <= origin]  # leakage boundary
        frames = [m.forecast(hist, origin, horizons) for m in models]
        if not frames:
            continue
        f = pd.concat(frames, ignore_index=True)
        weeks = sorted(f["week_start"].unique())
        norm = seasonal_norm(hist, origin, weeks)
        f = f.merge(norm, on=["borocd", "target", "week_start"], how="left")
        nv = naive.forecast(hist, origin, horizons)[["borocd", "target", "horizon", "yhat"]]
        f = f.merge(
            nv.rename(columns={"yhat": "y_naive"}), on=["borocd", "target", "horizon"], how="left"
        )
        parts.append(f)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.merge(actuals, on=["week_start", "borocd", "target"], how="left")
    out["y"] = out["y"].astype(float)
    out["surge_actual"] = flag_surge(out["y"], out["norm"], surge_threshold, surge_min_norm)
    out["surge_alert"] = flag_surge(out["yhat"], out["norm"], surge_threshold, surge_min_norm)
    out["period"] = period

    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(reports_dir / f"predictions_{period}.parquet", index=False)
    if log_mlflow:
        _log_mlflow(out, period, tags, params)
    return out


def _log_mlflow(
    out: pd.DataFrame, period: str, tags: dict[str, str] | None = None, params: dict | None = None
) -> None:
    import os

    import mlflow

    key = "MLFLOW_ALLOW_FILE_STORE"
    prior_env = os.environ.get(key)
    prior_uri = mlflow.get_tracking_uri()
    uri = os.environ.get("MLFLOW_TRACKING_URI") or (
        (Path(__file__).resolve().parents[2] / "mlruns").as_uri()
    )
    os.environ[key] = "true"  # mlruns/ file store (spec); scoped to this call
    try:
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=f"backtest_{period}"):
            mlflow.set_tag("period", period)
            for k, v in (tags or {}).items():
                mlflow.set_tag(k, v)
            for k, v in (params or {}).items():
                mlflow.log_param(k, v)
            for model, g in out.groupby("model"):
                g = g.dropna(subset=["y", "yhat"])
                with mlflow.start_run(run_name=str(model), nested=True):
                    mlflow.set_tag("period", period)
                    mlflow.log_metric("wape", wape(g["y"], g["yhat"]))
                    gm = g.dropna(subset=["y_naive"])
                    mlflow.log_metric("mase", mase(gm["y"], gm["yhat"], gm["y_naive"]))
    finally:
        mlflow.set_tracking_uri(prior_uri)
        if prior_env is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior_env
