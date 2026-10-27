import itertools
from datetime import date, timedelta

import pandas as pd

from nyc311.backtest import make_origins, run_backtest
from nyc311.models.base import forecast_frame
from nyc311.models.baselines import RecentAverage, SeasonalNaive

START = date(2022, 1, 3)


def test_make_origins():
    o = make_origins(date(2024, 11, 4), 13, 4, last_week=date(2025, 3, 3), max_h=4)
    assert o[0] == date(2024, 11, 4)
    assert all((b - a).days == 28 for a, b in itertools.pairwise(o))
    assert o[-1] + timedelta(weeks=4) <= date(2025, 3, 3)
    assert len(o) < 13


class Spy:
    name = "spy"

    def __init__(self):
        self.seen = []

    def forecast(self, panel, origin, horizons):
        self.seen.append((max(panel["week_start"]), origin))
        rows = panel[panel["week_start"] == origin]
        return forecast_frame(
            rows["borocd"].to_numpy(),
            rows["target"].to_numpy(),
            origin,
            1,
            origin + timedelta(weeks=1),
            self.name,
            rows["count"].to_numpy().astype(float),
        )


def _panel(panel_factory, f=lambda i, w, b, t: 50):
    return panel_factory(f, START, 160)


def test_backtest_columns_and_no_future_access(panel_factory):
    p = _panel(panel_factory)
    spy = Spy()
    origins = [START + timedelta(weeks=w) for w in (110, 114)]
    out = run_backtest([spy, SeasonalNaive()], p, origins, [1, 2], "val", log_mlflow=False)
    assert spy.seen and all(mx <= o for mx, o in spy.seen)
    for c in ["y", "norm", "surge_actual", "surge_alert", "y_naive", "period", "yhat_p10"]:
        assert c in out.columns
    assert (out["period"] == "val").all()
    n_series = p[["borocd", "target"]].drop_duplicates().shape[0]
    # spy emits h=1 only; seasonal_naive emits both requested horizons
    assert len(out) == n_series * len(origins) * (1 + 2)
    key = ["model", "borocd", "target", "origin", "horizon"]
    assert not out.duplicated(key).any()


def test_backtest_joins_actuals_and_surge(panel_factory, tmp_path):
    spike = START + timedelta(weeks=112)
    p = _panel(panel_factory, lambda i, w, b, t: 200 if w == spike else 50)
    origin = START + timedelta(weeks=110)
    out = run_backtest(
        [RecentAverage(), SeasonalNaive()],
        p,
        [origin],
        [1, 2, 3],
        "val",
        log_mlflow=False,
        reports_dir=tmp_path,
    )
    ra = out[(out["model"] == "recent_average")].set_index("horizon")
    assert ra.loc[2, "y"] == 200 and ra.loc[1, "y"] == 50
    assert bool(ra.loc[2, "surge_actual"]) and not bool(ra.loc[1, "surge_actual"])
    assert ra.loc[2, "norm"] == 50.0
    assert not ra["surge_alert"].any()
    assert (ra["y_naive"] == 50).all()
    assert (tmp_path / "predictions_val.parquet").exists()
    assert len(pd.read_parquet(tmp_path / "predictions_val.parquet")) == len(out)


def test_log_mlflow_restores_env_and_uri(panel_factory, tmp_path, monkeypatch):
    import os

    import pytest

    mlflow = pytest.importorskip("mlflow")
    from nyc311.backtest import _log_mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    before_uri = mlflow.get_tracking_uri()
    p = panel_factory(lambda i, w, b, t: 50, START, 160)
    origin = START + timedelta(weeks=110)
    out = run_backtest([SeasonalNaive()], p, [origin], [1], "val", log_mlflow=False)
    _log_mlflow(out, "val")
    assert os.environ.get("MLFLOW_ALLOW_FILE_STORE") is None
    assert mlflow.get_tracking_uri() == before_uri
    assert (tmp_path / "mlruns").exists()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    _log_mlflow(out, "val")
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "false"
