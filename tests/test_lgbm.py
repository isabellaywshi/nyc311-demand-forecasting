from datetime import date, timedelta

import numpy as np

from nyc311.metrics import wape
from nyc311.models.base import FORECAST_COLUMNS
from nyc311.models.baselines import SeasonalNaive
from nyc311.models.lgbm import LGBMForecaster

START = date(2021, 1, 4)
N = 156
SERIES = tuple((b, t) for b in (101, 102) for t in ("total", "noise"))
PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
    "tweedie_variance_power": 1.2,
}


def _panel(panel_factory, n=N, series=SERIES, trend=0.0):
    rng = np.random.default_rng(0)
    return panel_factory(
        lambda i, w, b, t: max(
            0, round(100 + trend * i + 30 * np.sin(2 * np.pi * i / 52) + rng.normal(0, 3))
        ),
        START,
        n,
        series,
    )


def test_schema_and_finite(panel_factory):
    p = _panel(panel_factory)
    origin = START + timedelta(weeks=N - 1)
    f = LGBMForecaster(PARAMS)
    assert f.name == "lgbm"
    out = f.forecast(p, origin, [1, 4])
    assert list(out.columns) == FORECAST_COLUMNS
    assert len(out) == len(SERIES) * 2
    assert (out["model"] == "lgbm").all()
    assert np.isfinite(out[["yhat", "yhat_p10", "yhat_p90"]].to_numpy()).all()
    assert (out["yhat_p10"] >= 0).all()
    r = out[out["horizon"] == 4].iloc[0]
    assert r["week_start"] == origin + timedelta(weeks=4)
    assert r["origin"] == origin
    assert set(f.models_) == {(h, k) for h in (1, 4) for k in ("point", "p10", "p90")}
    assert len(f.train_frame_) > 0


def test_quantiles_ordered(panel_factory):
    p = _panel(panel_factory)
    out = LGBMForecaster(PARAMS).forecast(p, START + timedelta(weeks=N - 1), [1, 2, 3, 4])
    assert (out["yhat_p10"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["yhat_p90"]).all()


def test_all_zero_series_forecasts_zero(panel_factory):
    rng = np.random.default_rng(1)
    p = panel_factory(
        lambda i, w, b, t: 0 if b == 102 else max(0, round(50 + rng.normal(0, 3))),
        START,
        N,
        ((101, "total"), (102, "total")),
    )
    out = LGBMForecaster(PARAMS).forecast(p, START + timedelta(weeks=N - 1), [1, 2])
    z = out[out["borocd"] == 102]
    assert (z["yhat"].abs() < 1e-6).all()


def test_no_leakage_future_rows_ignored(panel_factory):
    p = _panel(panel_factory)
    origin = START + timedelta(weeks=120)
    a = LGBMForecaster(PARAMS).forecast(p, origin, [2])
    p2 = p.copy()
    p2.loc[p2["week_start"] > origin, "count"] = 10_000
    b = LGBMForecaster(PARAMS).forecast(p2, origin, [2])
    np.testing.assert_allclose(a["yhat"], b["yhat"])
    # training labels all observed by origin
    f = LGBMForecaster(PARAMS)
    f.fit(p, origin)
    assert (f.train_frame_["week_start"] <= origin).all()


def test_train_origin_step_subsamples(panel_factory):
    p = _panel(panel_factory)
    origin = START + timedelta(weeks=N - 1)
    f1 = LGBMForecaster(PARAMS, train_origin_step=1)
    f2 = LGBMForecaster(PARAMS, train_origin_step=2)
    f1.fit(p, origin)
    f2.fit(p, origin)
    assert len(f2.train_frame_) < 0.6 * len(f1.train_frame_)


def test_beats_seasonal_naive_with_trend(panel_factory):
    p = _panel(panel_factory, n=200, trend=1.0)
    origins = [START + timedelta(weeks=n) for n in (150, 160, 170, 180)]
    preds = {"lgbm": [], "sn": []}
    for o in origins:
        for key, m in (("lgbm", LGBMForecaster(PARAMS)), ("sn", SeasonalNaive())):
            f = m.forecast(p, o, [4])
            preds[key].append(f)
    import pandas as pd

    def score(frames):
        f = pd.concat(frames).merge(
            p.rename(columns={"count": "actual"})[["borocd", "target", "week_start", "actual"]],
            on=["borocd", "target", "week_start"],
        )
        return wape(f["actual"].to_numpy(), f["yhat"].to_numpy())

    assert score(preds["lgbm"]) < score(preds["sn"])
