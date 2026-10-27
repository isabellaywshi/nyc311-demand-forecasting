import warnings
from datetime import date, timedelta

import numpy as np
import pytest

from nyc311.models.base import FORECAST_COLUMNS
from nyc311.models.statistical import StatsForecaster

START = date(2022, 1, 3)
N = 156
SERIES = tuple((b, t) for b in (101, 102, 103) for t in ("total", "noise"))


def _panel(panel_factory, n=N, series=SERIES):
    rng = np.random.default_rng(0)
    return panel_factory(
        lambda i, w, b, t: max(0, round(100 + 30 * np.sin(2 * np.pi * i / 52) + rng.normal(0, 3))),
        START,
        n,
        series,
    )


@pytest.mark.parametrize("kind,name", [("ets", "auto_ets"), ("mstl", "mstl")])
def test_shape_finite_nonneg(panel_factory, kind, name):
    p = _panel(panel_factory)
    origin = START + timedelta(weeks=N - 1)
    f = StatsForecaster(kind)
    assert f.name == name
    out = f.forecast(p, origin, [1, 2, 3, 4])
    assert list(out.columns) == FORECAST_COLUMNS
    assert len(out) == 3 * 2 * 4
    assert (out["model"] == name).all()
    assert np.isfinite(out[["yhat", "yhat_p10", "yhat_p90"]].to_numpy()).all()
    assert (out[["yhat", "yhat_p10", "yhat_p90"]] >= 0).all().all()
    r = out[out["horizon"] == 2].iloc[0]
    assert r["week_start"] == origin + timedelta(weeks=2)
    assert r["origin"] == origin


def test_mstl_peak_exceeds_trough(panel_factory):
    p = _panel(panel_factory, series=((101, "total"),))
    # origin at week index 100; seasonal trough near index 91 (39 mod 52), peak near 117 (13 mod 52)
    out = StatsForecaster("mstl").forecast(p, START + timedelta(weeks=100), list(range(1, 21)))
    by_i = {(r.week_start - START).days // 7: r.yhat for r in out.itertuples()}
    assert by_i[117] > by_i[104] + 20  # 104 = 0 mod 52 (mid-level), 117 = peak


def test_mstl_short_series_fallback(panel_factory):
    p = _panel(panel_factory, n=60, series=((101, "total"), (102, "total")))
    out = StatsForecaster("mstl").forecast(p, START + timedelta(weeks=59), [1, 2])
    assert len(out) == 4 and np.isfinite(out["yhat"]).all()
    assert (out["model"] == "mstl").all()


def test_uses_only_data_up_to_origin(panel_factory):
    p = _panel(panel_factory, series=((101, "total"),))
    origin = START + timedelta(weeks=120)
    a = StatsForecaster("ets").forecast(p, origin, [1, 2])
    b = StatsForecaster("ets").forecast(p[p["week_start"] <= origin], origin, [1, 2])
    np.testing.assert_allclose(a["yhat"], b["yhat"])


@pytest.mark.parametrize("kind", ["ets", "mstl"])
def test_all_zeros(panel_factory, kind):
    p = panel_factory(lambda i, w, b, t: 0, START, N, ((101, "total"),))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = StatsForecaster(kind).forecast(p, START + timedelta(weeks=N - 1), [1, 2, 3, 4])
    assert len(out) == 4
    assert (out[["yhat", "yhat_p10", "yhat_p90"]] == 0).all().all()
