from datetime import date, timedelta

import numpy as np

from nyc311.models.base import FORECAST_COLUMNS
from nyc311.models.baselines import RecentAverage, SeasonalNaive

START = date(2023, 1, 2)


def _panel(panel_factory):
    return panel_factory(lambda i, w, b, t: i * 3 + 1, START, 80)


def test_seasonal_naive_value(panel_factory):
    p = _panel(panel_factory)
    origin = START + timedelta(weeks=70)
    out = SeasonalNaive().forecast(p, origin, [1, 2, 3, 4])
    assert list(out.columns) == FORECAST_COLUMNS
    assert (out["model"] == "seasonal_naive").all()
    r = out[out["horizon"] == 2].iloc[0]
    assert r["week_start"] == origin + timedelta(weeks=2)
    assert r["origin"] == origin
    assert r["yhat"] == (70 + 2 - 52) * 3 + 1
    assert out[["yhat_p10", "yhat_p90"]].isna().all().all()


def test_recent_average_value(panel_factory):
    p = _panel(panel_factory)
    origin = START + timedelta(weeks=70)
    out = RecentAverage().forecast(p, origin, [1, 2, 3, 4])
    expected = np.mean([(70 - k) * 3 + 1 for k in range(4)])
    assert np.allclose(out["yhat"], expected)
    assert len(out) == 4
    assert out[["yhat_p10", "yhat_p90"]].isna().all().all()
