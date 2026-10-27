from datetime import date

import numpy as np
import pandas as pd
import pytest

from nyc311.surge import flag_surge, seasonal_norm

START = date(2022, 1, 3)  # Monday, ISO week 1 of 2022


def norm_of(panel, origin, targets):
    return seasonal_norm(panel, origin, targets)["norm"].tolist()


def test_flag_surge_thresholds():
    v = pd.Series([125, 124, 100, 200, 200])
    n = pd.Series([100, 100, 4, np.nan, 5.0])
    assert flag_surge(v, n).tolist() == [True, False, False, False, True]


def test_norm_constant_series(panel_factory):
    p = panel_factory(lambda i, w, b, t: 50, START, 160)
    origin = date(2024, 12, 30)
    out = seasonal_norm(p, origin, [date(2025, 1, 6), date(2025, 1, 27)])
    assert list(out.columns) == ["borocd", "target", "week_start", "norm"]
    assert out["norm"].tolist() == [50.0, 50.0]


def test_norm_level_adjustment_clipped(panel_factory):
    # last 13 weeks are 10x the same weeks a year earlier -> factor clipped to 2.0
    n = 160
    origin_i = 150
    p = panel_factory(lambda i, w, b, t: 100 if origin_i - 12 <= i <= origin_i else 10, START, n)
    origin = START + pd.Timedelta(weeks=origin_i).to_pytimedelta()
    target = origin + pd.Timedelta(weeks=1).to_pytimedelta()
    out = seasonal_norm(p, origin, [target])
    base = 10.0  # prior-year same-week mean (target week is not in the boosted window)
    assert out["norm"].iloc[0] == pytest.approx(base * 2.0)


def test_norm_uses_only_past(panel_factory):
    p = panel_factory(lambda i, w, b, t: 10 + (i % 52), START, 160)
    origin = date(2024, 12, 30)
    targets = [date(2025, 1, 6)]
    a = seasonal_norm(p, origin, targets)
    q = p.copy()
    q.loc[q["week_start"] > origin, "count"] = 99999
    b = seasonal_norm(q, origin, targets)
    pd.testing.assert_frame_equal(a, b)


def test_norm_week_53_and_year_boundary(panel_factory):
    # counts = 1000*ISO-year-offset + week-of-year, so each cell identifies its source
    def f(i, w, b, t):
        y, k, _ = w.isocalendar()
        return (y - 2000) * 100 + min(k, 52)

    p = panel_factory(f, date(2020, 1, 6), 370)
    origin = date(2026, 12, 21)
    out = seasonal_norm(p, origin, [date(2026, 12, 28), date(2027, 1, 4)])
    # 2026-12-28 is ISO 2026-W53 -> week-52 history of 2020..2025 (factor 1 shape: level differs)
    assert out["norm"].notna().all()
    s = p.set_index("week_start")["count"]
    # factor: recent 13 vs same 13 a year before
    idx = [origin - pd.Timedelta(weeks=j).to_pytimedelta() for j in range(13)]
    cur = np.mean([s[d] for d in idx])
    prev = np.mean([s[d - pd.Timedelta(weeks=52).to_pytimedelta()] for d in idx])
    fac = float(np.clip(cur / prev, 0.5, 2.0))
    w52 = np.mean([
        s[d] for d in s.index
        if d.isocalendar()[0] < 2026 and min(d.isocalendar()[1], 52) == 52
    ])
    w1 = np.mean([
        s[d] for d in s.index
        if d.isocalendar()[0] < 2027 and d.isocalendar()[1] == 1
    ])
    assert out["norm"].iloc[0] == pytest.approx(w52 * fac)
    assert out["norm"].iloc[1] == pytest.approx(w1 * fac)


def test_norm_nan_without_prior_year(panel_factory):
    p = panel_factory(lambda i, w, b, t: 50, START, 40)
    origin = START + pd.Timedelta(weeks=35).to_pytimedelta()
    out = seasonal_norm(p, origin, [origin + pd.Timedelta(weeks=1).to_pytimedelta()])
    assert out["norm"].isna().all()


def test_norm_multiple_series(panel_factory):
    p = panel_factory(lambda i, w, b, t: b, START, 160, series=((101, "a"), (102, "b")))
    out = seasonal_norm(p, date(2024, 12, 30), [date(2025, 1, 6)])
    assert dict(zip(out["borocd"], out["norm"])) == {101: 101.0, 102: 102.0}
