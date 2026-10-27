from datetime import date, timedelta

import numpy as np
import pytest

from conftest import make_panel
from nyc311.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_features

START = date(2023, 1, 2)
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
SERIES = ((101, "total"), (102, "total"))


def _weekly(i, ws, borocd, target):
    return (i * 7 + borocd) % 23 + 3 * (i % 5)


@pytest.fixture
def synthetic_panel():
    return make_panel(_weekly, START, 120, series=SERIES)


@pytest.fixture
def synthetic_panel_with_zero_series():
    p = make_panel(_weekly, START, 120, series=SERIES)
    p.loc[p["borocd"] == 102, "count"] = 0
    return p


def test_leakage_guard(synthetic_panel):
    origin = START + timedelta(weeks=80)
    hs = [1, 2, 3, 4]
    a = build_features(synthetic_panel, [origin], hs, with_label=False)
    p2 = synthetic_panel.copy()
    p2.loc[p2["week_start"] > origin, "count"] *= 1000
    b = build_features(p2, [origin], hs, with_label=False)
    assert len(a) == len(b) == 2 * 4
    cols = FEATURE_COLUMNS + ["scale"]
    assert a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))


def test_scale_floor_and_finite_on_zero_series(synthetic_panel_with_zero_series):
    origins = [START + timedelta(weeks=w) for w in (51, 60, 100)]
    f = build_features(synthetic_panel_with_zero_series, origins, [1, 4], with_label=True)
    z = f[f["borocd"] == 102]
    assert (f["scale"] >= 1.0).all() and (z["scale"] == 1.0).all()
    assert np.isfinite(f[NUMERIC]).all().all()
    assert np.isfinite(f["label"]).all()
    assert (z["roll_std_13"] == 0).all()


def test_lag_and_same_week_last_year_values(synthetic_panel):
    t_idx = 90
    origin = START + timedelta(weeks=t_idx)
    f = build_features(synthetic_panel, [origin], [2], with_label=True)
    row = f[f["borocd"] == 101].iloc[0]
    y = np.array([_weekly(i, None, 101, "total") for i in range(120)], dtype=float)
    scale = max(1.0, y[t_idx - 51 : t_idx + 1].mean())
    assert row["scale"] == pytest.approx(scale)
    for k in range(4):
        assert row[f"lag_{k + 1}"] == pytest.approx(y[t_idx - k] / scale)
    assert row["same_week_last_year"] == pytest.approx(y[t_idx + 2 - 52] / scale)
    assert row["roll_mean_4"] == pytest.approx(y[t_idx - 3 : t_idx + 1].mean() / scale)
    assert row["roll_mean_52"] == pytest.approx(1.0) or row["roll_mean_52"] > 0
    assert row["roll_std_13"] == pytest.approx(y[t_idx - 12 : t_idx + 1].std() / scale)
    assert row["label"] == pytest.approx(y[t_idx + 2] / scale)
    assert row["week_start"] == origin + timedelta(weeks=2)
    assert row["origin"] == origin
    assert row["horizon"] == 2


def test_origin_skipping_and_label_dropping(synthetic_panel):
    early = START + timedelta(weeks=50)
    ok = START + timedelta(weeks=51)
    last = START + timedelta(weeks=119)
    assert build_features(synthetic_panel, [early], [1], False).empty
    assert len(build_features(synthetic_panel, [ok], [1], False)) == 2
    assert len(build_features(synthetic_panel, [last], [1, 2], True)) == 0
    assert len(build_features(synthetic_panel, [last], [1, 2], False)) == 4


def test_holiday_count_thanksgiving_week(synthetic_panel):
    p = make_panel(lambda *a: 5, date(2024, 1, 1), 100, series=((101, "total"),))
    f = build_features(p, [date(2025, 11, 17)], [1, 2], with_label=False)
    by_h = dict(zip(f["horizon"], f["n_holidays"]))
    assert by_h[1] == 1  # week of 2025-11-24
    assert by_h[2] == 0


def test_covid_flag_bounds():
    p = make_panel(lambda *a: 5, date(2019, 1, 7), 150, series=((101, "total"),))
    origins = [date(2020, 2, 24), date(2021, 6, 28), date(2021, 6, 21)]
    f = build_features(p, origins, [1], with_label=False).set_index("week_start")
    assert f.loc[date(2020, 3, 2), "covid"] == 1
    assert f.loc[date(2021, 7, 5), "covid"] == 0
    assert f.loc[date(2021, 6, 28), "covid"] == 1
    f2 = build_features(p, [date(2020, 2, 17)], [1], False)
    assert f2["covid"].iloc[0] == 0


def test_calendar_features(synthetic_panel):
    f = build_features(synthetic_panel, [date(2023, 12, 25)], [1], False)
    r = f.iloc[0]  # target 2024-01-01 -> ISO week 1
    assert r["woy_sin"] == pytest.approx(np.sin(2 * np.pi / 52))
    assert r["woy_cos"] == pytest.approx(np.cos(2 * np.pi / 52))
    assert r["month"] == 1
    assert r["n_holidays"] == 1  # New Year's Day
