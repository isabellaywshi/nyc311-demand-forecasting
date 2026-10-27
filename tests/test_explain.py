from datetime import date, timedelta

import numpy as np
import pandas as pd

from nyc311.explain import shap_summary
from nyc311.models.lgbm import LGBMForecaster

START = date(2021, 1, 4)
N = 156
PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 20,
    "tweedie_variance_power": 1.2,
}

SERIES = tuple((b, t) for b in (101, 102) for t in ("total", "noise"))


def _fit(panel_factory, weekly):
    p = panel_factory(weekly, START, N, SERIES)
    f = LGBMForecaster(PARAMS)
    f.fit(p, START + timedelta(weeks=N - 1), [1, 2])
    return f


def _walk_panel(panel_factory):
    rng = np.random.default_rng(0)
    state = {}

    def weekly(i, w, b, t):
        prev = state.get((b, t), 200.0)
        cur = max(1.0, prev + rng.normal(0, 25))
        state[(b, t)] = cur
        return round(cur)

    return _fit(panel_factory, weekly)


def test_levels_ranks_and_values(panel_factory):
    f = _walk_panel(panel_factory)
    out = shap_summary(f.models_[(1, "point")], f.train_frame_, sample=500)
    assert set(out["level"]) == {"global", "target", "area_target"}
    assert list(out.columns) == ["level", "borocd", "target", "feature", "mean_abs_shap", "rank"]
    assert (out["mean_abs_shap"] >= 0).all()
    for _, g in out.groupby(["level", "borocd", "target"], dropna=False):
        assert sorted(g["rank"]) == [1, 2, 3, 4, 5]
    g = out[out["level"] == "global"]
    assert g["borocd"].isna().all() and g["target"].isna().all()
    assert out[out["level"] == "target"]["borocd"].isna().all()
    assert out[out["level"] == "area_target"].groupby(["borocd", "target"]).ngroups == 4
    assert str(out["borocd"].dtype) == "Int16"
    assert str(out["target"].dtype) == "string"


def test_lag_1_is_global_top_driver(panel_factory):
    f = _walk_panel(panel_factory)
    out = shap_summary(f.models_[(1, "point")], f.train_frame_, sample=500)
    top = out[(out["level"] == "global") & (out["rank"] == 1)]
    assert top["feature"].iloc[0] == "lag_1"


def test_deterministic_sampling(panel_factory):
    f = _walk_panel(panel_factory)
    m, tf = f.models_[(1, "point")], f.train_frame_
    pd.testing.assert_frame_equal(
        shap_summary(m, tf, sample=100, seed=3), shap_summary(m, tf, sample=100, seed=3)
    )
