from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
TARGETS = ["total", "noise", "heat_hot_water", "illegal_parking", "sanitation", "street_conditions"]
AREAS = {204: "BRONX", 101: "MANHATTAN", 164: "MANHATTAN"}  # 164 = Central Park (special)
NAN_AREA = 164


def _square(code, i):
    x, y = -74.0 + 0.05 * i, 40.6
    ring = [[x, y], [x + 0.04, y], [x + 0.04, y + 0.04], [x, y + 0.04], [x, y]]
    return {
        "type": "Feature",
        "properties": {"BoroCD": code},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    import json

    d = tmp_path / "data"
    d.mkdir()
    through = date(2026, 10, 19)
    weeks = [through - timedelta(weeks=k) for k in range(103, -1, -1)]
    hist, fc = [], []
    rng = np.random.default_rng(0)
    for code, boro in AREAS.items():
        for t in TARGETS:
            for w in weeks:
                hist.append((w, code, t, int(50 + rng.integers(0, 20)), code == 164, boro))
            for h in range(1, 5):
                nan = code == NAN_AREA
                norm = np.nan if nan else 50.0
                y = 80.0
                fc.append(
                    (
                        code,
                        t,
                        through,
                        h,
                        through + timedelta(weeks=h),
                        "lgbm",
                        y,
                        y - 10,
                        y + 10,
                        norm,
                        np.nan if nan else y / 50 - 1,
                        (not nan) and y / 50 - 1 >= 0.25,
                        "All agencies",
                    )
                )
    cols = [
        "borocd",
        "target",
        "origin",
        "horizon",
        "week_start",
        "model",
        "yhat",
        "yhat_p10",
        "yhat_p90",
        "norm",
        "pct_vs_norm",
        "surge_alert",
        "agency",
    ]
    pd.DataFrame(fc, columns=cols).astype({"borocd": "int16"}).to_parquet(d / "forecasts.parquet")
    pd.DataFrame(
        hist, columns=["week_start", "borocd", "target", "count", "is_special_area", "borough"]
    ).to_parquet(d / "history.parquet")
    rows = []
    for tbl, extra in [
        ("test_overall", {}),
        ("test_by_horizon", {"horizon": 1.0}),
        ("test_by_horizon", {"horizon": 2.0}),
        ("test_by_borough", {"borough": "BRONX"}),
        ("test_by_target", {"target": "noise"}),
        ("test_by_area_type", {"area_type": "regular"}),
    ]:
        for m in ["seasonal_naive", "lgbm"]:
            rows.append(
                {
                    "table": tbl,
                    "model": m,
                    "wape": 0.12,
                    "mase": 0.6,
                    "coverage": 0.8 if m == "lgbm" else np.nan,
                    "width": 100.0,
                    **extra,
                }
            )
    rows.append(
        {
            "table": "test_surge",
            "model": "lgbm",
            "precision": 0.66,
            "recall": 0.67,
            "f1": 0.67,
            "n_actual": 100.0,
            "n_alerts": 99.0,
        }
    )
    pd.DataFrame(rows).to_parquet(d / "metrics.parquet")
    shap = [("global", None, None, "lag_1", 0.3, 1), ("target", None, "noise", "woy_sin", 0.2, 1)]
    shap += [
        ("area_target", c, t, f, 0.1 * (6 - r), r)
        for c in AREAS
        for t in TARGETS
        for r, f in enumerate(["lag_1", "roll_mean_52"], 1)
    ]
    pd.DataFrame(
        shap, columns=["level", "borocd", "target", "feature", "mean_abs_shap", "rank"]
    ).astype({"borocd": "Int16", "target": "string"}).to_parquet(d / "shap_summary.parquet")
    (d / "districts.geojson").write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": [_square(c, i) for i, c in enumerate(AREAS)]}
        )
    )
    (d / "meta.json").write_text(
        json.dumps({"data_through": "2026-10-19", "generated_at": "2026-10-27T00:00:00+00:00"})
    )
    monkeypatch.setenv("NYC311_APP_DATA", str(d))
    return d


def _fc(**kw):
    base = {
        "borocd": 204,
        "target": "total",
        "week_start": date(2026, 10, 26),
        "horizon": 1,
        "yhat": 100.0,
        "norm": 70.0,
        "pct_vs_norm": 0.43,
        "surge_alert": True,
        "agency": "All agencies",
    }
    base.update(kw)
    return base


def test_surge_watch_filters_and_sorts():
    import sys
    from importlib import import_module

    sys.path.insert(0, str(APP))
    lib = import_module("lib")
    fc = pd.DataFrame(
        [
            _fc(borocd=101, pct_vs_norm=0.30),
            _fc(borocd=204, pct_vs_norm=0.90),
            _fc(borocd=164, pct_vs_norm=np.nan, surge_alert=False),
            _fc(borocd=105, surge_alert=False, pct_vs_norm=0.1),
            _fc(borocd=106, horizon=2, pct_vs_norm=2.0),
            _fc(borocd=107, target="noise", pct_vs_norm=2.0),
        ]
    )
    out = lib.surge_watch(fc, "total", 1)
    assert list(out.columns) == [
        "area",
        "borough",
        "target",
        "week_start",
        "yhat",
        "norm",
        "pct_vs_norm",
        "agency",
    ]
    assert list(out["area"]) == ["Bronx 04", "Manhattan 01"]
    assert list(out["borough"]) == ["Bronx", "Manhattan"]


def test_labels_and_formatting():
    import sys

    sys.path.insert(0, str(APP))
    import lib

    assert lib.area_label(204) == "Bronx 04"
    assert lib.area_label(355) == "Prospect Park (special area)"
    assert lib.area_label(595) == "Great Kills / Gateway (special area)"
    assert lib.fmt_pct(float("nan")) == "n/a"
    assert lib.fmt_pct(0.4321) == "+43%"
    assert lib.fmt_pct(-0.1) == "-10%"
    assert lib.fmt_int(12345.6) == "12,346"
    assert lib.fmt_int(float("nan")) == "n/a"
    assert lib.feature_label("lag_1") == "Last week's count"
    assert lib.feature_label("something_new") == "Something new"
    assert set(lib.TARGET_COLORS) == set(TARGETS)


PAGES = [
    "Home.py",
    "pages/1_District_Detail.py",
    "pages/2_Model_Performance.py",
    "pages/3_About.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_pages_run(data_dir, page):
    at = AppTest.from_file(str(APP / page), default_timeout=60).run()
    assert not at.exception
    assert any("Data through week of 2026-10-19" in c.value for c in at.caption)


def test_home_nan_norm_area_shows_na(data_dir):
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    assert not at.exception
    at.selectbox(key="home_area").select(NAN_AREA).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "n/a" in text
    assert "Central Park" in text


def test_detail_nan_norm_area_shows_na(data_dir):
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60)
    at.query_params["area"] = str(NAN_AREA)
    at.run()
    assert not at.exception
    assert at.selectbox(key="detail_area").value == NAN_AREA
    text = " ".join(m.value for m in at.markdown)
    assert "n/a" in text


def test_detail_other_target(data_dir):
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60).run()
    at.selectbox(key="detail_target").select("noise").run()
    assert not at.exception


def test_app_does_not_import_package():
    for p in APP.rglob("*.py"):
        assert "nyc311" not in p.read_text().replace("NYC311_APP_DATA", "").replace(
            "nyc311-", ""
        ).replace("nyc311_", ""), p


# ---- branch coverage -------------------------------------------------------------------


def _rewrite(path, fn):
    df = pd.read_parquet(path)
    fn(df).to_parquet(path)


def _figure_trace_names(at, key_hint=None):
    import json

    names = []
    for el in at.get("plotly_chart"):
        spec = json.loads(el.proto.spec)
        names.append([t.get("name") for t in spec["data"]])
    return names


def test_home_empty_watch_shows_info(data_dir):
    _rewrite(data_dir / "forecasts.parquet", lambda d: d.assign(surge_alert=False))
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No districts are flagged" in i.value for i in at.info)


def test_home_map_has_grey_layer_for_nan_areas(data_dir):
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No norm" in (n or "") for n in _figure_trace_names(at)[0])


def test_detail_shap_falls_back_to_target_then_global(data_dir):
    _rewrite(data_dir / "shap_summary.parquet", lambda d: d[d["level"] != "area_target"])
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60)
    at.query_params["area"] = "164"
    at.query_params["target"] = "noise"
    at.run()
    assert not at.exception
    assert any("all districts, Noise" in c.value for c in at.caption)


def test_detail_shap_falls_back_to_global(data_dir):
    _rewrite(
        data_dir / "shap_summary.parquet",
        lambda d: d[~d["level"].isin(["area_target", "target"])],
    )
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60)
    at.query_params["area"] = "164"
    at.query_params["target"] = "noise"
    at.run()
    assert not at.exception
    assert any("all districts and categories" in c.value for c in at.caption)


def test_detail_shap_empty_shows_info(data_dir):
    _rewrite(data_dir / "shap_summary.parquet", lambda d: d.iloc[0:0])
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No driver summary" in i.value for i in at.info)


def test_detail_merges_shared_labels(data_dir):
    def swap(d):
        d = d[d["level"] != "area_target"]
        rows = [
            ("area_target", 204, "total", f, 0.1 * (i + 1), i + 1)
            for i, f in enumerate(
                ["woy_sin", "woy_cos", "lag_1", "roll_mean_52", "covid", "n_holidays"]
            )
        ]
        extra = pd.DataFrame(rows, columns=d.columns).astype(d.dtypes.to_dict())
        return pd.concat([d, extra], ignore_index=True)

    _rewrite(data_dir / "shap_summary.parquet", swap)
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60)
    at.query_params["area"] = "204"
    at.query_params["target"] = "total"
    at.run()
    assert not at.exception
    import json

    spec = json.loads(at.get("plotly_chart")[1].proto.spec)
    labels = list(spec["data"][0]["y"])
    assert len(labels) == 5 and labels.count("Time of year") == 1


def test_map_select_callback_updates_session(monkeypatch):
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(APP))
    import lib

    state = {"home_area": 204}
    monkeypatch.setattr(lib.st, "session_state", state, raising=False)
    sel = SimpleNamespace(selection=SimpleNamespace(points=[{"location": 164}]))
    state["home_map"] = sel
    lib.on_map_select()
    assert state["home_area"] == 164
    state["home_map"] = SimpleNamespace(selection=SimpleNamespace(points=[]))
    lib.on_map_select()
    assert state["home_area"] == 164
    assert lib.area_from_selection(None) is None


def test_detail_nan_norm_has_no_norm_trace(data_dir):
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60)
    at.query_params["area"] = str(NAN_AREA)
    at.run()
    assert not at.exception
    assert "Seasonal norm" not in _figure_trace_names(at)[0]
    at = AppTest.from_file(str(APP / "pages/1_District_Detail.py"), default_timeout=60)
    at.query_params["area"] = "204"
    at.run()
    assert "Seasonal norm" in _figure_trace_names(at)[0]


def test_about_copy_derived(data_dir):
    at = AppTest.from_file(str(APP / "pages/3_About.py"), default_timeout=60).run()
    text = " ".join(m.value for m in at.markdown)
    assert "to 2026-10-19" in text and "12 parks" in text
    assert "can inflate" in text and "no weather" in text.lower()
