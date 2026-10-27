import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from nyc311 import cli
from nyc311.clean import REGULAR_DISTRICTS, SPECIAL_AREAS
from nyc311.config import load_settings
from nyc311.publish import forecast, publish, train

BOROCDS = sorted(REGULAR_DISTRICTS | SPECIAL_AREAS)
TARGETS = ["total", "noise", "heat_hot_water", "illegal_parking", "sanitation", "street_conditions"]
N_WEEKS = 120
START = date(2024, 1, 1)
EXPECTED = {
    "forecasts.parquet", "history.parquet", "metrics.parquet",
    "shap_summary.parquet", "districts.geojson", "meta.json",
}  # fmt: skip


def _square(i):
    x, y = (i % 10) * 0.01, (i // 10) * 0.01
    ring = [[x, y], [x + 0.01, y], [x + 0.01, y + 0.01], [x, y + 0.01], [x, y]]
    return ring


def _write_geojson(path, borocds):
    feats = [
        {
            "type": "Feature",
            "properties": {"boro_cd": str(b), "shape_area": "1.0"},
            "geometry": {"type": "Polygon", "coordinates": [_square(i)]},
        }
        for i, b in enumerate(borocds)
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    raw = yaml.safe_load(Path("config/settings.yaml").read_text())
    raw.update(
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        reports_dir=str(tmp_path / "reports"),
        app_data_dir=str(tmp_path / "app" / "data"),
        models_dir=str(tmp_path / "models"),
        lgbm_params={**raw["lgbm_params"], "n_estimators": 8},
    )
    for d in ("raw", "processed", "reports", "app/data"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "app" / "data" / ".gitkeep").touch()
    _write_geojson(tmp_path / "raw" / "districts_raw.geojson", BOROCDS)
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("NYC311_SETTINGS", str(cfg))

    rng = np.random.default_rng(0)
    rows = []
    for b in BOROCDS:
        for t in TARGETS:
            base = rng.integers(20, 200)
            for i in range(N_WEEKS):
                ws = START + timedelta(weeks=i)
                rows.append((ws, b, t, int(base * (1 + 0.3 * np.sin(i / 8.3)) + rng.poisson(5))))
    panel = pd.DataFrame(rows, columns=["week_start", "borocd", "target", "count"])
    panel["borocd"] = panel["borocd"].astype("int16")
    panel["is_special_area"] = panel["borocd"].isin(SPECIAL_AREAS)
    panel["borough"] = "MANHATTAN"
    panel.to_parquet(tmp_path / "processed" / "weekly_panel.parquet", index=False)
    pd.DataFrame({"model": ["lgbm"], "wape": [0.1]}).to_csv(
        tmp_path / "reports" / "test_overall.csv", index=False
    )
    pd.DataFrame({"horizon": [1, 2], "model": ["lgbm"] * 2, "wape": [0.1, 0.2]}).to_csv(
        tmp_path / "reports" / "test_by_horizon.csv", index=False
    )
    return tmp_path


@pytest.fixture
def published(env, cats):
    settings = load_settings()
    train(settings)
    forecast(settings, cats)
    publish(settings, cats)
    return env / "app" / "data"


def test_publish_writes_all_files(published):
    assert {p.name for p in published.iterdir()} == EXPECTED | {".gitkeep"}


def test_geojson_small_with_71_int_borocds(published):
    path = published / "districts.geojson"
    assert path.stat().st_size < 1_000_000
    feats = json.loads(path.read_text())["features"]
    assert len(feats) == 71
    assert {f["properties"]["BoroCD"] for f in feats} == set(BOROCDS)
    assert all(list(f["properties"]) == ["BoroCD"] for f in feats)


def test_forecasts_shape_and_columns(published):
    fc = pd.read_parquet(published / "forecasts.parquet")
    assert len(fc) == 71 * 6 * 4
    for col in ("norm", "pct_vs_norm", "surge_alert", "agency", "yhat_p10", "yhat_p90"):
        assert col in fc.columns
    assert fc["agency"].notna().all()
    assert (fc.loc[fc["target"] == "total", "agency"] == "All agencies").all()


def test_pct_vs_norm_nan_iff_norm_nan(published):
    fc = pd.read_parquet(published / "forecasts.parquet")
    assert fc["norm"].notna().any()
    assert (fc["pct_vs_norm"].isna() == fc["norm"].isna()).all()
    ok = fc["norm"].notna()
    np.testing.assert_allclose(fc.loc[ok, "pct_vs_norm"], fc.loc[ok, "yhat"] / fc.loc[ok, "norm"] - 1)
    assert not fc.loc[fc["norm"].isna(), "surge_alert"].any()


def test_history_and_metrics_and_meta(published, env):
    hist = pd.read_parquet(published / "history.parquet")
    assert hist["week_start"].nunique() == 104
    assert len(hist) == 104 * 71 * 6
    metrics = pd.read_parquet(published / "metrics.parquet")
    assert set(metrics["table"]) == {"test_overall", "test_by_horizon"}
    meta = json.loads((published / "meta.json").read_text())
    assert meta["data_through"] == (START + timedelta(weeks=N_WEEKS - 1)).isoformat()
    assert "generated_at" in meta
    shap_df = pd.read_parquet(published / "shap_summary.parquet")
    assert {"global", "target", "area_target"} <= set(shap_df["level"])


def test_train_saves_model(env):
    train(load_settings())
    assert (env / "models" / "lgbm.pkl").exists()


def test_bad_geojson_leaves_app_data_untouched(env, cats):
    settings = load_settings()
    train(settings)
    forecast(settings, cats)
    app_data = env / "app" / "data"
    (app_data / "meta.json").write_text("old")
    _write_geojson(env / "raw" / "districts_raw.geojson", BOROCDS[:5])
    with pytest.raises(ValueError):
        publish(settings, cats)
    assert (app_data / "meta.json").read_text() == "old"
    assert (app_data / ".gitkeep").exists()


def test_refresh_failing_at_panel_leaves_app_data_unchanged(env, monkeypatch):
    app_data = env / "app" / "data"
    (app_data / "meta.json").write_text("old")

    def boom():
        raise RuntimeError("panel failed")

    monkeypatch.setattr(cli, "ingest_cmd", lambda today=None: None)
    monkeypatch.setattr(cli, "panel_cmd", boom)
    result = CliRunner().invoke(cli.app, ["refresh"])
    assert result.exit_code != 0
    assert {p.name for p in app_data.iterdir()} == {"meta.json", ".gitkeep"}
    assert (app_data / "meta.json").read_text() == "old"


def test_pct_vs_norm_nan_for_nan_or_zero_norm():
    from nyc311.publish import pct_vs_norm

    out = pct_vs_norm(pd.Series([5.0, 0.0, 3.0, 4.0]), pd.Series([0.0, 0.0, np.nan, 2.0]))
    assert out.isna().tolist() == [True, True, True, False]
    assert not np.isinf(out.dropna()).any()
    assert out.iloc[3] == 1.0


def test_failed_swap_restores_app_data(env, cats, monkeypatch):
    settings = load_settings()
    train(settings)
    forecast(settings, cats)
    app_data = env / "app" / "data"
    (app_data / "meta.json").write_text("old")
    real = Path.rename

    def flaky(self, target):
        if self.name == "data.tmp":
            raise OSError("boom")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(OSError):
        publish(settings, cats)
    monkeypatch.undo()
    assert (app_data / "meta.json").read_text() == "old"
    assert (app_data / ".gitkeep").exists()
    assert not (env / "app" / "data.old").exists()


def test_publish_restores_orphaned_old_dir(env, cats):
    settings = load_settings()
    train(settings)
    forecast(settings, cats)
    app_data = env / "app" / "data"
    app_data.rename(env / "app" / "data.old")  # crashed between the two renames
    (env / "app" / "data.old" / "meta.json").write_text("old")
    _write_geojson(env / "raw" / "districts_raw.geojson", BOROCDS[:5])
    with pytest.raises(ValueError):
        publish(settings, cats)
    assert (app_data / "meta.json").read_text() == "old"


def test_stale_forecast_rejected(env, cats):
    settings = load_settings()
    train(settings)
    forecast(settings, cats)
    panel = pd.read_parquet(env / "processed" / "weekly_panel.parquet")
    last = pd.to_datetime(panel["week_start"]).max()
    panel[pd.to_datetime(panel["week_start"]) < last].to_parquet(
        env / "processed" / "weekly_panel.parquet", index=False
    )
    with pytest.raises(ValueError, match="stale"):
        publish(settings, cats)
