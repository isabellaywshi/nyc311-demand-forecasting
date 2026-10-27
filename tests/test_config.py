import subprocess
from datetime import date
from pathlib import Path

from nyc311.config import load_categories, load_settings


def test_load_settings_values():
    s = load_settings()
    assert s.horizons == [1, 2, 3, 4]
    assert s.validation_start == date(2024, 11, 4) and s.test_start == date(2025, 11, 3)
    assert s.surge_threshold == 1.25 and s.surge_min_norm == 5
    assert s.start_date == date(2020, 1, 1)
    assert (s.n_origins, s.origin_step_weeks) == (13, 4)
    assert s.raw_dir == Path("data/raw") and s.app_data_dir == Path("app/data")
    assert s.lgbm_params["tweedie_variance_power"] == 1.2


def test_load_settings_env_override(tmp_path, monkeypatch):
    src = Path("config/settings.yaml").read_text().replace("n_origins: 13", "n_origins: 7")
    alt = tmp_path / "alt.yaml"
    alt.write_text(src)
    monkeypatch.setenv("NYC311_SETTINGS", str(alt))
    assert load_settings().n_origins == 7
    other = tmp_path / "other.yaml"
    other.write_text(src.replace("n_origins: 7", "n_origins: 3"))
    assert load_settings(other).n_origins == 3  # explicit path wins over env


def test_load_categories():
    c = load_categories()
    assert c.targets == [
        "total", "noise", "heat_hot_water", "illegal_parking", "sanitation", "street_conditions",
    ]
    assert c.exact["heat/hot water"] == "heat_hot_water"
    assert c.exact["residential disposal complaint"] == "sanitation"
    assert c.exact["broken parking meter"] == "street_conditions"
    assert ("noise", "noise") in c.prefixes
    assert c.agencies["sanitation"] == "DSNY"


def _ignored(path: str) -> int:
    return subprocess.run(["git", "check-ignore", "-q", path], check=False).returncode


def test_app_data_not_ignored():
    assert _ignored("app/data/forecasts.parquet") == 1
    assert _ignored("data/raw/x.parquet") == 0
