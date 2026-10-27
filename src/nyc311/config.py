"""Typed loaders for config/settings.yaml and config/categories.yaml."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")
DEFAULT_CATEGORIES_PATH = Path("config/categories.yaml")
SETTINGS_ENV_VAR = "NYC311_SETTINGS"


@dataclass(frozen=True)
class Settings:
    start_date: date
    horizons: list[int]
    validation_start: date
    test_start: date
    n_origins: int
    origin_step_weeks: int
    surge_threshold: float
    surge_min_norm: float
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    app_data_dir: Path
    lgbm_params: dict


@dataclass(frozen=True)
class Categories:
    exact: dict[str, str]
    prefixes: list[tuple[str, str]]
    agencies: dict[str, str]
    targets: list[str]


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from `path`, else $NYC311_SETTINGS, else config/settings.yaml."""
    if path is None:
        path = Path(os.environ.get(SETTINGS_ENV_VAR) or DEFAULT_SETTINGS_PATH)
    raw = yaml.safe_load(Path(path).read_text())
    return Settings(
        start_date=raw["start_date"],
        horizons=list(raw["horizons"]),
        validation_start=raw["validation_start"],
        test_start=raw["test_start"],
        n_origins=int(raw["n_origins"]),
        origin_step_weeks=int(raw["origin_step_weeks"]),
        surge_threshold=float(raw["surge_threshold"]),
        surge_min_norm=float(raw["surge_min_norm"]),
        raw_dir=Path(raw["raw_dir"]),
        processed_dir=Path(raw["processed_dir"]),
        reports_dir=Path(raw["reports_dir"]),
        app_data_dir=Path(raw["app_data_dir"]),
        lgbm_params=dict(raw["lgbm_params"]),
    )


def _key(text: str) -> str:
    return text.strip().lower()


def load_categories(path: Path = DEFAULT_CATEGORIES_PATH) -> Categories:
    raw = yaml.safe_load(Path(path).read_text())["targets"]
    exact: dict[str, str] = {}
    prefixes: list[tuple[str, str]] = []
    agencies: dict[str, str] = {}
    for target, spec in raw.items():
        agencies[target] = spec["agency"]
        prefixes += [(_key(p), target) for p in spec.get("prefixes", [])]
        exact.update({_key(e): target for e in spec.get("exact", [])})
    return Categories(exact=exact, prefixes=prefixes, agencies=agencies, targets=["total", *raw])
