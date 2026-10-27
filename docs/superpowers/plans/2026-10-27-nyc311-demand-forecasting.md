# NYC 311 Service Demand Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast weekly NYC 311 demand for 71 community-district areas × 6 targets, 1–4 weeks ahead, flag surges, and serve the results on a weekly-refreshed Streamlit dashboard.

**Architecture:** A `nyc311` Python package with one module per pipeline stage (ingest → clean → panel → features/models → backtest → publish), driven by a `typer` CLI. Every forecaster implements one `Forecaster` protocol, so baselines, statsforecast and LightGBM all run through the same rolling-origin backtest. The Streamlit app reads only precomputed artifacts in `app/data/`. GitHub Actions runs CI and the weekly refresh.

**Tech Stack:** Python 3.12, uv, pandas, pyarrow, DuckDB, requests, statsforecast, LightGBM, SHAP, MLflow, holidays, PyYAML, typer, Streamlit, Plotly, GeoPandas (only for simplifying the GeoJSON at publish time), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-27-nyc311-demand-forecasting-design.md`

## Global Constraints

- Python pinned to 3.12 (`requires-python = ">=3.12,<3.13"`); the system default 3.14 is not used.
- Dataset id `erm2-nwe9`; base URL `https://data.cityofnewyork.us/resource/erm2-nwe9.csv`; page size 50000; token from env `SOCRATA_APP_TOKEN` sent as header `X-App-Token`.
- Weeks start Monday; only complete weeks; panel = 71 areas (59 regular + 12 special: 164, 226, 227, 228, 355, 356, 480, 481, 482, 483, 484, 595) × 6 targets (`total`, `noise`, `heat_hot_water`, `illegal_parking`, `sanitation`, `street_conditions`), zero-filled.
- Horizons `[1, 2, 3, 4]`; validation origins start 2024-11-04, test origins start 2025-11-03, 13 origins each, 4 weeks apart.
- Surge threshold 1.25 × norm, minimum norm 5; norm level adjustment clipped to [0.5, 2.0].
- LightGBM defaults `n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20`; tweedie power 1.2; quantiles 0.1 / 0.9; scale floor 1.0.
- No network access in unit tests. `data/`, `mlruns/` and `.env` are git-ignored; `app/data/` is committed.
- Headline metric: WAPE on `target="total"`, regular districts, h=1..4 pooled, test period.

## Review Focus

1. **API fails mid-month (429/5xx/timeout on page 3 of 5).** The expected behavior is retries, then either a complete month file or no file at all, never a partial one → test in Task 3.
2. **New or renamed complaint types appear (the 2025 taxonomy change).** They must still count toward `total`, and `panel` must report unmapped volume rather than drop it silently → test in Task 4.
3. **Series with all-zero or very short history** (special areas, a category absent in an area). Features and forecasts must be finite and ≥ 0, with no NaN or inf from dividing by zero → test in Task 9 and Task 10.
4. **Data lag at refresh time.** If the newest record is a Sunday, that week is incomplete and must be excluded → test in Task 4.
5. **ISO week 53 and year boundaries in the seasonal norm.** Week 53 uses week 52's history, and January weeks don't look up the wrong year → test in Task 6.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | deps, `nyc311` script entry, ruff + pytest config |
| `config/settings.yaml` | dates, horizons, thresholds, paths, LightGBM params |
| `config/categories.yaml` | complaint_type → target mapping + agency per target |
| `src/nyc311/config.py` | `Settings` dataclass, `load_settings`, `load_categories` |
| `src/nyc311/clean.py` | area parsing, category mapping |
| `src/nyc311/ingest.py` | Socrata client, month planning, atomic writes |
| `src/nyc311/panel.py` | DuckDB weekly rollup, zero-fill, validation |
| `src/nyc311/surge.py` | seasonal norm, surge flags |
| `src/nyc311/metrics.py` | WAPE, MASE, coverage, width, surge PRF |
| `src/nyc311/models/base.py` | `Forecaster` protocol, forecast-frame schema helper |
| `src/nyc311/models/baselines.py` | `SeasonalNaive`, `RecentAverage` |
| `src/nyc311/models/statistical.py` | `StatsForecaster` (AutoETS, MSTL) |
| `src/nyc311/features.py` | leakage-safe feature frame |
| `src/nyc311/models/lgbm.py` | `LGBMForecaster` (direct, point + P10/P90) |
| `src/nyc311/backtest.py` | origin generation, rolling backtest, MLflow logging |
| `src/nyc311/report.py` | metric breakdown tables → `reports/` |
| `src/nyc311/explain.py` | SHAP summaries |
| `src/nyc311/publish.py` | final forecasts + app artifacts |
| `src/nyc311/cli.py` | typer CLI |
| `app/Home.py`, `app/pages/*.py`, `app/lib.py` | Streamlit UI; `lib.py` holds loading/formatting logic (testable) |
| `.github/workflows/ci.yml`, `weekly_refresh.yml` | automation |
| `notebooks/01_explore.ipynb` | EDA |
| `dags/` | stretch: Airflow |

**Shared data contracts** (referenced by every task):

- **Raw month file** `data/raw/daily_YYYY-MM.parquet`: `day: date, community_board: str, complaint_type: str, n: int64`.
- **Citywide file** `data/raw/citywide_YYYY-MM.parquet`: `day: date, n: int64`.
- **Panel** (`data/processed/weekly_panel.parquet`): `week_start: date, borocd: int16, target: str, count: int32, is_special_area: bool, borough: str`.
- **Forecast frame:** `borocd, target, origin: date, horizon: int, week_start: date (target week), model: str, yhat: float, yhat_p10: float|NaN, yhat_p90: float|NaN`.

---

### Task 1: Project scaffold and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `config/settings.yaml`, `config/categories.yaml`, `src/nyc311/__init__.py`, `src/nyc311/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Settings` (frozen dataclass), with fields:
    - `start_date: date`
    - `horizons: list[int]`
    - `validation_start: date`, `test_start: date`
    - `n_origins: int`, `origin_step_weeks: int`
    - `surge_threshold: float`, `surge_min_norm: float`
    - `raw_dir: Path`, `processed_dir: Path`, `reports_dir: Path`, `app_data_dir: Path`
    - `lgbm_params: dict`
  - `load_settings(path: Path = Path("config/settings.yaml")) -> Settings`
  - `Categories` (frozen dataclass), with fields:
    - `exact: dict[str, str]`: lowercased complaint_type → target
    - `prefixes: list[tuple[str, str]]`: lowercased prefix → target
    - `agencies: dict[str, str]`: target → agency label
    - `targets: list[str]`: always starts with `"total"`
  - `load_categories(path: Path = Path("config/categories.yaml")) -> Categories`

- [ ] **Step 1: Install uv and create the environment**

Run: `brew install uv && uv python install 3.12`
Expected: `uv --version` prints a version.

- [ ] **Step 2: Write `pyproject.toml`**

Requirements:
- dependencies: `pandas pyarrow duckdb requests statsforecast lightgbm shap mlflow holidays pyyaml typer streamlit plotly geopandas`
- dev group: `pytest ruff nbformat ipykernel`
- `[project.scripts] nyc311 = "nyc311.cli:app"`
- src layout
- ruff `line-length = 100`
- pytest `testpaths = ["tests"]`

`.gitignore` covers: `data/`, `mlruns/`, `.env`, `.venv/`, `__pycache__/`, `.DS_Store`, `*.xlsx`. `.env.example` contains `SOCRATA_APP_TOKEN=`.

- [ ] **Step 3: Write the config files** with the Global Constraints values. `categories.yaml` has this shape:

```yaml
targets:
  noise:            {agency: "NYPD / DEP", prefixes: ["Noise"], exact: []}
  heat_hot_water:   {agency: "HPD", prefixes: [], exact: ["HEAT/HOT WATER"]}
  illegal_parking:  {agency: "NYPD", prefixes: [], exact: ["Illegal Parking", "Blocked Driveway"]}
  sanitation:       {agency: "DSNY", prefixes: [], exact: [<spec §9 list>]}
  street_conditions: {agency: "DOT", prefixes: [], exact: [<spec §9 list>]}
```

- [ ] **Step 4: Write failing tests**

```python
def test_load_settings_values():
    s = load_settings()
    assert s.horizons == [1, 2, 3, 4]
    assert s.validation_start == date(2024, 11, 4) and s.test_start == date(2025, 11, 3)
    assert s.surge_threshold == 1.25 and s.surge_min_norm == 5

def test_load_categories():
    c = load_categories()
    assert c.targets == ["total", "noise", "heat_hot_water", "illegal_parking", "sanitation", "street_conditions"]
    assert c.exact["heat/hot water"] == "heat_hot_water"
    assert ("noise", "noise") in c.prefixes
    assert c.agencies["sanitation"] == "DSNY"
```

- [ ] **Step 5: Run** `uv run pytest tests/test_config.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 6: Implement `config.py`.**
- [ ] **Step 7: Run** `uv sync && uv run pytest tests/test_config.py -v && uv run ruff check`. Expected: PASS, no lint errors.
- [ ] **Step 8: Commit** `git add -A && git commit -m "chore: scaffold project, config loading"`

---

### Task 2: Area parsing and category mapping

**Files:**
- Create: `src/nyc311/clean.py`
- Test: `tests/test_clean.py`

**Interfaces:**
- Consumes: `Categories` (Task 1).
- Produces:
  - `REGULAR_DISTRICTS: frozenset[int]` (59)
  - `SPECIAL_AREAS: frozenset[int]` (12)
  - `BOROUGH_CODES = {"MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3, "QUEENS": 4, "STATEN ISLAND": 5}`
  - `parse_area(community_board: str | None) -> int | None`: returns the borocd, or None if the value isn't one of the 71
  - `borough_of(borocd: int) -> str`
  - `map_category(complaint_type: str | None, cats: Categories) -> str | None`: matches exact entries first, then prefixes; case-insensitive and whitespace-trimmed

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("raw,expected", [
    ("04 BRONX", 204), ("01 MANHATTAN", 101), ("18 BROOKLYN", 318), ("03 STATEN ISLAND", 503),
    ("64 MANHATTAN", 164), ("83 QUEENS", 483), ("95 STATEN ISLAND", 595),
    ("Unspecified BRONX", None), ("0 Unspecified", None), ("13 MANHATTAN", None),
    ("19 BROOKLYN", None), (None, None), ("", None),
])
def test_parse_area(raw, expected): assert parse_area(raw) == expected

def test_district_sets(): assert len(REGULAR_DISTRICTS) == 59 and len(SPECIAL_AREAS) == 12

def test_map_category(cats):
    assert map_category("Noise - Residential", cats) == "noise"
    assert map_category("  heat/hot water ", cats) == "heat_hot_water"
    assert map_category("Blocked Driveway", cats) == "illegal_parking"
    assert map_category("Rodent", cats) is None
    assert map_category(None, cats) is None
```

- [ ] **Step 2: Run** `uv run pytest tests/test_clean.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `clean.py`.** Parse with the regex `^(\d{1,2}) (MANHATTAN|BRONX|BROOKLYN|QUEENS|STATEN ISLAND)$`, then compute the borocd and check that it is in the union of the two sets.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: area parsing and category mapping`

---

### Task 3: Socrata ingestion

**Files:**
- Create: `src/nyc311/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces:
  - `class SocrataClient(session: requests.Session | None = None, token: str | None = None, max_attempts: int = 5, base_sleep: float = 1.0)`, with:
    - `.query(params: dict) -> pd.DataFrame`: pages via `$offset` until a page returns fewer than 50000 rows
    - `.fetch_month_daily(year: int, month: int) -> pd.DataFrame`: raw month schema
    - `.fetch_month_citywide(year: int, month: int) -> pd.DataFrame`: citywide schema
  - `months_to_fetch(start: date, today: date, existing: set[str]) -> list[tuple[int, int]]`. `existing` holds `"YYYY-MM"` keys. A month is skipped only if it exists and ended more than 31 days before `today`; the current and previous months are always included.
  - `write_atomic(df: pd.DataFrame, path: Path) -> None`: writes to `path.with_suffix(".tmp")`, then calls `os.replace`
  - `ingest(settings: Settings, client: SocrataClient, today: date) -> list[Path]`

- [ ] **Step 1: Write failing tests.** Use a fake session class whose `.get` returns queued responses (no network).

```python
def test_paging_concatenates_until_short_page(): ...  # pages of 50000, 50000, 3 rows → 100003 rows, 3 calls, offsets 0/50000/100000
def test_retries_on_429_then_succeeds(): ...          # responses [429, 503, 200] → succeeds, 3 calls; base_sleep=0
def test_gives_up_after_max_attempts(): ...           # 5×503 → raises requests.HTTPError
def test_month_where_clause():                         # params contain created_date >= '2024-02-01T00:00:00' AND < '2024-03-01T00:00:00'
def test_token_header_sent(): ...                      # X-App-Token present when token given
def test_months_to_fetch():
    got = months_to_fetch(date(2020, 1, 1), date(2020, 4, 15), {"2020-01", "2020-02", "2020-03"})
    assert got == [(2020, 3), (2020, 4)]
def test_failed_write_leaves_no_file(tmp_path, monkeypatch):  # Review Focus 1
    # make DataFrame.to_parquet raise mid-write → neither target nor .tmp exists afterwards
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ingest.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `ingest.py`.**
  - Request a CSV response and parse it with `pd.read_csv(io.StringIO(text), parse_dates=["day"])`.
  - The daily query uses `$select=date_trunc_ymd(created_date) AS day, community_board, complaint_type, count(*) AS n`, `$group=day, community_board, complaint_type`, and `$order=day, community_board, complaint_type`.
  - Retry on status 429 or ≥ 500 and on `requests.ConnectionError`/`Timeout`, sleeping `base_sleep * 2**attempt`. Use a 120 s timeout.
  - `write_atomic` deletes the `.tmp` file on exception, then re-raises.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: incremental Socrata ingestion with retries and atomic writes`

---

### Task 4: Weekly panel and validation

**Files:**
- Create: `src/nyc311/panel.py`
- Test: `tests/test_panel.py`, `tests/conftest.py` (fixture writing tiny raw/citywide parquet files to `tmp_path`)

**Interfaces:**
- Consumes: `parse_area`, `map_category`, `borough_of`, `REGULAR_DISTRICTS`, `SPECIAL_AREAS`, `Categories`.
- Produces:
  - `last_complete_week(max_day: date) -> date`: returns the Monday of the latest week whose Sunday < `max_day`
  - `build_panel(raw_dir: Path, cats: Categories) -> tuple[pd.DataFrame, pd.DataFrame]`: returns `(panel, unmapped)`, where `unmapped` has columns `complaint_type, n` sorted descending
  - `validate_panel(panel: pd.DataFrame, raw_dir: Path) -> None`: raises `PanelValidationError(msg)` listing every failed check
  - `class PanelValidationError(ValueError)`

- [ ] **Step 1: Write failing tests**

```python
def test_last_complete_week():
    assert last_complete_week(date(2026, 10, 26)) == date(2026, 10, 19)  # Mon: prior week (ends Sun 25th) complete
    assert last_complete_week(date(2026, 10, 25)) == date(2026, 10, 12)  # Review Focus 4: newest is Sunday → that week excluded
def test_panel_shape_and_zero_fill(raw_fixture):  # 71 areas × 6 targets × n_weeks rows, absent combos are 0
def test_weeks_are_mondays_and_partial_week_dropped(raw_fixture)
def test_total_includes_unmapped_types(raw_fixture):  # Review Focus 2: "Brand New Type" rows add to total only, appear in `unmapped`
def test_unspecified_area_dropped_from_panel(raw_fixture)
def test_validate_passes_on_good_panel(raw_fixture)
def test_validate_flags_gap_negative_duplicate_and_reconciliation(raw_fixture)  # each mutation → PanelValidationError mentioning the check
```

- [ ] **Step 2: Run** `uv run pytest tests/test_panel.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `panel.py`.**
  - Use DuckDB to read `daily_*.parquet` (glob) and group by `(community_board, complaint_type)` distinct values.
  - Map those pairs in pandas with `parse_area`/`map_category`, a small lookup that stays under ~20k distinct pairs.
  - Join the mapping back in DuckDB. Compute `week_start = date_trunc('week', day)`, which returns Monday in DuckDB, and aggregate. Emit the `total` rows and the per-category rows.
  - Cross-join the full grid and fill with 0.
  - For reconciliation, compare the panel total plus the dropped-area counts, per week, to the citywide files.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: weekly panel with zero-fill and validation`

---

### Task 5: CLI for ingest and panel, first real data pull, EDA

**Files:**
- Create: `src/nyc311/cli.py` (commands `ingest`, `panel`), `notebooks/01_explore.ipynb`
- Modify: `config/categories.yaml` (finalize from real volumes)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ingest`, `build_panel`, `validate_panel`, `load_settings`, `load_categories`.
- Produces: `app = typer.Typer()` with:
  - `nyc311 ingest [--today YYYY-MM-DD]`
  - `nyc311 panel`: writes `data/processed/weekly_panel.parquet`, prints the top 20 unmapped types and the unmapped share of total volume

- [ ] **Step 1: Write a failing test.** Use `typer.testing.CliRunner` to invoke `panel` on the raw fixture directory (via `NYC311_SETTINGS` pointing to a tmp settings file). Assert exit code 0, that the panel file exists, and that `"unmapped"` appears in the output.
- [ ] **Step 2: Run, implement, run.** Expected: PASS.
- [ ] **Step 3: Real pull.** Ask the user to create `.env` with their token. Then run `uv run nyc311 ingest`.
  Expected: `data/raw/` holds 2 files per month from 2020-01 to the current month, and the run is idempotent (an immediate re-run fetches only 2 months).
- [ ] **Step 4: Run** `uv run nyc311 panel`. Expected: validation passes and the panel has 71 × 6 × N rows.
- [ ] **Step 5: Finalize categories.** In the notebook, list the top 60 complaint types for 2025 by volume. Add any obvious members of the 5 categories (for example, new noise or sanitation names under the 2025 "Problem" taxonomy) to `categories.yaml`, then re-run `panel`.
  Done when each of the 5 categories has ≥ 1,000 requests/week citywide on average, and the notebook records the remaining unmapped share.
- [ ] **Step 6: EDA.** The notebook should show:
  - a citywide weekly total line
  - the seasonality of each target (mean by week-of-year)
  - a small-multiples chart of 6 districts
  - a boxplot of weekly totals per borough
  - a COVID-period annotation
- [ ] **Step 7: Commit** `feat: CLI ingest/panel, finalized categories, EDA notebook`. Commit the notebook with its outputs; data stays ignored.

---

### Task 6: Seasonal norm, surge flags and metrics

**Files:**
- Create: `src/nyc311/surge.py`, `src/nyc311/metrics.py`
- Test: `tests/test_surge.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `seasonal_norm(panel: pd.DataFrame, origin: date, target_weeks: list[date]) -> pd.DataFrame`: columns `borocd, target, week_start, norm`. It uses only data ≤ `origin`, and `norm` is NaN when there is no prior-year data.
  - `flag_surge(values: pd.Series, norm: pd.Series, threshold: float = 1.25, min_norm: float = 5) -> pd.Series[bool]`. NaN norm → False.
  - `wape(y, yhat) -> float`
  - `mase(y, yhat, y_naive) -> float`: MAE(yhat)/MAE(naive), NaN if the denominator is 0
  - `coverage(y, lo, hi) -> float`
  - `mean_width(lo, hi) -> float`
  - `surge_prf(actual_flag, alert_flag) -> dict[str, float]`: keys `precision`, `recall`, `f1`; 0.0 when undefined

- [ ] **Step 1: Write failing tests**

```python
def test_wape(): assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)
def test_mase(): assert mase([10, 20], [11, 21], [8, 24]) == pytest.approx(1 / 3)
def test_coverage_and_width(): assert coverage([1, 5, 9], [0, 6, 8], [2, 7, 10]) == pytest.approx(2/3); assert mean_width([0, 6], [2, 8]) == 2
def test_surge_prf(): assert surge_prf([True, True, False, False], [True, False, True, False]) == {"precision": .5, "recall": .5, "f1": .5}
def test_flag_surge_thresholds(): # 125 vs norm 100 → True; 124 → False; norm 4 → False; NaN → False
def test_norm_constant_series(): # 3 years of constant 50 → norm 50 for any week
def test_norm_level_adjustment_clipped(): # last 13 wks 10× prior year → factor clipped to 2.0
def test_norm_uses_only_past(): # perturb values after origin → identical norm
def test_norm_week_53_and_year_boundary(): # Review Focus 5: target week ISO-53 (2026-12-28) uses week-52 history; week of 2027-01-04 uses ISO week 1 of 2025/2026
def test_norm_nan_without_prior_year()
```

- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** In the norm, ISO week 53 maps to 52. The level factor is `mean(last 13 wks ≤ origin) / mean(same 13 wks one year earlier)`, clipped to [0.5, 2.0], and set to 1.0 when the denominator is 0.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat: seasonal norm, surge flags, forecast metrics`

---

### Task 7: Forecaster protocol, baselines and backtest engine

**Files:**
- Create: `src/nyc311/models/__init__.py`, `src/nyc311/models/base.py`, `src/nyc311/models/baselines.py`, `src/nyc311/backtest.py`
- Test: `tests/test_baselines.py`, `tests/test_backtest.py`

**Interfaces:**
- Consumes: the panel contract and the `Settings` origin settings.
- Produces:
  - `class Forecaster(Protocol)`:
    - `name: str`
    - `def forecast(self, panel: pd.DataFrame, origin: date, horizons: list[int]) -> pd.DataFrame`: returns a forecast frame and must use only rows with `week_start <= origin`
  - `forecast_frame(...)`: a helper that builds the forecast-frame schema
  - `SeasonalNaive()` (name `"seasonal_naive"`), `RecentAverage()` (name `"recent_average"`). Both set p10/p90 to NaN.
  - `make_origins(start: date, n: int, step_weeks: int, last_week: date, max_h: int) -> list[date]`: drops origins where origin + max_h weeks > last_week
  - `run_backtest(models: list[Forecaster], panel: pd.DataFrame, origins: list[date], horizons: list[int], period: str, log_mlflow: bool = True) -> pd.DataFrame`. Returns the forecast frame joined with `y` (actual), `norm`, `surge_actual`, `surge_alert`, and `y_naive` (seasonal-naive yhat for the same row), plus a `period` column. Saves to `reports/predictions_{period}.parquet`.

- [ ] **Step 1: Write failing tests**

```python
def test_seasonal_naive_value(synthetic_panel):  # yhat for origin t, h=2 equals y[t+2-52]
def test_recent_average_value(synthetic_panel)   # mean of 4 weeks ending at origin, same for all h
def test_make_origins():
    o = make_origins(date(2024, 11, 4), 13, 4, last_week=date(2025, 3, 3), max_h=4)
    assert o[0] == date(2024, 11, 4) and all((b - a).days == 28 for a, b in zip(o, o[1:]))
    assert o[-1] + timedelta(weeks=4) <= date(2025, 3, 3)
def test_backtest_columns_and_no_future_access(synthetic_panel):
    # a spy Forecaster records max(panel.week_start) it received → always <= origin
def test_backtest_joins_actuals_and_surge(synthetic_panel)
```

- [ ] **Step 2: Run, implement, run.** Expected: PASS.
  - `run_backtest` slices `panel[panel.week_start <= origin]` before calling each model; that slicing is the leakage boundary.
  - It calls `seasonal_norm` once per origin.
  - MLflow logging uses a parent run named `backtest_{period}` with nested runs per model and logs WAPE/MASE. Tests pass `log_mlflow=False`.
- [ ] **Step 3: Commit** `feat: forecaster protocol, baselines, rolling-origin backtest`

---

### Task 8: Statistical models

**Files:**
- Create: `src/nyc311/models/statistical.py`
- Test: `tests/test_statistical.py`

**Interfaces:**
- Consumes: `Forecaster`, `forecast_frame`.
- Produces: `StatsForecaster(kind: Literal["ets", "mstl"])`, with names `"auto_ets"` and `"mstl"`. p10/p90 come from statsforecast `level=[80]` (`lo-80`/`hi-80`), and all values are clipped to ≥ 0.

- [ ] **Step 1: Write a failing test.** Use a synthetic 3-year series `100 + 30*sin(2πw/52) + noise` for 3 areas.
  - Both kinds return 3 × targets × 4 rows, finite and ≥ 0.
  - The MSTL forecast at a seasonal peak exceeds its trough.
  - A series of all zeros returns zeros, not an error.
- [ ] **Step 2: Run, implement, run.**
  - Reshape to statsforecast's long format (`unique_id = f"{borocd}|{target}"`, `ds`, `y`).
  - Use `StatsForecast(models=[...], freq="W-MON", n_jobs=-1)`.
  - `AutoETS(season_length=1)`; `MSTL(season_length=52, trend_forecaster=AutoETS(model="ZZN"))`.
  - Series shorter than 104 weeks fall back to AutoETS for MSTL.
  - Expected: PASS.
- [ ] **Step 3: Commit** `feat: AutoETS and MSTL forecasters`

---

### Task 9: Leakage-safe features

**Files:**
- Create: `src/nyc311/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: the panel contract and `holidays`.
- Produces:
  - `FEATURE_COLUMNS: list[str]`, exactly the spec §3 list
  - `CATEGORICAL_COLUMNS = ["borocd", "borough", "target", "is_special_area"]`
  - `build_features(panel: pd.DataFrame, origins: list[date], horizons: list[int], with_label: bool) -> pd.DataFrame`. One row per (series, origin, horizon), with columns `FEATURE_COLUMNS + ["origin", "week_start", "scale"]` plus `"label"` (= y[t+h]/scale) when `with_label`.

- [ ] **Step 1: Write failing tests**

```python
def test_leakage_guard(synthetic_panel):
    # build features at origin t; multiply all counts after t by 1000; rebuild → feature columns identical
def test_scale_floor_and_finite_on_zero_series(synthetic_panel_with_zero_series):  # Review Focus 3
    f = build_features(...); assert (f.scale >= 1.0).all() and np.isfinite(f[NUMERIC]).all().all()
def test_lag_and_same_week_last_year_values(synthetic_panel)
def test_holiday_count_thanksgiving_week():  # week of 2025-11-24 → n_holidays == 1
def test_covid_flag_bounds()                 # 2020-03-02 True, 2021-07-05 False
```

- [ ] **Step 2: Run, implement, run.** Expected: PASS. Pivot the panel to wide form (weeks × series) once, compute rolling statistics on the wide frame, then index by origin. Origins earlier than 52 weeks after the panel start are skipped.
- [ ] **Step 3: Commit** `feat: leakage-safe feature builder`

---

### Task 10: Global LightGBM forecaster

**Files:**
- Create: `src/nyc311/models/lgbm.py`
- Test: `tests/test_lgbm.py`

**Interfaces:**
- Consumes: `build_features`, `FEATURE_COLUMNS`, `CATEGORICAL_COLUMNS`, `Forecaster`.
- Produces:
  - `LGBMForecaster(params: dict, train_origin_step: int = 1)`, name `"lgbm"`. `forecast()` builds training rows for every origin from panel start + 52 weeks up to `origin − h` (so labels exist), fits 3 models per horizon, predicts at `origin`, multiplies by `scale`, clips to ≥ 0, and sorts (p10, yhat, p90) per row.
  - `.fit(panel, last_origin) -> None`
  - `.predict(panel, origin) -> pd.DataFrame`
  - `.models_: dict[tuple[int, str], lgb.LGBMRegressor]`, keyed by `(h, "point"|"p10"|"p90")`
  - `.train_frame_: pd.DataFrame`, the training features, kept for SHAP

- [ ] **Step 1: Write failing tests** on the synthetic seasonal panel, with `n_estimators=50` for speed:
  - output schema and finiteness
  - `p10 <= yhat <= p90` on every row
  - an all-zero series gives yhat == 0 within 1e-6 after clipping (Review Focus 3)
  - beats `SeasonalNaive` on WAPE over 4 synthetic origins, using a series with a trend so last-year values are biased low
- [ ] **Step 2: Run, implement, run.** Expected: PASS. Categoricals are cast to `category` dtype, with the category set fixed from the full panel so train and predict share the same codes.
- [ ] **Step 3: Commit** `feat: global direct LightGBM with quantile intervals`

---

### Task 11: Validation backtest, tuning, reports, and the test evaluation

**Files:**
- Create: `src/nyc311/report.py`
- Modify: `src/nyc311/cli.py` (add `backtest --period validation`, `evaluate-test`)
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `run_backtest`, all forecasters, metrics.
- Produces:
  - `summarize(preds: pd.DataFrame) -> dict[str, pd.DataFrame]`, with keys `overall`, `by_horizon`, `by_target`, `by_borough`, `by_area_type`, `surge`. Each table has rows per model and columns `wape, mase, coverage, width` (surge: `precision, recall, f1` for `lgbm` and `rule_baseline`). `overall` is the headline subset (total, regular, all h).
  - `write_reports(tables, period, reports_dir) -> None`: writes `reports/{period}_{key}.csv` and `.md`, plus `reports/figures/{period}_*.png` (WAPE by horizon per model, WAPE by borough for lgbm, coverage by target)

- [ ] **Step 1: Write failing tests.** Run `summarize` on a hand-built predictions frame and check the WAPE values against hand-calculated numbers, the headline filtering (special areas and non-total rows excluded from `overall`), and that the rule baseline uses `recent_average` vs. `norm`.
- [ ] **Step 2: Run, implement, run.** Expected: PASS.
- [ ] **Step 3: Validation run.**
  Run: `uv run nyc311 backtest --period validation`
  Expected: `reports/validation_overall.md` lists 5 models. LightGBM WAPE should be lower than `seasonal_naive`. If it isn't, investigate the features before tuning.
- [ ] **Step 4: Tuning.** `nyc311 backtest --period validation --grid` runs the 9-combination grid from spec §3 for LGBM only. Write the best params into `config/settings.yaml` and log all combinations to MLflow.
- [ ] **Step 5: Test evaluation, run exactly once.**
  Run: `uv run nyc311 evaluate-test`. It refuses to run if `reports/test_overall.csv` exists, unless `--force`.
  Expected: `reports/test_*` files are written and the MLflow run is tagged `final=true`. Record the success criteria (spec §1) as pass or fail in `reports/test_summary.md`. Report honestly either way.
- [ ] **Step 6: Commit** `feat: evaluation reports; validation tuning; final test results`. Include `reports/` and the settings.

---

### Task 12: SHAP explanations

**Files:**
- Create: `src/nyc311/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `LGBMForecaster.models_[(1, "point")]`, `.train_frame_`.
- Produces: `shap_summary(model, train_frame, sample: int = 20000, seed: int = 0) -> pd.DataFrame`, with columns `level` ("global"|"target"|"area_target"), `borocd` (nullable), `target` (nullable), `feature`, `mean_abs_shap`, `rank`, keeping the top 5 per group.

- [ ] **Step 1: Write a failing test** on a small fitted model:
  - the levels are present
  - ranks run 1..5
  - `mean_abs_shap >= 0`
  - on synthetic data where only `lag_1` carries signal, `lag_1` is the global rank 1
- [ ] **Step 2: Run, implement, run.** Expected: PASS.
- [ ] **Step 3: Commit** `feat: SHAP driver summaries`

---

### Task 13: Final training, forecast, publish, refresh

**Files:**
- Create: `src/nyc311/publish.py`, `app/data/.gitkeep`
- Modify: `src/nyc311/cli.py` (add `train`, `forecast`, `publish`, `refresh`)
- Test: `tests/test_publish.py`

**Interfaces:**
- Consumes: `LGBMForecaster`, `seasonal_norm`, `flag_surge`, `shap_summary`, the test reports, `Categories.agencies`.
- Produces:
  - `train` saves the fitted models to `data/models/lgbm.pkl` via `joblib`.
  - `forecast` writes `data/processed/forecast_latest.parquet`. This is the forecast frame plus `norm`, `pct_vs_norm = yhat/norm − 1` (NaN when norm is NaN), `surge_alert`, and `agency`.
  - `publish(settings, cats) -> None` writes the six files listed in spec §7 to `app/data/`:
    - `history` holds the last 104 weeks
    - `metrics` concatenates the test tables with a `table` column
    - `districts.geojson` is downloaded once from NYC Open Data (Community Districts GeoJSON), cached in `data/raw/`, and simplified with GeoPandas `simplify(0.0005)`, keeping only the `BoroCD` property
    - `meta.json` has the shape `{"data_through": "YYYY-MM-DD", "generated_at": ISO8601}`
  - `refresh` runs `ingest → panel → train → forecast → publish` and exits non-zero on any failure.

- [ ] **Step 1: Write failing tests** on the synthetic pipeline in `tmp_path`, with GeoJSON from a 2-polygon fixture:
  - all six files are produced
  - `districts.geojson` < 1 MB and has 71 features (use a 71-feature synthetic fixture)
  - `forecasts.parquet` has 71 × 6 × 4 rows
  - `pct_vs_norm` is NaN wherever `norm` is NaN
  - `meta.json` parses
  - a smoke test where `refresh` fails at `panel` (monkeypatched to raise) leaves existing `app/data/` files unchanged
- [ ] **Step 2: Run, implement, run.** Expected: PASS. `publish` writes to `app/data.tmp/` and swaps directories at the end.
- [ ] **Step 3: Real run** `uv run nyc311 refresh`. Expected: `app/data/` is populated and under 10 MB total.
- [ ] **Step 4: Commit** `feat: final training, forecast and publish pipeline` (including `app/data/`)

---

### Task 14: Streamlit dashboard

**Files:**
- Create: `app/lib.py`, `app/Home.py`, `app/pages/1_District_Detail.py`, `app/pages/2_Model_Performance.py`, `app/pages/3_About.py`, `app/requirements.txt` (for Streamlit Cloud: pinned `streamlit pandas pyarrow plotly`)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: the `app/data/` files only (no `nyc311` imports in the app).
- Produces:
  - in `app/lib.py`:
    - `load_data(data_dir: Path) -> AppData`, a dataclass of the frames, geojson and meta, wrapped in `st.cache_data`
    - `surge_watch(forecasts, target: str, horizon: int) -> pd.DataFrame`: alerts only, sorted by `pct_vs_norm` descending, with columns `area, borough, target, week_start, yhat, norm, pct_vs_norm, agency`
    - `area_label(borocd) -> str`, e.g. `"Bronx 04"` or `"Central Park (special area)"`
    - `TARGET_COLORS: dict[str, str]`
    - `fmt_pct(x) -> str`: NaN → `"n/a"`

- [ ] **Step 1: Write failing tests**
  - unit tests for `surge_watch` (filtering and order) and for `fmt_pct(nan) == "n/a"`
  - `streamlit.testing.v1.AppTest` smoke tests: each page runs against a fixture `app/data` with no exception
  - an area with NaN norm renders "n/a" on Home and in District Detail
- [ ] **Step 2: Run, implement, run.** Expected: PASS.
  - **Home:**
    - `px.choropleth_mapbox` (or `choropleth_map`) on `BoroCD`, colored by `pct_vs_norm` with a diverging scale centered at 0 (clipped to ±50%)
    - selectbox for target, slider for horizon 1–4
    - the Surge watch table
    - `st.query_params` carries the selected area to the detail page
  - **Detail:**
    - Plotly fan chart: actual line, forecast line, P10–P90 band, norm dashed line
    - the top-5 SHAP bar chart
    - a 2×3 small-multiples grid of all targets
  - **Every page:**
    - a header caption `Data through week of {data_through}`
    - colors from `TARGET_COLORS`
    - charts use `template="plotly_white"` plus `st.plotly_chart(theme="streamlit")` so they work in light and dark mode
- [ ] **Step 3: Local check** `uv run streamlit run app/Home.py`. Click through all 4 pages, pick Bronx 04 → Heat/Hot Water, and confirm the fan chart shows a winter peak.
- [ ] **Step 4: Commit** `feat: Streamlit dashboard`

---

### Task 15: CI, weekly refresh and deployment

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/weekly_refresh.yml`

**Interfaces:**
- Consumes: `nyc311 refresh` and the pytest suite.

- [ ] **Step 1: `ci.yml`.** Runs on push/PR: `astral-sh/setup-uv`, `uv sync`, `uv run ruff check`, `uv run pytest -q`.
- [ ] **Step 2: `weekly_refresh.yml`.** Runs on cron `0 10 * * 1` plus `workflow_dispatch`, with `permissions: contents: write` and the env var `SOCRATA_APP_TOKEN: ${{ secrets.SOCRATA_APP_TOKEN }}`.
  - Cache `data/raw` with `actions/cache` keyed by week, so only recent months are fetched.
  - Run `uv run nyc311 refresh`.
  - Then commit, only if `git diff --quiet app/data` reports changes: `git add app/data && git commit -m "data: weekly refresh $(date +%F)" && git push`.
  - Because of the cache, a cold run builds the history once. Run the full test-period evaluation locally, not in CI.
- [ ] **Step 3: Push to GitHub.** Ask the user before creating the remote repo: `gh repo create nyc311-demand-forecasting --public --source . --push`. Then add the secret `gh secret set SOCRATA_APP_TOKEN`.
- [ ] **Step 4: Verify.** CI is green on the push. Trigger `gh workflow run weekly_refresh.yml`; it should succeed and either commit or report "no changes".
- [ ] **Step 5: Deploy.** The user connects the repo at share.streamlit.io with main file `app/Home.py`. Confirm the public URL loads all 4 pages.
- [ ] **Step 6: Commit** `ci: tests, lint and weekly data refresh`

---

### Task 16: README

**Files:**
- Create: `README.md`, `reports/figures/*.png` (already generated)

- [ ] **Step 1: Write the README** in this order:
  1. a one-line pitch, plus the live-app link and the CI badge
  2. the problem and why it matters (3 sentences)
  3. results, using the headline table from `reports/test_overall.md` **copied from the file**, the WAPE-by-horizon figure, and coverage and surge-alert precision/recall
  4. a dashboard screenshot
  5. the approach (pipeline diagram in Mermaid; models; backtest design)
  6. the equity check, using WAPE by borough with an honest commentary
  7. what drives demand (SHAP global figure)
  8. limitations and future work (weather covariates, deep learning, Airflow)
  9. how to reproduce (`uv sync`, `.env`, `nyc311 refresh`, `streamlit run`)
- [ ] **Step 2: Verify** that every number in the README matches a file in `reports/`. Grep each number.
- [ ] **Step 3: Commit** `docs: README`

---

### Task 17 (stretch): Airflow local demo

**Files:**
- Create: `dags/nyc311_weekly.py`, `dags/docker-compose.yml`, `dags/README.md`

- [ ] **Step 1:** Write a DAG `nyc311_weekly` with schedule `0 10 * * 1`, `catchup=False`, and five `BashOperator`s (`nyc311 ingest` → `panel` → `train` → `forecast` → `publish`), each with `retries=2`.
- [ ] **Step 2:** Base the Compose file on the official `apache/airflow` standalone image, with the repo mounted and `pip install -e .` at startup.
- [ ] **Step 3: Verify** with `docker compose -f dags/docker-compose.yml up`. Trigger the DAG in the UI at localhost:8080; all 5 tasks should succeed.
- [ ] **Step 4: Commit** `feat: Airflow DAG local demo`
