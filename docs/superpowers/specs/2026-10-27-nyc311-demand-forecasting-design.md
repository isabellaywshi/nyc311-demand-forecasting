# NYC 311 Service Demand Forecasting — Design Spec

**Date:** 2026-10-27
**Status:** Approved — implemented on feat/v1
**Owner:** Isabella Shi

## 1. Purpose

Portfolio / resume project answering: *Can city agencies predict where and when 311 service requests will surge, so they can allocate resources more effectively?*

**Success criteria**
- The global LightGBM model beats both baselines (seasonal naive, recent average) on test-period WAPE and MASE for Total demand across the 59 regular community districts, horizons 1–4.
- P10–P90 interval coverage on the test period is within 70–90%.
- A Streamlit dashboard is deployed publicly and refreshed weekly by automation.
- The README is skimmable in ~2 minutes, and every number in it comes from generated report files.

**Non-goals (v1):** deep learning models, weather covariates, Spark, Docker, a paid or hosted orchestrator.

## 2. Data

- **Source:** NYC Open Data, *311 Service Requests from 2020 to Present*, dataset id `erm2-nwe9`, Socrata SoQL API `https://data.cityofnewyork.us/resource/erm2-nwe9.csv`. About 22.9M rows as of 2026-10-26, starting 2020-01-01.
- **Pull:** server-side aggregation of `date_trunc_ymd(created_date) AS day, community_board, complaint_type, count(*) AS n`, grouped by those three columns, one calendar month per request series, paged at `$limit=50000` with `$order=day, community_board, complaint_type`.
- **Auth:** the `SOCRATA_APP_TOKEN` environment variable, sent as the `X-App-Token` header. It is optional locally but required in CI.
- **Boundaries:** NYC DCP Community Districts GeoJSON (71 polygons, property `BoroCD`), simplified to under 1 MB.

### 2.1 Areas

- `community_board` values look like `"04 BRONX"`. The borough codes are MANHATTAN=1, BRONX=2, BROOKLYN=3, QUEENS=4, STATEN ISLAND=5, and `borocd = boro_code*100 + int(number)`.
- **Regular districts (59):** Manhattan 101–112, Bronx 201–212, Brooklyn 301–318, Queens 401–414, Staten Island 501–503.
- **Special areas (12, `is_special_area=True`):** 164, 226, 227, 228, 355, 356, 480, 481, 482, 483, 484, 595.
- Any other value (`"Unspecified …"`, `"0 Unspecified"`, unparseable) is dropped from the area panel. It still counts toward the citywide reconciliation total.

### 2.2 Targets

There are six targets: `total` plus five categories, mapped in `config/categories.yaml` (complaint type → category, exact match, case-insensitive, after trimming whitespace; a `prefixes` list also applies). Unmapped types count only toward `total`. The category set (finalized from 2025 volumes) is listed in `config/categories.yaml` and repeated in §9.

### 2.3 Weekly panel

- Weeks start on Monday (`week_start`). Only complete weeks are kept: the last kept week ends on or before the most recent full Sunday that is strictly before the newest record's date.
- The panel is a complete grid of 71 areas × 6 targets × every week from the first Monday on or after 2020-01-01 to the last complete week. Missing combinations are filled with 0.
- Schema: `week_start: date, borocd: int16, target: str, count: int32, is_special_area: bool, borough: str`.
- **Validation (pipeline fails on violation):**
  - exactly 71 areas, 59 of them regular
  - consecutive weeks with no gaps
  - `count >= 0`
  - no duplicate keys
  - for each week, the panel's `total` summed over areas plus that week's dropped (unlocated or invalid area) count matches an independently pulled citywide total (`date_trunc_ymd(created_date) AS day, count(*) AS n`, saved as `data/raw/citywide_YYYY-MM.parquet`), within 1%

## 3. Models

- **Series:** 71 × 6 = 426. Horizons h ∈ {1, 2, 3, 4} weeks. Origin = the last observed week; the forecast covers weeks origin+1 … origin+4.
- **Baselines:**
  - `seasonal_naive`: y[t+h−52]
  - `recent_average`: mean(y[t−3..t])
- **Statistical** (`statsforecast`, fit per series at each origin):
  - `AutoETS(season_length=1)`
  - `MSTL(season_length=52, trend_forecaster=AutoETS(model="ZZN"))`
- **Global LightGBM:**
  - **Direct strategy:** one model per horizon.
  - **Scale:** each row is divided by `scale = mean(y[t−51..t])` (computed as of the origin, floored at 1.0). The label is `y[t+h]/scale`, and predictions are multiplied by `scale`.
  - **Features** (all computed as of origin t):
    - `lag_1..lag_4` (scaled)
    - `roll_mean_4`, `roll_mean_13`, `roll_mean_52`, `roll_std_13` (scaled)
    - `same_week_last_year = y[t+h−52]/scale`
    - `woy_sin`, `woy_cos` of the target week
    - `month` of the target week
    - `n_holidays` in the target week (`holidays` package, country US, subdivision NY)
    - `covid` (target week between 2020-03-01 and 2021-06-30)
    - `borocd`, `borough`, `target`, `is_special_area` (categorical)
    - `horizon`
  - **Models per horizon:** point (`objective="tweedie"`, `tweedie_variance_power=1.2`), P10 (`objective="quantile", alpha=0.1`), P90 (`alpha=0.9`).
  - **Post-processing:** predictions are clipped to ≥ 0. If P10 > point or point > P90, the three are sorted.
  - **Defaults:** `n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20`. A small grid tuned on validation only covers `num_leaves ∈ {15, 31, 63}` × `learning_rate ∈ {0.03, 0.05, 0.1}`.
- **Training rows:** every (series, origin t) with t ≥ 52 weeks after panel start, so that every lag exists.

## 4. Surge definition

- **Seasonal norm** for a series in target week w is the mean of the counts in the same ISO week-of-year over all prior years that have data, multiplied by a level adjustment. The adjustment is the ratio of the last 13 weeks' mean to the same 13 weeks' mean one year earlier, clipped to [0.5, 2.0]. If no prior year exists, the norm is undefined and no surge is flagged.
- **Surge:** `actual >= 1.25 × norm` and `norm >= 5`. The minimum norm keeps tiny series from flagging on noise.
- **Surge alert:** `forecast_point >= 1.25 × norm` and `norm >= 5`.
- **Rule baseline alert:** `recent_average >= 1.25 × norm`.

## 5. Evaluation

| Period | Origins |
|---|---|
| Training only | panel start → 2024-10-28 |
| Validation | 13 origins, every 4 weeks starting 2024-11-04 |
| Test | 13 origins, every 4 weeks starting 2025-11-03 (only those whose t+4 ≤ last complete week) |

- At each origin, every model trains on data ≤ origin and forecasts h=1..4.
- **Metrics:**
  - WAPE = Σ|y−ŷ| / Σy
  - MASE = MAE / MAE of `seasonal_naive` over the same rows
  - coverage = share of y within [P10, P90]
  - mean interval width
  - surge precision, recall and F1 (ML alert vs. the rule baseline)
- **Breakdowns:** overall, by horizon, by target, by borough, regular vs. special.
- **Headline:** `target="total"`, regular districts, h=1..4 pooled.
- The test period runs only via `nyc311 evaluate-test`, and its MLflow run is tagged `final=true`.
- **Leakage guard:** a test asserts that features for origin t are unchanged when all panel values after t are perturbed.

## 6. Explainability and tracking

- SHAP `TreeExplainer` on the h=1 point model, using up to 20,000 training rows sampled at the final fit.
- **Outputs:** mean |SHAP| per feature globally, per target, and per (borocd, target), keeping the top 5 features.
- MLflow tracks to the local `mlruns/` folder (git-ignored), with one run per (model, period) under a parent run per backtest.

## 7. Dashboard

- Streamlit multipage app deployed on Streamlit Community Cloud. It reads only `app/data/`:
  - `forecasts.parquet`
  - `history.parquet` (last 104 weeks)
  - `metrics.parquet`
  - `shap_summary.parquet`
  - `districts.geojson`
  - `meta.json` (`data_through`, `generated_at`)
- **Pages:**
  - **Home** (map colored by % vs. norm, category and horizon selectors, Surge watch table with responsible agency)
  - **District detail** (fan chart, seasonal norm line, top-5 SHAP drivers, small multiples)
  - **Model performance**
  - **About**
- Every page header shows "Data through week of {data_through}".
- Plotly charts. A diverging palette for % vs. norm, and one fixed color per target across all pages.

## 8. Pipeline, automation, engineering

- **CLI** (`typer`, entry point `nyc311`): `ingest`, `panel`, `backtest --period validation`, `evaluate-test`, `train`, `forecast`, `publish`, plus `refresh` (ingest → panel → train → forecast → publish).
- **Ingest** is incremental. Months already completed (a file exists in `data/raw/` and the month ended more than 31 days before today) are skipped. The current and previous months are always re-pulled. Files are written to a temporary path and then atomically renamed. Requests retry on 429 or 5xx with exponential backoff, at most 5 attempts.
- **GitHub Actions:**
  - `ci.yml` runs `ruff check` and `pytest` on push and PR.
  - `weekly_refresh.yml` runs on cron `0 18 * * 2` (Tuesdays 18:00 UTC; the dataset loads with ~1-day lag, so by Tuesday afternoon Monday's records exist and the prior Mon-Sun week counts as complete) plus `workflow_dispatch`. It runs `nyc311 refresh` and commits `app/data/` only if the step succeeds.
- **Stretch:** an Airflow DAG in `dags/nyc311_weekly.py` with `BashOperator`s calling the same CLI commands, plus `dags/docker-compose.yml`.
- **Environment:** `uv`, Python 3.12. `.env` is git-ignored and `.env.example` is committed. `data/` and `mlruns/` are git-ignored.

## 9. Category mapping (initial)

See `config/categories.yaml`. Categories and their responsible agencies:

| target | agency | members |
|---|---|---|
| `noise` | NYPD / DEP | all complaint types with prefix `Noise` |
| `heat_hot_water` | HPD | `HEAT/HOT WATER` |
| `illegal_parking` | NYPD | `Illegal Parking`, `Blocked Driveway`, `Abandoned Vehicle` |
| `sanitation` | DSNY | `Dirty Condition`, `Missed Collection`, `Illegal Dumping`, `Residential Disposal Complaint`, `Commercial Disposal Complaint`, `Litter Basket Request`, `Overflowing Litter Baskets`, `Graffiti`, `Sanitation Condition`, `Derelict Vehicles` |
| `street_conditions` | DOT | `Street Condition`, `Street Light Condition`, `Traffic Signal Condition`, `Sidewalk Condition`, `Curb Condition`, `Broken Parking Meter` |

## 10. Risks

- **API outages** (one was observed during design with a `service-unavailable` response): retries, plus incremental re-runs.
- **Complaint taxonomy drift** (the 2025 rename of "Complaint Type" to "Problem"): the mapping lives in config, and an unmapped-volume report is printed by `panel`.
- **Python 3.14** is the system default, but it is pinned to 3.12 for LightGBM and statsforecast wheels.
