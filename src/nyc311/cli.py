"""Command-line entry point: `nyc311 ingest` and `nyc311 panel`."""

import itertools
import os
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import typer

from nyc311.backtest import make_origins, run_backtest
from nyc311.config import DEFAULT_SETTINGS_PATH, Settings, load_categories, load_settings
from nyc311.ingest import SocrataClient, ingest
from nyc311.models.baselines import RecentAverage, SeasonalNaive
from nyc311.models.lgbm import LGBMForecaster
from nyc311.models.statistical import StatsForecaster
from nyc311.panel import build_panel, validate_panel
from nyc311.report import enrich, summarize, write_reports, write_test_summary

app = typer.Typer(help="NYC 311 weekly demand forecasting pipeline.", no_args_is_help=True)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip().removeprefix("export "), value.strip().strip("'\""))


@app.callback()
def _main() -> None:
    load_dotenv()


@app.command("ingest")
def ingest_cmd(
    today: str = typer.Option(None, "--today", help="Override today's date (YYYY-MM-DD)."),
) -> None:
    """Download monthly aggregates from Socrata into the raw directory (incremental)."""
    settings = load_settings()
    day = date.fromisoformat(today) if today else date.today()
    written = ingest(settings, SocrataClient(), day)
    typer.echo(f"Fetched {len(written) // 2} month(s); wrote {len(written)} files.")


@app.command("panel")
def panel_cmd() -> None:
    """Build and validate the weekly panel; report unmapped complaint types."""
    settings = load_settings()
    panel, unmapped = build_panel(settings.raw_dir, load_categories())
    validate_panel(panel, settings.raw_dir)
    out = Path(settings.processed_dir) / "weekly_panel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    glob = str(Path(settings.raw_dir) / "daily_*.parquet")
    total = duckdb.sql(f"SELECT sum(n) FROM read_parquet('{glob}')").fetchall()[0][0] or 0
    share = float(unmapped["n"].sum()) / total if total else 0.0
    typer.echo(f"Wrote {out} ({len(panel):,} rows); validation passed.")
    typer.echo(f"Top 20 unmapped complaint types (unmapped share of volume: {share:.2%}):")
    typer.echo(unmapped.head(20).to_string(index=False))


GRID = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1]}


def _load_panel(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(Path(settings.processed_dir) / "weekly_panel.parquet").assign(
        week_start=lambda d: pd.to_datetime(d["week_start"]).dt.date
    )


def _origins(settings: Settings, panel: pd.DataFrame, start: date) -> list[date]:
    return make_origins(
        start, settings.n_origins, settings.origin_step_weeks,
        panel["week_start"].max(), max(settings.horizons),
    )  # fmt: skip


def _models(settings: Settings, lgbm_params: dict) -> list:
    return [
        SeasonalNaive(),
        RecentAverage(),
        StatsForecaster("ets"),
        StatsForecaster("mstl"),
        LGBMForecaster(lgbm_params),
    ]


def _update_settings_params(path: Path, params: dict) -> None:
    """Rewrite values under lgbm_params in place, keeping the rest of the file untouched."""
    text = path.read_text()
    for key, value in params.items():
        text, n = re.subn(rf"(?m)^(\s+{key}:\s*)\S+", rf"\g<1>{value}", text)
        if n != 1:
            raise ValueError(f"lgbm_params.{key} not found exactly once in {path}")
    path.write_text(text)


def _evaluate(settings, panel, origins, period, models, tags=None) -> dict[str, pd.DataFrame]:
    preds = run_backtest(
        models, panel, origins, settings.horizons, period, log_mlflow=True, tags=tags,
        surge_threshold=settings.surge_threshold, surge_min_norm=settings.surge_min_norm,
    )  # fmt: skip
    preds = enrich(preds, panel)
    reports = Path(settings.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(reports / f"predictions_{period}.parquet", index=False)
    tables = summarize(preds, settings.surge_threshold, settings.surge_min_norm)
    write_reports(tables, period, reports)
    return tables


@app.command("backtest")
def backtest_cmd(
    period: str = typer.Option("validation", "--period", help="Only 'validation' is allowed."),
    grid: bool = typer.Option(False, "--grid", help="Run the LGBM tuning grid (spec section 3)."),
) -> None:
    """Rolling-origin backtest on the validation period; --grid tunes LightGBM."""
    if period != "validation":
        raise typer.BadParameter("only --period validation; use `nyc311 evaluate-test` for test.")
    settings = load_settings()
    panel = _load_panel(settings)
    origins = _origins(settings, panel, settings.validation_start)
    if not grid:
        tables = _evaluate(
            settings, panel, origins, period, _models(settings, settings.lgbm_params)
        )
        typer.echo(tables["overall"].to_string(index=False))
        return
    results = []
    for nl, lr in itertools.product(GRID["num_leaves"], GRID["learning_rate"]):
        params = {**settings.lgbm_params, "num_leaves": nl, "learning_rate": lr}
        preds = run_backtest(
            [LGBMForecaster(params)], panel, origins, settings.horizons, "validation",
            log_mlflow=True, tags={"grid": "true"}, params=params,
            surge_threshold=settings.surge_threshold, surge_min_norm=settings.surge_min_norm,
        )  # fmt: skip
        o = summarize(enrich(preds, panel))["overall"].iloc[0]
        results.append({"num_leaves": nl, "learning_rate": lr, **o[["wape", "mase", "coverage"]]})
        typer.echo(f"num_leaves={nl} lr={lr}: wape={o['wape']:.4f} mase={o['mase']:.4f}")
    res = pd.DataFrame(results).sort_values("wape").reset_index(drop=True)
    Path(settings.reports_dir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(settings.reports_dir) / "validation_grid.csv", index=False)
    best = res.iloc[0]
    chosen = {"num_leaves": int(best["num_leaves"]), "learning_rate": float(best["learning_rate"])}
    _update_settings_params(
        Path(os.environ.get("NYC311_SETTINGS") or DEFAULT_SETTINGS_PATH), chosen
    )
    typer.echo(f"Best (lowest validation WAPE): {chosen}; settings updated.")


@app.command("evaluate-test")
def evaluate_test_cmd(
    force: bool = typer.Option(False, "--force", help="Re-run even if test results exist."),
) -> None:
    """Final test-period evaluation with tuned params. Run once."""
    settings = load_settings()
    if (Path(settings.reports_dir) / "test_overall.csv").exists() and not force:
        typer.echo("reports/test_overall.csv exists; the test period runs once. Use --force.")
        raise typer.Exit(1)
    panel = _load_panel(settings)
    origins = _origins(settings, panel, settings.test_start)
    typer.echo(f"Test origins: {len(origins)} ({origins[0]}..{origins[-1]})")
    tables = _evaluate(
        settings, panel, origins, "test", _models(settings, settings.lgbm_params),
        tags={"final": "true"},
    )  # fmt: skip
    write_test_summary(tables, Path(settings.reports_dir))
    typer.echo(tables["overall"].to_string(index=False))
