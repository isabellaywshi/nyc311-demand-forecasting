"""Command-line entry point: `nyc311 ingest` and `nyc311 panel`."""

import os
from datetime import date
from pathlib import Path

import duckdb
import typer

from nyc311.config import load_categories, load_settings
from nyc311.ingest import SocrataClient, ingest
from nyc311.panel import build_panel, validate_panel

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
