"""Incremental NYC 311 ingestion from the Socrata API into monthly parquet files."""

import io
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from nyc311.config import Settings

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
PAGE_SIZE = 50000
TIMEOUT_S = 120
TOKEN_ENV_VAR = "SOCRATA_APP_TOKEN"
DAILY_COLUMNS = ["day", "community_board", "complaint_type", "n"]
CITYWIDE_COLUMNS = ["day", "n"]


class SocrataClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        token: str | None = None,
        max_attempts: int = 5,
        base_sleep: float = 5.0,
    ):
        self.session = session or requests.Session()
        self.token = token if token is not None else os.environ.get(TOKEN_ENV_VAR)
        self.max_attempts = max_attempts
        self.base_sleep = base_sleep

    def _get_text(self, params: dict) -> str:
        headers = {"X-App-Token": self.token} if self.token else {}
        for attempt in range(self.max_attempts):
            last = attempt == self.max_attempts - 1
            try:
                resp = self.session.get(BASE_URL, params=params, headers=headers, timeout=TIMEOUT_S)
            except (requests.ConnectionError, requests.Timeout):
                if last:
                    raise
            else:
                if resp.status_code != 429 and resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.text
                if last:
                    resp.raise_for_status()
            time.sleep(self.base_sleep * 2**attempt)
        raise RuntimeError("unreachable")  # pragma: no cover

    def query(self, params: dict) -> pd.DataFrame:
        pages = []
        offset = 0
        while True:
            text = self._get_text({**params, "$limit": PAGE_SIZE, "$offset": offset})
            page = pd.read_csv(
                io.StringIO(text),
                parse_dates=["day"] if "day" in text.split("\n", 1)[0].split(",") else False,
                dtype={"community_board": str, "complaint_type": str},
            )
            pages.append(page)
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        return pd.concat(pages, ignore_index=True)

    @staticmethod
    def _month_where(year: int, month: int) -> str:
        ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
        return (
            f"created_date >= '{year:04d}-{month:02d}-01T00:00:00' "
            f"AND created_date < '{ny:04d}-{nm:02d}-01T00:00:00'"
        )

    @staticmethod
    def _to_date(df: pd.DataFrame) -> pd.DataFrame:
        df["day"] = pd.to_datetime(df["day"]).dt.date
        return df

    def fetch_month_daily(self, year: int, month: int) -> pd.DataFrame:
        df = self.query(
            {
                "$select": "date_trunc_ymd(created_date) AS day, community_board, "
                "complaint_type, count(*) AS n",
                "$where": self._month_where(year, month),
                "$group": "day, community_board, complaint_type",
                "$order": "day, community_board, complaint_type",
            }
        )
        return self._to_date(df)[DAILY_COLUMNS]

    def fetch_month_citywide(self, year: int, month: int) -> pd.DataFrame:
        df = self.query(
            {
                "$select": "date_trunc_ymd(created_date) AS day, count(*) AS n",
                "$where": self._month_where(year, month),
                "$group": "day",
                "$order": "day",
            }
        )
        return self._to_date(df)[CITYWIDE_COLUMNS]


def months_to_fetch(start: date, today: date, existing: set[str]) -> list[tuple[int, int]]:
    """Months from `start` to `today`; skip existing ones that ended over 31 days ago."""
    out = []
    y, m = start.year, start.month
    while (y, m) <= (today.year, today.month):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        ended_days_ago = (today - date(ny, nm, 1)).days
        if not (f"{y:04d}-{m:02d}" in existing and ended_days_ago > 31):
            out.append((y, m))
        y, m = ny, nm
    return out


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def ingest(settings: Settings, client: SocrataClient, today: date) -> list[Path]:
    raw_dir = Path(settings.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        p.stem.removeprefix("daily_")
        for p in raw_dir.glob("daily_*.parquet")
        if (raw_dir / f"citywide_{p.stem.removeprefix('daily_')}.parquet").exists()
    }
    written = []
    for year, month in months_to_fetch(settings.start_date, today, existing):
        key = f"{year:04d}-{month:02d}"
        for name, df in (
            ("daily", client.fetch_month_daily(year, month)),
            ("citywide", client.fetch_month_citywide(year, month)),
        ):
            path = raw_dir / f"{name}_{key}.parquet"
            write_atomic(df, path)
            written.append(path)
    return written
