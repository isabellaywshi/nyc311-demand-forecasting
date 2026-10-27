from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from nyc311.config import load_settings
from nyc311.ingest import SocrataClient, ingest, months_to_fetch, write_atomic


class FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": headers})
        return self.responses.pop(0)


def daily_csv(n_rows, day="2024-02-01T00:00:00.000"):
    lines = ["day,community_board,complaint_type,n"]
    lines += [f"{day},01 BRONX,Noise,{i + 1}" for i in range(n_rows)]
    return "\n".join(lines) + "\n"


def make_client(responses, **kw):
    session = FakeSession(responses)
    return SocrataClient(session=session, base_sleep=0, **kw), session


def test_paging_concatenates_until_short_page():
    client, session = make_client(
        [
            FakeResponse(200, daily_csv(50000)),
            FakeResponse(200, daily_csv(50000)),
            FakeResponse(200, daily_csv(3)),
        ]
    )
    df = client.query({"$select": "x"})
    assert len(df) == 100003
    assert len(session.calls) == 3
    assert [c["params"]["$offset"] for c in session.calls] == [0, 50000, 100000]


def test_retries_on_429_then_succeeds():
    client, session = make_client(
        [FakeResponse(429), FakeResponse(503), FakeResponse(200, daily_csv(2))]
    )
    assert len(client.query({})) == 2
    assert len(session.calls) == 3


def test_gives_up_after_max_attempts():
    client, session = make_client([FakeResponse(503) for _ in range(5)])
    with pytest.raises(requests.HTTPError):
        client.query({})
    assert len(session.calls) == 5


def test_month_where_clause():
    client, session = make_client([FakeResponse(200, daily_csv(1))])
    df = client.fetch_month_daily(2024, 2)
    where = session.calls[0]["params"]["$where"]
    assert "created_date >= '2024-02-01T00:00:00'" in where
    assert "created_date < '2024-03-01T00:00:00'" in where
    assert list(df.columns) == ["day", "community_board", "complaint_type", "n"]
    assert df["day"].iloc[0] == date(2024, 2, 1)
    assert df["n"].dtype == "int64"


def test_december_rolls_year():
    client, session = make_client([FakeResponse(200, daily_csv(1))])
    client.fetch_month_daily(2023, 12)
    assert "created_date < '2024-01-01T00:00:00'" in session.calls[0]["params"]["$where"]


def test_missing_values_kept_as_strings_or_nan():
    text = (
        "day,community_board,complaint_type,n\n"
        "2024-02-01T00:00:00.000,0 Unspecified,Noise,4\n"
        "2024-02-01T00:00:00.000,,,2\n"
    )
    client, _ = make_client([FakeResponse(200, text)])
    df = client.fetch_month_daily(2024, 2)
    assert df["community_board"].iloc[0] == "0 Unspecified"
    assert len(df) == 2
    assert pd.isna(df["community_board"].iloc[1])
    assert pd.isna(df["complaint_type"].iloc[1])


def test_citywide_query():
    client, session = make_client(
        [FakeResponse(200, "day,n\n2024-02-01T00:00:00.000,10\n2024-02-02T00:00:00.000,12\n")]
    )
    df = client.fetch_month_citywide(2024, 2)
    p = session.calls[0]["params"]
    assert p["$select"] == "date_trunc_ymd(created_date) AS day, count(*) AS n"
    assert p["$group"] == "day"
    assert p["$order"] == "day"
    assert list(df.columns) == ["day", "n"]
    assert df["day"].iloc[1] == date(2024, 2, 2)


def test_token_header_sent():
    client, session = make_client([FakeResponse(200, daily_csv(1))], token="abc")
    client.query({})
    assert session.calls[0]["headers"]["X-App-Token"] == "abc"


def test_no_token_header_when_unset(monkeypatch):
    monkeypatch.delenv("SOCRATA_APP_TOKEN", raising=False)
    client, session = make_client([FakeResponse(200, daily_csv(1))])
    client.query({})
    assert "X-App-Token" not in (session.calls[0]["headers"] or {})


def test_months_to_fetch():
    got = months_to_fetch(date(2020, 1, 1), date(2020, 4, 15), {"2020-01", "2020-02", "2020-03"})
    assert got == [(2020, 3), (2020, 4)]


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    def boom(self, path, *a, **kw):
        Path(path).write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    target = tmp_path / "x.parquet"
    with pytest.raises(OSError):
        write_atomic(pd.DataFrame({"a": [1]}), target)
    assert not target.exists()
    assert not target.with_suffix(".tmp").exists()


def test_ingest_writes_both_files(tmp_path):
    settings = load_settings()
    settings = type(settings)(
        **{**settings.__dict__, "raw_dir": tmp_path, "start_date": date(2024, 3, 1)}
    )
    client, _ = make_client(
        [
            FakeResponse(200, daily_csv(2, "2024-03-01T00:00:00.000")),
            FakeResponse(200, "day,n\n2024-03-01T00:00:00.000,7\n"),
        ]
    )
    paths = ingest(settings, client, date(2024, 3, 15))
    assert sorted(p.name for p in paths) == ["citywide_2024-03.parquet", "daily_2024-03.parquet"]
    daily = pd.read_parquet(tmp_path / "daily_2024-03.parquet")
    assert list(daily.columns) == ["day", "community_board", "complaint_type", "n"]
    assert pd.read_parquet(tmp_path / "citywide_2024-03.parquet")["n"].iloc[0] == 7
