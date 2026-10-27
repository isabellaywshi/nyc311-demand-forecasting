import pytest

from nyc311.config import load_categories


@pytest.fixture
def cats():
    return load_categories()


@pytest.fixture
def raw_fixture(tmp_path):
    """Tiny raw dir: Mon 2024-01-01 .. Mon 2024-01-22 (newest, so week of the 22nd is partial)."""
    from datetime import date

    import pandas as pd

    rows = [
        (date(2024, 1, 1), "01 MANHATTAN", "Noise - Residential", 5),
        (date(2024, 1, 1), "01 MANHATTAN", "HEAT/HOT WATER", 3),
        (date(2024, 1, 3), "01 MANHATTAN", "Brand New Type", 2),
        (date(2024, 1, 9), "02 BRONX", "Noise - Street/Sidewalk", 4),
        (date(2024, 1, 10), "Unspecified BRONX", "Noise - Street/Sidewalk", 7),
        (date(2024, 1, 10), None, None, 1),
        (date(2024, 1, 17), "01 MANHATTAN", "Illegal Parking", 6),
        (date(2024, 1, 22), "01 MANHATTAN", "HEAT/HOT WATER", 9),
    ]
    daily = pd.DataFrame(rows, columns=["day", "community_board", "complaint_type", "n"])
    daily["n"] = daily["n"].astype("int64")
    daily.to_parquet(tmp_path / "daily_2024-01.parquet")
    city = daily.groupby("day", as_index=False)["n"].sum()
    city.to_parquet(tmp_path / "citywide_2024-01.parquet")
    return tmp_path


def make_panel(weekly, start, n_weeks, series=((101, "total"),)):
    """Synthetic complete panel. weekly(i, week_start, borocd, target) -> count for week index i."""
    from datetime import timedelta

    import pandas as pd

    rows = []
    for borocd, target in series:
        for i in range(n_weeks):
            ws = start + timedelta(weeks=i)
            rows.append((ws, borocd, target, weekly(i, ws, borocd, target)))
    df = pd.DataFrame(rows, columns=["week_start", "borocd", "target", "count"])
    df["borocd"] = df["borocd"].astype("int16")
    df["count"] = df["count"].astype("int32")
    df["is_special_area"] = False
    df["borough"] = "MANHATTAN"
    return df


@pytest.fixture
def panel_factory():
    return make_panel
