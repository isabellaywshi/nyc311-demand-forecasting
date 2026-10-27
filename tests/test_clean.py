import pytest

from nyc311.clean import (
    BOROUGH_CODES,
    REGULAR_DISTRICTS,
    SPECIAL_AREAS,
    borough_of,
    map_category,
    parse_area,
)


@pytest.mark.parametrize("raw,expected", [
    ("04 BRONX", 204), ("01 MANHATTAN", 101), ("18 BROOKLYN", 318), ("03 STATEN ISLAND", 503),
    ("64 MANHATTAN", 164), ("83 QUEENS", 483), ("95 STATEN ISLAND", 595),
    ("Unspecified BRONX", None), ("0 Unspecified", None), ("13 MANHATTAN", None),
    ("19 BROOKLYN", None), (None, None), ("", None),
])
def test_parse_area(raw, expected):
    assert parse_area(raw) == expected


def test_district_sets():
    assert len(REGULAR_DISTRICTS) == 59 and len(SPECIAL_AREAS) == 12


def test_borough_of():
    assert borough_of(204) == "BRONX"
    assert borough_of(164) == "MANHATTAN"
    assert borough_of(595) == "STATEN ISLAND"
    assert set(BOROUGH_CODES) == {"MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND"}


def test_map_category(cats):
    assert map_category("Noise - Residential", cats) == "noise"
    assert map_category("  heat/hot water ", cats) == "heat_hot_water"
    assert map_category("Blocked Driveway", cats) == "illegal_parking"
    assert map_category("Rodent", cats) is None
    assert map_category(None, cats) is None
