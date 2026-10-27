"""Community-district parsing and complaint-type to category mapping."""

import re

from nyc311.config import Categories

REGULAR_DISTRICTS: frozenset[int] = frozenset(
    [*range(101, 113), *range(201, 213), *range(301, 319), *range(401, 415), *range(501, 504)]
)
SPECIAL_AREAS: frozenset[int] = frozenset(
    [164, 226, 227, 228, 355, 356, 480, 481, 482, 483, 484, 595]
)
BOROUGH_CODES = {"MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3, "QUEENS": 4, "STATEN ISLAND": 5}
_BOROUGH_BY_CODE = {code: name for name, code in BOROUGH_CODES.items()}
_AREA_RE = re.compile(r"(\d{1,2}) (MANHATTAN|BRONX|BROOKLYN|QUEENS|STATEN ISLAND)")
_VALID_AREAS = REGULAR_DISTRICTS | SPECIAL_AREAS


def parse_area(community_board: str | None) -> int | None:
    """Map e.g. '04 BRONX' to borocd 204; None if not one of the 71 known areas."""
    if not community_board:
        return None
    match = _AREA_RE.fullmatch(community_board)
    if match is None:
        return None
    borocd = BOROUGH_CODES[match.group(2)] * 100 + int(match.group(1))
    return borocd if borocd in _VALID_AREAS else None


def borough_of(borocd: int) -> str:
    return _BOROUGH_BY_CODE[borocd // 100]


def map_category(complaint_type: str | None, cats: Categories) -> str | None:
    """Exact matches win over prefix matches; comparison is case-insensitive and trimmed."""
    if complaint_type is None:
        return None
    key = complaint_type.strip().lower()
    if key in cats.exact:
        return cats.exact[key]
    for prefix, target in cats.prefixes:
        if key.startswith(prefix):
            return target
    return None
