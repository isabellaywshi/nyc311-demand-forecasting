from datetime import date

import pandas as pd
import pytest

from nyc311.panel import PanelValidationError, build_panel, last_complete_week, validate_panel

WEEKS = [date(2024, 1, 1), date(2024, 1, 8), date(2024, 1, 15)]


def _get(panel, week, borocd, target):
    sel = panel[(panel.week_start == week) & (panel.borocd == borocd) & (panel.target == target)]
    assert len(sel) == 1
    return sel["count"].iloc[0]


def test_last_complete_week():
    assert last_complete_week(date(2026, 10, 26)) == date(2026, 10, 19)
    assert last_complete_week(date(2026, 10, 25)) == date(2026, 10, 12)


def test_panel_shape_and_zero_fill(raw_fixture, cats):
    panel, _ = build_panel(raw_fixture, cats)
    assert len(panel) == 71 * 6 * 3
    assert list(panel.columns) == [
        "week_start", "borocd", "target", "count", "is_special_area", "borough"
    ]
    assert _get(panel, WEEKS[0], 101, "noise") == 5
    assert _get(panel, WEEKS[0], 102, "noise") == 0
    assert panel[panel.borocd == 164].is_special_area.all()
    assert (panel[panel.borocd == 201].borough == "BRONX").all()


def test_weeks_are_mondays_and_partial_week_dropped(raw_fixture, cats):
    panel, _ = build_panel(raw_fixture, cats)
    assert sorted(panel.week_start.unique()) == WEEKS
    assert all(w.weekday() == 0 for w in panel.week_start.unique())


def test_total_includes_unmapped_types(raw_fixture, cats):
    panel, unmapped = build_panel(raw_fixture, cats)
    assert _get(panel, WEEKS[0], 101, "total") == 10
    assert list(unmapped.columns) == ["complaint_type", "n"]
    assert unmapped.set_index("complaint_type")["n"].to_dict() == {"Brand New Type": 2}


def test_unspecified_area_dropped_from_panel(raw_fixture, cats):
    panel, _ = build_panel(raw_fixture, cats)
    assert panel[(panel.week_start == WEEKS[1]) & (panel.target == "total")]["count"].sum() == 4


def test_validate_passes_on_good_panel(raw_fixture, cats):
    panel, _ = build_panel(raw_fixture, cats)
    validate_panel(panel, raw_fixture)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda p: p[p.week_start != WEEKS[1]], "gap"),
        (lambda p: p.assign(count=p["count"].where(p.index != 0, -1)), "negative"),
        (lambda p: pd.concat([p, p.iloc[[0]]], ignore_index=True), "duplicate"),
        (
            lambda p: p.assign(
                count=p["count"].where(~((p.borocd == 101) & (p.target == "total")), 1000)
            ),
            "reconcil",
        ),
    ],
)
def test_validate_flags_failures(raw_fixture, cats, mutate, needle):
    panel, _ = build_panel(raw_fixture, cats)
    with pytest.raises(PanelValidationError, match=needle):
        validate_panel(mutate(panel.copy()), raw_fixture)
