import numpy as np
import pandas as pd
import pytest

from nyc311.report import summarize, write_reports


def row(
    model,
    y,
    yhat,
    *,
    h=1,
    target="total",
    special=False,
    borough="BRONX",
    norm=10.0,
    naive=None,
    lo=np.nan,
    hi=np.nan,
    surge_actual=False,
):
    return {
        "model": model,
        "horizon": h,
        "target": target,
        "borocd": 201,
        "borough": borough,
        "is_special_area": special,
        "y": y,
        "yhat": yhat,
        "yhat_p10": lo,
        "yhat_p90": hi,
        "y_naive": y + 1 if naive is None else naive,
        "norm": norm,
        "surge_actual": surge_actual,
        "surge_alert": False,
    }


def frame():
    rows = [
        row("lgbm", 10, 12, lo=8, hi=9, h=1),  # err 2, y outside
        row("lgbm", 10, 9, lo=8, hi=11, h=2),  # err 1, inside
        row("seasonal_naive", 10, 14, h=1),
        row("seasonal_naive", 10, 10, h=2),
        # excluded from overall: special area, non-total target
        row("lgbm", 100, 0, special=True),
        row("lgbm", 100, 0, target="noise"),
        row("seasonal_naive", 100, 0, special=True),
        row("seasonal_naive", 100, 0, target="noise"),
        # rule baseline: recent_average
        row("recent_average", 14, 13, norm=10.0, surge_actual=True),  # 13>=12.5 alert
        row("recent_average", 5, 12, norm=10.0, surge_actual=False),  # 12<12.5 no alert
        row("recent_average", 5, 20, norm=2.0, surge_actual=False),  # norm<5 no alert
    ]
    return pd.DataFrame(rows)


def test_overall_values_and_filters():
    t = summarize(frame())
    o = t["overall"].set_index("model")
    assert o.loc["lgbm", "wape"] == pytest.approx(3 / 20)
    assert o.loc["seasonal_naive", "wape"] == pytest.approx(4 / 20)
    assert o.loc["lgbm", "coverage"] == pytest.approx(0.5)
    assert o.loc["lgbm", "width"] == pytest.approx(2.0)
    # mase: lgbm MAE 1.5 / naive MAE 1.0
    assert o.loc["lgbm", "mase"] == pytest.approx(1.5)
    assert np.isnan(o.loc["seasonal_naive", "coverage"])
    assert set(t) == {"overall", "by_horizon", "by_target", "by_borough", "by_area_type", "surge"}


def test_breakdown_slices():
    t = summarize(frame())
    bh = t["by_horizon"].set_index(["horizon", "model"])
    assert bh.loc[(1, "lgbm"), "wape"] == pytest.approx(0.2)
    assert bh.loc[(2, "lgbm"), "wape"] == pytest.approx(0.1)
    bt = t["by_target"].set_index(["target", "model"])
    assert bt.loc[("noise", "lgbm"), "wape"] == pytest.approx(1.0)
    assert ("noise", "lgbm") in bt.index
    ba = t["by_area_type"].set_index(["area_type", "model"])
    assert ba.loc[("special", "lgbm"), "wape"] == pytest.approx(1.0)
    assert ba.loc[("regular", "lgbm"), "wape"] == pytest.approx(0.15)
    # by_target and by_borough exclude special areas
    assert t["by_borough"].set_index(["borough", "model"]).loc[("BRONX", "lgbm"), "wape"] == (
        pytest.approx(0.15)
    )


def test_surge_rule_baseline_uses_recent_average_vs_norm():
    s = summarize(frame())["surge"].set_index("model")
    assert s.loc["rule_baseline", "precision"] == pytest.approx(1.0)
    assert s.loc["rule_baseline", "recall"] == pytest.approx(1.0)
    assert "lgbm" in s.index


def test_write_reports(tmp_path):
    write_reports(summarize(frame()), "validation", tmp_path)
    assert (tmp_path / "validation_overall.csv").exists()
    assert "lgbm" in (tmp_path / "validation_overall.md").read_text()
    assert (tmp_path / "figures" / "validation_wape_by_horizon.png").exists()
    assert (tmp_path / "figures" / "validation_wape_by_borough.png").exists()
    assert (tmp_path / "figures" / "validation_coverage_by_target.png").exists()


def test_write_test_summary_pass_fail(tmp_path):
    from nyc311.report import write_test_summary

    o = pd.DataFrame(
        {
            "model": ["seasonal_naive", "recent_average", "lgbm"],
            "wape": [0.2, 0.3, 0.1],
            "mase": [1.0, 1.5, 0.9],
            "coverage": [np.nan, np.nan, 0.95],
            "width": [np.nan, np.nan, 5.0],
        }
    )
    v = write_test_summary({"overall": o}, tmp_path)
    text = (tmp_path / "test_summary.md").read_text()
    assert v["lgbm_beats_seasonal_naive_wape"] and not v["coverage_70_90"]
    assert "FAIL: P10-P90 coverage 95.0%" in text and "PASS: LightGBM WAPE" in text


def test_update_settings_params_preserves_rest(tmp_path):
    from nyc311.cli import _update_settings_params

    p = tmp_path / "s.yaml"
    p.write_text("a: 1  # keep\nlgbm_params:\n  num_leaves: 31\n  learning_rate: 0.05\n")
    _update_settings_params(p, {"num_leaves": 63, "learning_rate": 0.03})
    assert p.read_text() == "a: 1  # keep\nlgbm_params:\n  num_leaves: 63\n  learning_rate: 0.03\n"
