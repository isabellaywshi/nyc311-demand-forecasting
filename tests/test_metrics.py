import pytest

from nyc311.metrics import coverage, mase, mean_width, surge_prf, wape


def test_wape():
    assert wape([10, 20], [12, 18]) == pytest.approx(4 / 30)


def test_mase():
    assert mase([10, 20], [11, 21], [8, 24]) == pytest.approx(1 / 3)


def test_mase_zero_denominator_is_nan():
    import math

    assert math.isnan(mase([1, 2], [2, 3], [1, 2]))


def test_coverage_and_width():
    assert coverage([1, 5, 9], [0, 6, 8], [2, 7, 10]) == pytest.approx(2 / 3)
    assert mean_width([0, 6], [2, 8]) == 2


def test_surge_prf():
    r = surge_prf([True, True, False, False], [True, False, True, False])
    assert r == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_surge_prf_undefined_is_zero():
    assert surge_prf([False, False], [False, False]) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
