"""SHAP driver summaries for the h=1 point model (spec section 6)."""

from __future__ import annotations

import pandas as pd
import shap

from nyc311.features import FEATURE_COLUMNS

_TOP_K = 5
_COLUMNS = ["level", "borocd", "target", "feature", "mean_abs_shap", "rank"]


def _top(abs_shap: pd.DataFrame, level: str, borocd, target) -> list[dict]:
    means = abs_shap.mean().sort_values(ascending=False, kind="stable").head(_TOP_K)
    return [
        {
            "level": level,
            "borocd": borocd,
            "target": target,
            "feature": feat,
            "mean_abs_shap": float(val),
            "rank": rank,
        }
        for rank, (feat, val) in enumerate(means.items(), start=1)
    ]


def shap_summary(
    model, train_frame: pd.DataFrame, sample: int = 20000, seed: int = 0
) -> pd.DataFrame:
    """Top-5 features by mean |SHAP| for global, per-target and per area x target groups."""
    frame = train_frame
    if "horizon" in frame.columns:
        frame = frame[frame["horizon"] == 1]  # the h=1 model's training distribution
    if len(frame) > sample:
        frame = frame.sample(n=sample, random_state=seed)
    frame = frame.reset_index(drop=True)

    values = shap.TreeExplainer(model).shap_values(frame[FEATURE_COLUMNS])
    abs_shap = pd.DataFrame(abs(values), columns=FEATURE_COLUMNS)

    borocd = frame["borocd"].astype("int64")
    target = frame["target"].astype(str)
    rows = _top(abs_shap, "global", pd.NA, pd.NA)
    for t, idx in abs_shap.groupby(target, sort=True).groups.items():
        rows += _top(abs_shap.loc[idx], "target", pd.NA, t)
    keys = pd.DataFrame({"b": borocd, "t": target})
    for (b, t), idx in keys.groupby(["b", "t"], sort=True).groups.items():
        rows += _top(abs_shap.loc[idx], "area_target", int(b), t)

    out = pd.DataFrame(rows, columns=_COLUMNS)
    out["borocd"] = out["borocd"].astype("Int16")
    out["target"] = out["target"].astype("string")
    out["rank"] = out["rank"].astype("int64")
    return out
