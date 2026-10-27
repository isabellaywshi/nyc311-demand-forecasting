"""Evaluation tables and figures from backtest predictions (spec section 5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nyc311.metrics import coverage, mase, mean_width, surge_prf, wape
from nyc311.surge import flag_surge

MODEL_ORDER = ["seasonal_naive", "recent_average", "auto_ets", "mstl", "lgbm"]
METRICS = ["wape", "mase", "coverage", "width"]
_NAN = float("nan")

FILTERS = {
    "overall": "target = total, regular districts (59), horizons 1-4 pooled (headline)",
    "by_horizon": "target = total, regular districts",
    "by_target": "all targets, regular districts, horizons 1-4 pooled",
    "by_borough": "target = total, regular districts, horizons 1-4 pooled",
    "by_area_type": "target = total, regular vs special areas, horizons 1-4 pooled",
    "surge": (
        "all targets, regular districts, horizons 1-4 pooled; lgbm alert = yhat >= 1.25 x norm "
        "and norm >= 5; rule_baseline alert = same rule on recent_average"
    ),
}


def _metrics(g: pd.DataFrame) -> dict[str, float]:
    g = g.dropna(subset=["y", "yhat"])
    if g.empty:
        return dict.fromkeys(METRICS, _NAN)
    gm = g.dropna(subset=["y_naive"])
    gi = g.dropna(subset=["yhat_p10", "yhat_p90"])
    return {
        "wape": wape(g["y"], g["yhat"]),
        "mase": mase(gm["y"], gm["yhat"], gm["y_naive"]) if len(gm) else _NAN,
        "coverage": coverage(gi["y"], gi["yhat_p10"], gi["yhat_p90"]) if len(gi) else _NAN,
        "width": mean_width(gi["yhat_p10"], gi["yhat_p90"]) if len(gi) else _NAN,
    }


def _table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby([*by, "model"], sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip([*by, "model"], keys, strict=True)), **_metrics(g)})
    out = pd.DataFrame(rows, columns=[*by, "model", *METRICS])
    rank = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(len(MODEL_ORDER))
    return out.assign(_r=rank).sort_values([*by, "_r"]).drop(columns="_r").reset_index(drop=True)


def enrich(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach borough and is_special_area from the panel (one row per borocd)."""
    meta = panel[["borocd", "borough", "is_special_area"]].drop_duplicates("borocd")
    return preds.drop(columns=["borough", "is_special_area"], errors="ignore").merge(
        meta, on="borocd", how="left"
    )


def summarize(
    preds: pd.DataFrame, surge_threshold: float = 1.25, surge_min_norm: float = 5
) -> dict[str, pd.DataFrame]:
    regular = preds[~preds["is_special_area"]]
    total = preds[preds["target"] == "total"]
    headline = total[~total["is_special_area"]]
    by_area = total.assign(area_type=np.where(total["is_special_area"], "special", "regular"))

    surge_rows = []
    for name, model in (("lgbm", "lgbm"), ("rule_baseline", "recent_average")):
        g = regular[regular["model"] == model].dropna(subset=["yhat", "norm"])
        alert = flag_surge(g["yhat"], g["norm"], surge_threshold, surge_min_norm)
        actual = g["surge_actual"].astype(bool)
        surge_rows.append(
            {
                "model": name,
                **surge_prf(actual, alert),
                "n_actual": int(actual.sum()),
                "n_alerts": int(alert.sum()),
            }
        )
    surge = pd.DataFrame(
        surge_rows, columns=["model", "precision", "recall", "f1", "n_actual", "n_alerts"]
    )
    return {
        "overall": _table(headline, []),
        "by_horizon": _table(headline, ["horizon"]),
        "by_target": _table(regular, ["target"]),
        "by_borough": _table(headline, ["borough"]),
        "by_area_type": _table(by_area, ["area_type"]),
        "surge": surge,
    }


def _md(df: pd.DataFrame, period: str, key: str) -> str:
    head = f"### {period} - {key}\n\nFilters: {FILTERS[key]}\n\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def _fig_horizon(t: pd.DataFrame, period: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in MODEL_ORDER:
        g = t[t["model"] == m]
        if len(g):
            ax.plot(g["horizon"], g["wape"], marker="o", label=m, lw=2.5 if m == "lgbm" else 1.5)
    ax.set_xlabel("Forecast horizon (weeks)")
    ax.set_ylabel("WAPE")
    ax.set_xticks(sorted(t["horizon"].unique()))
    ax.set_title(f"WAPE by horizon ({period}; total demand, regular districts)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_bar(t: pd.DataFrame, col: str, ylabel: str, title: str, path: Path, band=None) -> None:
    g = t[t["model"] == "lgbm"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(g[col].astype(str), g[ylabel.lower()], color="#2a6f97")
    if band:
        ax.axhspan(*band, color="#888", alpha=0.15, label="target band 70-90%")
        ax.legend(frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_reports(tables: dict[str, pd.DataFrame], period: str, reports_dir: Path) -> None:
    reports_dir = Path(reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        df.to_csv(reports_dir / f"{period}_{key}.csv", index=False)
        (reports_dir / f"{period}_{key}.md").write_text(_md(df, period, key))
    _fig_horizon(tables["by_horizon"], period, fig_dir / f"{period}_wape_by_horizon.png")
    _fig_bar(
        tables["by_borough"], "borough", "WAPE",
        f"LightGBM WAPE by borough ({period}; total, regular districts)",
        fig_dir / f"{period}_wape_by_borough.png",
    )  # fmt: skip
    cov = tables["by_target"].assign(coverage=lambda d: d["coverage"] * 100)
    _fig_bar(
        cov, "target", "Coverage", f"LightGBM P10-P90 coverage by target ({period}, %)",
        fig_dir / f"{period}_coverage_by_target.png", band=(70, 90),
    )  # fmt: skip


def write_test_summary(tables: dict[str, pd.DataFrame], reports_dir: Path) -> dict[str, bool]:
    """Write reports/test_summary.md with each spec section 1 model criterion as PASS/FAIL."""
    o = tables["overall"].set_index("model")
    lg = o.loc["lgbm"]
    verdicts: dict[str, bool] = {}
    lines = [
        "# Test-period summary",
        "",
        f"Headline slice: {FILTERS['overall']}.",
        "",
        "| model | WAPE | MASE | coverage | width |",
        "|---|---|---|---|---|",
    ]
    for m, r in o.iterrows():
        lines.append(f"| {m} | {r['wape']:.4f} | {r['mase']:.4f} | {r['coverage']:.4f} | "
                     f"{r['width']:.2f} |")  # fmt: skip
    lines += ["", "## Success criteria (spec section 1)", ""]
    for base in ("seasonal_naive", "recent_average"):
        for metric in ("wape", "mase"):
            ok = bool(lg[metric] < o.loc[base, metric])
            verdicts[f"lgbm_beats_{base}_{metric}"] = ok
            lines.append(
                f"- {'PASS' if ok else 'FAIL'}: LightGBM {metric.upper()} {lg[metric]:.4f} < "
                f"{base} {o.loc[base, metric]:.4f}"
            )
    cov = float(lg["coverage"])
    ok = 0.70 <= cov <= 0.90
    verdicts["coverage_70_90"] = ok
    lines.append(f"- {'PASS' if ok else 'FAIL'}: P10-P90 coverage {cov:.1%} within 70-90%")
    lines += [
        "",
        f"Overall: {'ALL PASS' if all(verdicts.values()) else 'NOT ALL CRITERIA MET'}",
        "",
    ]
    (Path(reports_dir) / "test_summary.md").write_text("\n".join(lines))
    return verdicts
