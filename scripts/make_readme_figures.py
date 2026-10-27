"""Build README figures from committed artifacts (run: uv run python scripts/make_readme_figures.py)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "roll_mean_52": "Average weekly volume, past 52 weeks",
    "roll_mean_4": "Average weekly volume, past 4 weeks",
    "roll_mean_13": "Average weekly volume, past 13 weeks",
    "lag_1": "Volume last week",
    "target": "Request category",
}


def shap_global(top_n: int = 5) -> Path:
    df = pd.read_parquet(ROOT / "app" / "data" / "shap_summary.parquet")
    g = df[df["level"] == "global"].sort_values("mean_abs_shap", ascending=False).head(top_n)
    g = g.iloc[::-1]
    labels = [LABELS.get(f, f) for f in g["feature"]]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.barh(labels, g["mean_abs_shap"], color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP value| (effect on scaled 1-week-ahead forecast)")
    ax.set_title("What the LightGBM model relies on most", loc="left", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, g["mean_abs_shap"].max() * 1.15)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "shap_global.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(shap_global())
