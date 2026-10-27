"""Model performance: how accurate the forecasts were on the held-out test period."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib
import plotly.express as px
import streamlit as st

data = lib.setup_page("Model performance", icon="\U0001f4ca")
m = data.metrics
MODEL_NAMES = {
    "seasonal_naive": "Seasonal naive",
    "recent_average": "Recent average",
    "auto_ets": "AutoETS",
    "mstl": "MSTL",
    "lgbm": "LightGBM",
    "rule_baseline": "Rule baseline",
}
MODEL_COLORS = {
    "Seasonal naive": "#9AA0A6",
    "Recent average": "#C5A572",
    "AutoETS": "#72B7B2",
    "MSTL": "#B279A2",
    "LightGBM": "#4C78A8",
}
ORDER = list(MODEL_COLORS)


def named(df):
    df = df.copy()
    df["Model"] = df["model"].map(lambda x: MODEL_NAMES.get(x, x))
    return df


st.markdown(
    "Every model was scored on a held-out test period it never saw during training, using "
    "a rolling-origin backtest. Lower WAPE (weighted absolute percentage error) is better; "
    "MASE below 1 beats the seasonal-naive baseline."
)

st.subheader("Model comparison")
overall = named(lib.metrics_table(m, "overall"))
if overall.empty:
    st.info("No overall metrics available.")
else:
    tbl = overall[["Model", "wape", "mase", "coverage", "width"]].copy()
    tbl["Model"] = tbl["Model"].astype(str)
    st.dataframe(
        tbl,
        hide_index=True,
        width="stretch",
        column_config={
            "wape": st.column_config.NumberColumn("WAPE", format="percent"),
            "mase": st.column_config.NumberColumn("MASE", format="%.2f"),
            "coverage": st.column_config.NumberColumn("80% interval coverage", format="percent"),
            "width": st.column_config.NumberColumn("Interval width", format="localized"),
        },
    )
    st.caption(
        "Coverage is the share of actual values inside the 80% interval (target: 80%). "
        "Naive baselines have no interval."
    )

c1, c2 = st.columns(2)
with c1:
    st.subheader("Error by forecast horizon")
    bh = named(lib.metrics_table(m, "by_horizon"))
    if not bh.empty:
        fig = px.line(
            bh.sort_values("horizon"),
            x="horizon",
            y="wape",
            color="Model",
            markers=True,
            color_discrete_map=MODEL_COLORS,
            category_orders={"Model": ORDER},
        )
        fig.update_layout(
            height=360,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            xaxis={"title": "Weeks ahead", "dtick": 1},
            yaxis={"title": "WAPE", "tickformat": ".0%"},
            legend={"orientation": "h", "y": -0.25, "title": None},
        )
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="perf_horizon")
        st.caption("Error grows as the forecast looks further ahead.")
with c2:
    st.subheader("Error by borough (LightGBM)")
    bb = named(lib.metrics_table(m, "by_borough"))
    bb = bb[bb["model"] == "lgbm"].copy()
    if not bb.empty:
        bb["borough"] = bb["borough"].astype(str).str.title()
        bb = bb.sort_values("wape")
        fig = px.bar(
            bb,
            x="wape",
            y="borough",
            orientation="h",
            color_discrete_sequence=[MODEL_COLORS["LightGBM"]],
        )
        fig.update_layout(
            height=360,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            xaxis={"title": "WAPE", "tickformat": ".0%"},
            yaxis={"title": None},
        )
        st.plotly_chart(fig, theme="streamlit", width="stretch", key="perf_borough")
        st.caption("Larger boroughs with steadier volumes are easier to forecast.")

st.subheader("Surge alerts")
sg = named(lib.metrics_table(m, "surge"))
if not sg.empty:
    show = sg[["Model", "precision", "recall", "f1", "n_actual", "n_alerts"]]
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "precision": st.column_config.NumberColumn("Precision", format="percent"),
            "recall": st.column_config.NumberColumn("Recall", format="percent"),
            "f1": st.column_config.NumberColumn("F1", format="percent"),
            "n_actual": st.column_config.NumberColumn("Actual surges", format="localized"),
            "n_alerts": st.column_config.NumberColumn("Alerts raised", format="localized"),
        },
    )
    st.caption(
        "Precision: share of alerts that were real surges. Recall: share of real surges that "
        "were caught. The rule baseline flags a surge when the average of the last 4 weeks is already well above the norm."
    )
