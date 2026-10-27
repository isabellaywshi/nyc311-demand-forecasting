"""About: plain-language explanation of the question, data, method and limits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib
import streamlit as st

data = lib.setup_page("About this project", icon="ℹ️")
m = data.metrics


def _val(table, model, col):
    t = lib.metrics_table(m, table)
    t = t[t["model"] == model]
    return t[col].iloc[0] if len(t) and col in t else float("nan")


lgbm_wape = _val("overall", "lgbm", "wape")
naive_wape = _val("overall", "seasonal_naive", "wape")
cover = _val("overall", "lgbm", "coverage")
prec = _val("surge", "lgbm", "precision")
rec = _val("surge", "lgbm", "recall")

st.header("The question")
st.markdown(
    "How many 311 service requests will each New York City neighborhood generate next week, "
    "and where is demand likely to spike? Early warning helps agencies plan crews and "
    "inspections before the calls pile up."
)

st.header("The data")
st.markdown(
    f"Public 311 records from NYC Open Data, from {lib.DATA_START_YEAR} to {data.data_through}, "
    "counted per week for each "
    "of the city's community districts. Requests are grouped into five categories (noise, "
    "heat/hot water, illegal parking, sanitation, street conditions) plus an all-requests total. "
    f"{len(lib.SPECIAL_AREAS)} parks and airports that have their own district code are shown as "
    "special areas."
)

st.header("The method")
st.markdown(
    "Simple baselines (repeat last year, recent average) were compared with statistical models "
    "(AutoETS, MSTL) and one global LightGBM model trained across all districts. Models were "
    "judged with a rolling-origin backtest: repeatedly forecast the weeks after a cut-off date "
    "using only earlier data, then check against what happened. Each forecast comes with an 80% "
    "interval."
)
st.markdown(
    f"On the test period the LightGBM model has a WAPE of **{lib.fmt_ratio(lgbm_wape)}**, versus "
    f"**{lib.fmt_ratio(naive_wape)}** for the seasonal-naive baseline. Its 80% intervals contain "
    f"**{lib.fmt_ratio(cover)}** of actual values."
)

st.header("What counts as a surge")
st.markdown(
    "A district-week is flagged as a surge when forecast demand is **at least 25% above the "
    "seasonal norm**, the average for the same week in prior years, adjusted for the recent level, "
    "and the norm must be at least 5 requests. "
    f"On the test period, **{lib.fmt_ratio(prec)}** of alerts were real surges and "
    f"**{lib.fmt_ratio(rec)}** of real surges were caught."
)

st.header("Limitations")
st.markdown(
    "- A 311 request measures reporting, not actual need.\n"
    "- Neighborhoods use 311 unevenly, so counts compare poorly across districts.\n"
    "- The norm is built from prior years, so long-term growth in requests can inflate the "
    "number of surge alerts.\n"
    "- The model has no weather inputs, so heat waves and storms are not anticipated."
)
