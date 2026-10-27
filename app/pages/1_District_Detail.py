"""District detail: history, forecast fan chart, drivers and all categories."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HIST_WEEKS = 78

data = lib.setup_page("District detail")
fc, hist = data.forecasts, data.history
areas = lib.area_options(fc)


def _initial(key, options, param):
    if key in st.session_state:
        return
    raw = st.query_params.get(param)
    try:
        val = int(raw) if param == "area" else raw
    except (TypeError, ValueError):
        val = None
    st.session_state[key] = val if val in options else options[0]


_initial("detail_area", areas, "area")
_initial("detail_target", lib.TARGETS, "target")

c1, c2 = st.columns(2)
area = c1.selectbox("District", areas, format_func=lib.area_label, key="detail_area")
target = c2.selectbox(
    "Request category", lib.TARGETS, format_func=lib.target_label, key="detail_target"
)
st.query_params["area"] = str(area)
st.query_params["target"] = target


def _series(a, t):
    h = hist[(hist["borocd"] == a) & (hist["target"] == t)].sort_values("week_start")
    f = fc[(fc["borocd"] == a) & (fc["target"] == t)].sort_values("horizon")
    return h.tail(HIST_WEEKS), f


h, f = _series(area, target)
color = lib.TARGET_COLORS[target]

if len(f):
    nxt = f.iloc[0]
    st.markdown(
        f"**{lib.area_label(area)}, {lib.target_label(target)}, week of {nxt['week_start']}:** "
        f"forecast {lib.fmt_int(nxt['yhat'])} requests "
        f"(80% range {lib.fmt_int(nxt['yhat_p10'])}–{lib.fmt_int(nxt['yhat_p90'])}), "
        f"seasonal norm {lib.fmt_int(nxt['norm'])}, vs norm {lib.fmt_pct(nxt['pct_vs_norm'])}"
    )
    if pd.isna(nxt["norm"]):
        st.caption("No seasonal norm is available for this area (too little history).")

fig = go.Figure()
if len(f):
    last = h.iloc[-1] if len(h) else None
    fx = list(f["week_start"])
    fig.add_trace(
        go.Scatter(
            x=fx,
            y=f["yhat_p90"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fx,
            y=f["yhat_p10"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=lib.rgba(color, 0.2),
            name="80% range",
        )
    )
    fx_line = ([last["week_start"]] if last is not None else []) + fx
    fy_line = ([last["count"]] if last is not None else []) + list(f["yhat"])
    fig.add_trace(
        go.Scatter(
            x=fx_line,
            y=fy_line,
            mode="lines+markers",
            name="Forecast",
            line={"color": color, "dash": "solid", "width": 2.5},
            marker={"size": 6},
        )
    )
    if f["norm"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=fx,
                y=f["norm"],
                mode="lines",
                name="Seasonal norm",
                line={"color": "grey", "dash": "dash", "width": 1.5},
            )
        )
fig.add_trace(
    go.Scatter(
        x=h["week_start"],
        y=h["count"],
        mode="lines",
        name="Actual",
        line={"color": color, "width": 1.8},
        opacity=0.9,
    )
)
fig.update_layout(
    height=lib.CHART_HEIGHT,
    hovermode="x unified",
    margin={"l": 0, "r": 0, "t": 10, "b": 0},
    yaxis_title="Requests per week",
    xaxis_title=None,
    legend={"orientation": "h", "y": 1.08},
)
st.plotly_chart(fig, theme="streamlit", width="stretch", key="detail_fan")
st.caption(
    "Solid line: weekly requests. Shaded band: 80% forecast interval. Dashed: seasonal norm."
)

# Drivers
st.subheader("What drives this forecast")
shap = data.shap
tmatch = shap["target"].eq(target).fillna(False)
cand = shap[(shap["level"] == "area_target") & shap["borocd"].eq(area).fillna(False) & tmatch]
scope = f"{lib.area_label(area)}, {lib.target_label(target)}"
if cand.empty:
    cand = shap[(shap["level"] == "target") & tmatch]
    scope = f"all districts, {lib.target_label(target)}"
if cand.empty:
    cand = shap[shap["level"] == "global"]
    scope = "all districts and categories"
if cand.empty:
    st.info("No driver summary is available.")
else:
    top = cand.assign(label=cand["feature"].map(lib.feature_label))
    top = top.groupby("label", as_index=False)["mean_abs_shap"].sum()
    top = top.nlargest(5, "mean_abs_shap").sort_values("mean_abs_shap")
    bar = go.Figure(
        go.Bar(x=top["mean_abs_shap"], y=top["label"], orientation="h", marker_color=color)
    )
    bar.update_layout(
        height=280,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis_title="Average influence on the forecast",
        yaxis_title=None,
    )
    st.plotly_chart(bar, theme="streamlit", width="stretch", key="detail_shap")
    st.caption(f"Top drivers for {scope} (mean absolute SHAP value).")

# Small multiples
st.subheader("All categories")
fig2 = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[lib.target_label(t) for t in lib.TARGETS],
    horizontal_spacing=0.06,
    vertical_spacing=0.15,
)
for i, t in enumerate(lib.TARGETS):
    r, c = divmod(i, 3)
    th, tf = _series(area, t)
    th = th.tail(52)
    col = lib.TARGET_COLORS[t]
    fig2.add_trace(
        go.Scatter(
            x=th["week_start"],
            y=th["count"],
            mode="lines",
            line={"color": col, "width": 1.5},
            showlegend=False,
            name=lib.target_label(t),
        ),
        row=r + 1,
        col=c + 1,
    )
    if len(tf):
        fig2.add_trace(
            go.Scatter(
                x=list(tf["week_start"]),
                y=tf["yhat"],
                mode="lines+markers",
                line={"color": col, "width": 2, "dash": "dot"},
                marker={"size": 4},
                showlegend=False,
                name="Forecast",
            ),
            row=r + 1,
            col=c + 1,
        )
fig2.update_layout(template="plotly_white", height=520, margin={"l": 0, "r": 0, "t": 30, "b": 0})
fig2.update_yaxes(rangemode="tozero")
st.plotly_chart(fig2, theme="streamlit", width="stretch", key="detail_grid")
st.caption("Last 52 weeks of actual requests (solid) and the four-week forecast (dotted).")
