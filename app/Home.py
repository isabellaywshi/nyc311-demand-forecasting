"""Home: map of next-weeks demand versus the seasonal norm, plus the surge watch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib
import plotly.express as px
import streamlit as st

data = lib.setup_page("NYC 311 demand forecast")
st.markdown(
    "Weekly 311 service-request demand for each NYC community district, forecast up to four "
    "weeks ahead and compared with what is normal for that time of year."
)

fc = data.forecasts
c1, c2 = st.columns([2, 1])
target = c1.selectbox(
    "Request category",
    lib.TARGETS,
    format_func=lib.target_label,
    key="home_target",
)
horizon = c2.slider("Weeks ahead", 1, 4, 1, key="home_horizon")

sel = fc[(fc["target"] == target) & (fc["horizon"] == horizon)].copy()
sel["area"] = sel["borocd"].map(lib.area_label)
sel["yhat_s"] = sel["yhat"].map(lib.fmt_int)
sel["norm_s"] = sel["norm"].map(lib.fmt_int)
sel["pct_s"] = sel["pct_vs_norm"].map(lib.fmt_pct)
week = sel["week_start"].iloc[0] if len(sel) else "n/a"
st.subheader(f"{lib.target_label(target)}: week of {week}")


def _on_map_select():
    state = st.session_state.get("home_map")
    points = state.selection.points if state else []
    if points and points[0].get("location") is not None:
        st.session_state["home_area"] = int(points[0]["location"])


fig = px.choropleth_map(
    sel,
    geojson=data.geojson,
    locations="borocd",
    featureidkey="properties.BoroCD",
    color="pct_vs_norm",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    range_color=(-0.5, 0.5),
    custom_data=["area", "yhat_s", "norm_s", "pct_s"],
    center={"lat": 40.70, "lon": -73.94},
    zoom=9.3,
    opacity=0.8,
    map_style="carto-positron",
    labels={"pct_vs_norm": "vs norm"},
)
fig.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>Forecast: %{customdata[1]}<br>"
        "Seasonal norm: %{customdata[2]}<br>vs norm: %{customdata[3]}<extra></extra>"
    ),
    marker_line_width=0.5,
)
fig.update_layout(
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
    height=520,
    coloraxis_colorbar={"title": "vs norm", "tickformat": "+.0%", "len": 0.6},
)
st.plotly_chart(
    fig,
    theme="streamlit",
    key="home_map",
    on_select=_on_map_select,
    selection_mode="points",
    width="stretch",
)
st.caption(
    "Color shows forecast demand relative to the seasonal norm (average of the same week in "
    "prior years); red is above normal, blue below, grey means no norm is available. "
    "Click a district to select it."
)

areas = lib.area_options(fc)
default_area = 204 if 204 in areas else areas[0]
st.session_state.setdefault("home_area", default_area)
area = st.selectbox("District", areas, format_func=lib.area_label, key="home_area")
row = sel[sel["borocd"] == area]
if len(row):
    r = row.iloc[0]
    st.markdown(
        f"**{lib.area_label(area)}**: forecast {lib.fmt_int(r['yhat'])} requests "
        f"(80% range {lib.fmt_int(r['yhat_p10'])}–{lib.fmt_int(r['yhat_p90'])}), "
        f"seasonal norm {lib.fmt_int(r['norm'])}, vs norm {lib.fmt_pct(r['pct_vs_norm'])}"
    )
if st.button("Open district detail", key="home_open"):
    st.query_params["area"] = str(area)
    st.query_params["target"] = target
    st.session_state["detail_area"] = area
    st.session_state["detail_target"] = target
    st.switch_page("pages/1_District_Detail.py")

st.subheader("Surge watch")
watch = lib.surge_watch(fc, target, horizon)
st.caption(
    f"Districts forecast at least 25% above their seasonal norm. {len(watch):,} of "
    f"{len(sel):,} districts are flagged."
)
if watch.empty:
    st.info("No districts are flagged for this selection.")
else:
    show = watch.drop(columns=["target"]).copy()
    show["pct_vs_norm"] = show["pct_vs_norm"] * 100
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "area": "District",
            "borough": "Borough",
            "week_start": "Week of",
            "yhat": st.column_config.NumberColumn("Forecast", format="localized"),
            "norm": st.column_config.NumberColumn("Seasonal norm", format="localized"),
            "pct_vs_norm": st.column_config.NumberColumn("vs norm", format="+%.0f%%"),
            "agency": "Agency",
        },
    )
