import streamlit as st

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None

from rb_elite_age_analysis import (
    build_rb_elite_age_study,
    pull_historical_seasons_df,
    resolve_age_study_columns,
)


st.set_page_config(page_title="RB Elite Season Age Study", layout="wide")

st.title("RB Elite Season Age Study")
st.caption("Standalone case study. Not part of Guaranteed Play draft recommendations.")
st.caption("Rate % = players at that age inside Top N RB slots / all Top N RB slots.")

RATE_SPECS = [
    ("top36", "Top 36"),
    ("top24", "Top 24"),
    ("top12", "Top 12"),
    ("top5", "Top 5"),
    ("top3", "Top 3"),
    ("top1", "#1 Overall"),
]


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_historical_seasons(scoring, force_refresh):
    return pull_historical_seasons_df(scoring=scoring, force_refresh=force_refresh)


def _rate_chart(summary_df, rate_col, count_col, title):
    if px is None:
        return None

    return px.line(
        summary_df,
        x="Age",
        y=rate_col,
        markers=True,
        title=title,
        labels={
            "Age": "Age",
            rate_col: title.replace(" by Age", " (%)"),
        },
        hover_data={
            "Age": True,
            "total_rb_seasons": True,
            count_col: True,
            rate_col: ":.2f",
        },
    )


def _render_rate_chart(summary_df, rate_col, count_col, title):
    fig = _rate_chart(summary_df, rate_col, count_col, title)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        return

    fallback_df = summary_df[["Age", rate_col]].rename(columns={rate_col: "Rate %"})
    st.line_chart(fallback_df.set_index("Age"), use_container_width=True)
    st.caption("Install Plotly to enable rich chart tooltips.")


control_cols = st.columns([1, 1, 4])
with control_cols[0]:
    scoring = st.selectbox(
        "Scoring",
        options=["ppr", "half_ppr", "dad", "standard"],
        format_func=lambda value: {
            "ppr": "PPR",
            "half_ppr": "Half PPR",
            "dad": "Dad's Settings",
            "standard": "Standard",
        }[value],
    )
with control_cols[1]:
    force_refresh = st.button("Refresh source data", use_container_width=True)

try:
    with st.spinner("Pulling historical player-season data..."):
        seasons_df = _cached_historical_seasons(scoring, force_refresh)
except Exception as exc:
    st.error(
        "Could not pull the historical nflverse source data. Check your internet connection "
        "and try Refresh source data again."
    )
    st.caption(str(exc))
    st.stop()

if seasons_df.empty:
    st.info("No historical player-season data was returned from the source pull.")
    st.stop()

source_label = seasons_df.attrs.get("source_path")
source_type = seasons_df.attrs.get("source_type")
if source_label:
    st.caption(f"Using {source_type or 'pulled'} source: {source_label}")

columns = resolve_age_study_columns(seasons_df)
missing_columns = [key for key, column in columns.items() if column is None]
if missing_columns:
    st.error(
        "Historical data is missing required columns: "
        + ", ".join(missing_columns)
        + ". Required inputs are position, age, and positional_finish."
    )
    st.write("Detected columns:", list(seasons_df.columns))
    st.stop()

study = build_rb_elite_age_study(seasons_df, min_sample_size=1)
summary_df = study["summary"]

if summary_df.empty:
    st.warning(
        "No RB ages were available after filtering for RB rows with age and positional finish."
    )
    st.metric("Raw RB Seasons", study.get("raw_rb_rows", 0))
    st.stop()

peaks = study["peaks"]
peak_cols = st.columns(3)
for index, (threshold, label) in enumerate(RATE_SPECS):
    col = peak_cols[index % len(peak_cols)]
    key = f"{threshold}_rate"
    peak = peaks.get(key)
    if peak:
        col.metric(f"Peak Age for {label} Rate", peak["age"], f"{peak['rate_pct']:.2f}%")
    else:
        col.metric(f"Peak Age for {label} Rate", "N/A")

st.markdown("#### RB Age Share Inside Elite Finish Groups")
table_df = summary_df[[
    "Age",
    "total_rb_seasons",
    "top36_rb_seasons",
    "top36_rate_pct",
    "top24_rb_seasons",
    "top24_rate_pct",
    "top12_rb_seasons",
    "top12_rate_pct",
    "top5_rb_seasons",
    "top5_rate_pct",
    "top3_rb_seasons",
    "top3_rate_pct",
    "top1_rb_seasons",
    "top1_rate_pct",
]].rename(columns={
    "total_rb_seasons": "RB Seasons At Age",
    "top36_rb_seasons": "Top 36 Age Count",
    "top36_rate_pct": "Top 36 Share",
    "top24_rb_seasons": "Top 24 Age Count",
    "top24_rate_pct": "Top 24 Share",
    "top12_rb_seasons": "Top 12 Age Count",
    "top12_rate_pct": "Top 12 Share",
    "top5_rb_seasons": "Top 5 Age Count",
    "top5_rate_pct": "Top 5 Share",
    "top3_rb_seasons": "Top 3 Age Count",
    "top3_rate_pct": "Top 3 Share",
    "top1_rb_seasons": "#1 Overall Age Count",
    "top1_rate_pct": "#1 Overall Share",
})
st.dataframe(table_df, use_container_width=True, hide_index=True)

for threshold, label in RATE_SPECS:
    st.markdown(f"#### {label} Age Share")
    _render_rate_chart(
        summary_df,
        f"{threshold}_rate_pct",
        f"{threshold}_rb_seasons",
        f"{label} Age Share",
    )

with st.expander("Study Inputs", expanded=False):
    st.write("Detected columns:", columns)
    st.metric("Raw RB Seasons Used", study.get("raw_rb_rows", 0))
    st.metric("Ages Passing Sample Filter", study.get("filtered_ages", 0))
    st.caption("Rate % denominator: all Top N RB slots in the source timeline.")
