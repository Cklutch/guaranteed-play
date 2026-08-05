import streamlit as st
from utils import (
    init_session_state,
    keep_session_state_alive,
    render_league_settings_sidebar,
    build_live_rankings_df,
    draft_player
)

st.set_page_config(page_title="Live Rankings", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Live Rankings")

top1, top2 = st.columns(2)
top1.metric("Current Pick", st.session_state.get("draft_pick_number", 1))
top2.metric("Players Drafted", len(st.session_state.get("my_team", [])))

filter_cols = st.columns(3)

with filter_cols[0]:
    position_filter = st.selectbox(
        "Position",
        ["All", "QB", "RB", "WR", "TE"]
    )

with filter_cols[1]:
    sort_by = st.selectbox(
        "Sort by",
        ["Best Fit Score", "Best Value Score", "Best Available Score", "Projection", "ADP"]
    )

with filter_cols[2]:
    rows_to_show = st.selectbox(
        "Rows",
        [25, 50, 100, 150],
        index=1
    )

rankings_df = build_live_rankings_df(position_filter=position_filter)

if rankings_df.empty:
    st.info("No rankings available.")
    st.stop()

ascending = True if sort_by == "ADP" else False
rankings_df = rankings_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

st.subheader("Rankings Table")
st.dataframe(rankings_df.head(rows_to_show), use_container_width=True, hide_index=True)

st.divider()

st.subheader("Top 10 Actions")

top_actions_df = rankings_df.head(10).copy()

for idx, row in top_actions_df.iterrows():
    player_name = str(row["Player"])

    with st.container(border=True):
        c1, c2 = st.columns([4, 2])

        with c1:
            st.write(f"**{player_name}**")
            st.caption(
                f"{row['Position']} - {row['Team']} - "
                f"Proj: {row['Projection']} - ADP: {row['ADP']} - "
                f"Fit: {row['Best Fit Score']} - Value: {row['Best Value Score']}"
            )

        with c2:
            if st.button("Draft", key=f"live_draft_{idx}_{player_name}", use_container_width=True):
                drafted = draft_player(player_name)
                if drafted:
                    st.rerun()
