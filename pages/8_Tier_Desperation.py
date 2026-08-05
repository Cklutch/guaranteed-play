import streamlit as st

from draftkit.draft_state import (
    init_session_state,
    keep_session_state_alive,
    get_next_pick_distance,
    handle_draft_action,
    get_current_pick_number,
    get_league_size,
)

from draftkit.draft_analysis import (
    build_recommendation_rankings_df,
    build_position_scarcity_df,
    build_position_cliff_df,
    build_turn_aware_cliff_df,
    get_falloff_recommendation,
    get_turn_aware_falloff_recommendation,
    build_position_tiers_df,
    build_tier_summary_df,
    get_tier_warning,
    build_desperation_targets_df,
    get_best_desperation_target,
    get_best_pick_recommendation,
    get_tier_build_debug_info,
)

from draftkit.data_access import (
    load_players_df,
    get_available_players_df,
    get_database_debug_info,
)

from draftkit.ui_helpers import render_league_settings_sidebar

st.set_page_config(page_title="Tier Desperation", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Tier Desperation")
st.caption("Use tiers first, then fall-off and scarcity as supporting signals.")

if st.session_state.get("last_action_message"):
    st.success(st.session_state["last_action_message"])
    st.session_state["last_action_message"] = ""

control_cols = st.columns(4)

with control_cols[0]:
    drop_threshold = st.selectbox(
        "Tier break projection gap",
        [5.0, 7.5, 10.0, 12.0, 15.0, 20.0],
        index=3
    )

with control_cols[1]:
    top_n = st.selectbox(
        "Players to evaluate for scarcity",
        [6, 8, 10, 12, 15, 20],
        index=3
    )

with control_cols[2]:
    manual_window_size = st.selectbox(
        "Manual wait size",
        [1, 2, 3, 4, 5, 6, 8, 10, 12],
        index=2
    )

with control_cols[3]:
    view_mode = st.selectbox(
        "Fall-off mode",
        ["Until my next pick", "Manual wait"]
    )

raw_df = load_players_df()
avail_df = get_available_players_df()
db_debug = get_database_debug_info()
debug_info = get_tier_build_debug_info()

tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)
tier_summary_df = build_tier_summary_df(drop_threshold=drop_threshold)
tier_warning = get_tier_warning(drop_threshold=drop_threshold)
desperation_df = build_desperation_targets_df(drop_threshold=drop_threshold)
best_target = get_best_desperation_target(drop_threshold=drop_threshold)
best_pick = get_best_pick_recommendation()
recommendation_rankings_df = build_recommendation_rankings_df()
if view_mode == "Manual wait":
    recommendation = get_falloff_recommendation(window_size=manual_window_size)
    cliff_df = build_position_cliff_df(window_size=manual_window_size)
else:
    recommendation = get_turn_aware_falloff_recommendation()
    cliff_df = build_turn_aware_cliff_df()

scarcity_df = build_position_scarcity_df(top_n=top_n)

with st.expander("Tier build debug", expanded=False):
    st.subheader("Database")
    st.write("Database path:", db_debug["db_path"])
    st.write("Database exists:", db_debug["db_exists"])
    st.write("Tables found:", db_debug["tables"])
    st.write("Selected table:", db_debug["selected_table"])
    st.write("Selected table columns:", db_debug["selected_table_columns"])
    st.write("Selected table row count:", db_debug["row_count"])
    st.write("Database error:", db_debug["error"])

    st.subheader("Tier Input")
    st.write("Message:", debug_info["message"])
    st.write("Raw shape:", debug_info["raw_shape"])
    st.write("Available shape:", avail_df.shape)
    st.write("Filtered shape:", debug_info["filtered_shape"])
    st.write("Raw columns:", debug_info["raw_columns"])
    st.write("Matched player column:", debug_info["player_col"])
    st.write("Matched position column:", debug_info["pos_col"])
    st.write("Matched team column:", debug_info["team_col"])
    st.write("Matched projection column:", debug_info["proj_col"])
    st.write("Matched ADP column:", debug_info["adp_col"])
    st.write("Projection dtype:", debug_info["projection_dtype"])
    st.write("Projection non-null count:", debug_info["projection_non_null_count"])
    st.write("Position values found:", debug_info["position_values"])

    if not raw_df.empty:
        st.subheader("Raw Player Data Preview")
        st.dataframe(raw_df.head(15), use_container_width=True)

    if not avail_df.empty:
        st.subheader("Available Player Data Preview")
        st.dataframe(avail_df.head(15), use_container_width=True)

if tiers_df.empty or tier_summary_df.empty:
    st.info("Not enough player data available to build tiers.")
    st.stop()

st.warning(f"{tier_warning['headline']} - {tier_warning['message']}")
st.caption(f"{recommendation['headline']} - {recommendation['message']}")

league_size = get_league_size()
next_pick_distance = get_next_pick_distance()

m1, m2, m3 = st.columns(3)
m1.metric("League Size", league_size)
m2.metric("Current Pick", get_current_pick_number())
m3.metric("Picks Until Next Turn", next_pick_distance)

st.divider()

if best_pick is not None:
    st.subheader("Best Overall Pick")

    with st.container(border=True):
        c1, c2 = st.columns([4, 2])

        with c1:
            st.write(
                f"**{best_pick['player']}** "
                f"({best_pick['position']}, {best_pick.get('team', '')})"
            )
            st.caption(
                f"Projection: {best_pick['projection_points']} | "
                f"Recommendation Score: {best_pick['final_score']}"
            )
            st.caption(
                " | ".join(
                    [
                        f"Need: {best_pick['score_breakdown']['need_bonus']}",
                        f"ADP: {best_pick['score_breakdown']['adp_bonus']}",
                        f"Tier: {best_pick['score_breakdown']['tier_bonus']}",
                        f"Fit: {best_pick['score_breakdown']['team_fit_bonus']}",
                    ]
                )
            )
            st.markdown("\n".join(f"- {reason}" for reason in best_pick["reasons"]))

        with c2:
            st.button(
                "Draft",
                key=f"best_pick_draft_{best_pick['player']}",
                use_container_width=True,
                on_click=handle_draft_action,
                args=(best_pick["player"],)
            )

st.divider()

if best_target is not None:
    st.subheader("Best Desperation Target")

    with st.container(border=True):
        c1, c2 = st.columns([4, 2])

        with c1:
            st.write(
                f"**{best_target['Player']}** "
                f"({best_target['Position']}, {best_target['Team']})"
            )
            st.caption(
                f"Tier {int(best_target['Tier'])} is {best_target['Tier Status'].lower()} "
                f"with {int(best_target['Players Left In Tier'])} left."
            )
            st.caption(
                f"Desperation Score: {best_target['Desperation Score']} | "
                f"Projection: {best_target['Projection']} | "
                f"ADP: {best_target['ADP']}"
            )

            if best_target["Next Tier Player"] is not None:
                st.caption(
                    f"Next tier starts with {best_target['Next Tier Player']} "
                    f"({best_target['Next Tier Projection']} proj). "
                    f"Tier drop: {best_target['Tier Drop']} | "
                    f"Wait drop: {best_target['Drop If Wait']}."
                )

            st.progress(
                min(float(best_target["Desperation Score"]) / 100.0, 1.0),
                text="Desperation score"
            )

        with c2:
            st.button(
                "Draft",
                key=f"best_target_draft_{best_target['Player']}",
                use_container_width=True,
                on_click=handle_draft_action,
                args=(best_target["Player"],)
            )

st.divider()

st.divider()

tier_top_cols = st.columns(4)

for i, pos in enumerate(["QB", "RB", "WR", "TE"]):
    pos_summary = tier_summary_df[tier_summary_df["Position"] == pos].copy()

    if pos_summary.empty:
        tier_top_cols[i].metric(pos, "No data")
        continue

    top_tier = pos_summary.iloc[0]
    tier_top_cols[i].metric(
        pos,
        f"{int(top_tier['Players Left In Tier'])} left",
        delta=top_tier["Tier Status"]
    )

st.divider()

st.subheader("Priority Targets")

if desperation_df.empty:
    st.caption("No thin or shrinking tiers right now.")
else:
    for idx, row in desperation_df.iterrows():
        player_name = row["Player"]

        with st.container(border=True):
            c1, c2 = st.columns([4, 2])

            with c1:
                st.write(f"**{player_name}** ({row['Position']}, {row['Team']})")
                st.caption(
                    f"Tier {int(row['Tier'])} is {row['Tier Status'].lower()} with "
                    f"{int(row['Players Left In Tier'])} left."
                )
                st.caption(
                    f"Desperation Score: {row['Desperation Score']} | "
                    f"Projection: {row['Projection']} | ADP: {row['ADP']} | "
                    f"Need Weight: {row['Need Weight']}"
                )

                if row["Next Tier Player"] is not None:
                    st.caption(
                        f"Next tier: {row['Next Tier Player']} "
                        f"({row['Next Tier Projection']} proj) | "
                        f"Tier drop: {row['Tier Drop']} | "
                        f"Wait drop: {row['Drop If Wait']} | "
                        f"{row['Draft Signal']}"
                    )
                else:
                    st.caption(
                        f"Wait drop: {row['Drop If Wait']} | {row['Draft Signal']}"
                    )

            with c2:
                st.button(
                    "Draft",
                    key=f"desperation_draft_{idx}_{player_name}",
                    use_container_width=True,
                    on_click=handle_draft_action,
                    args=(player_name,)
                )

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Tiers", "Targets", "Recommendations", "Fall-Off", "Scarcity"]
)

with tab1:
    st.subheader("Tier Summary")
    st.dataframe(tier_summary_df, use_container_width=True, hide_index=True)

    st.subheader("Tier Details")
    st.dataframe(tiers_df, use_container_width=True, hide_index=True)

with tab2:
    if desperation_df.empty:
        st.info("No desperation targets available.")
    else:
        st.subheader("Desperation Targets")
        st.dataframe(desperation_df, use_container_width=True, hide_index=True)

with tab3:
    if recommendation_rankings_df.empty:
        st.info("No recommendation rankings available.")
    else:
        st.subheader("Risk-Aware Recommendation Rankings")
        display_cols = [
            "player_name",
            "position",
            "team",
            "projection_points",
            "archetype",
            "need_bonus",
            "adp_bonus",
            "tier_bonus",
            "team_fit_bonus",
            "final_score",
        ]
        st.dataframe(
            recommendation_rankings_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    if cliff_df.empty:
        st.info("Not enough player data available to calculate fall-off.")
    else:
        st.subheader("Fall-Off Table")
        st.dataframe(cliff_df, use_container_width=True, hide_index=True)

        st.subheader("Fall-Off Chart")
        chart_df = cliff_df.set_index("Position")[["Drop If Wait"]]
        st.bar_chart(chart_df, use_container_width=True)

with tab5:
    if scarcity_df.empty:
        st.info("Not enough player data available to calculate scarcity.")
    else:
        st.subheader("Scarcity Table")
        st.dataframe(scarcity_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("How to use this")
st.caption("Start with the best desperation target. It combines tier thinness, roster need, and how much value may disappear before your next pick.")
