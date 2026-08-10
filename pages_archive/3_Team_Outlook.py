import math

import pandas as pd
import streamlit as st

from draftkit.data_access import load_players_df, safe_col
from draftkit.draft_analysis import get_team_profile
from draftkit.draft_state import (
    get_league_size,
    get_my_team_positions,
    get_player_position_map,
    init_session_state,
    keep_session_state_alive,
    remove_from_my_team,
)
from draftkit.ui_helpers import render_league_settings_sidebar


st.set_page_config(page_title="Team Outlook", layout="wide")

init_session_state()
keep_session_state_alive()
render_league_settings_sidebar()

st.title("Team Outlook")
st.caption("Compare your roster profile against estimated leaguemate rosters.")

players_df = load_players_df()
my_team = st.session_state.get("my_team", [])

if players_df.empty:
    st.error("No player data loaded.")
    st.stop()

if not my_team:
    st.info("No players drafted to your roster yet.")
    st.stop()

player_col = safe_col(players_df, ["player_name", "Player", "player", "name", "full_name"])
pos_col = safe_col(players_df, ["position", "pos", "Pos", "Position", "POS"])
team_col = safe_col(players_df, ["team", "Team", "team_abbr", "TEAM"])
proj_col = safe_col(
    players_df,
    ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"],
)
adp_col = safe_col(players_df, ["adp", "ADP", "rank", "Rank"])

if player_col is None:
    st.error("Could not find a player-name column in the loaded data.")
    st.stop()

if proj_col is None:
    st.error("Could not find a projection column in the loaded data.")
    st.stop()

players_df = players_df.copy()
players_df[proj_col] = pd.to_numeric(players_df[proj_col], errors="coerce").fillna(0)

if adp_col is not None:
    players_df[adp_col] = pd.to_numeric(players_df[adp_col], errors="coerce")
else:
    adp_col = "_Estimated ADP"
    players_df[adp_col] = players_df[proj_col].rank(ascending=False, method="first")

if pos_col is None:
    pos_col = "_Position"
    players_df[pos_col] = "Unknown"

if team_col is None:
    team_col = "_Team"
    players_df[team_col] = ""

position_order = ["QB", "RB", "WR", "TE", "DST", "K"]
position_rank = {position: index for index, position in enumerate(position_order)}
roster_settings = st.session_state.get("roster_settings", {})

get_player_position_map(players_df)
position_counts = get_my_team_positions(players_df)
team_profile = get_team_profile()


def clean_name(value):
    return str(value).strip()


def get_player_rows(player_names):
    wanted = {clean_name(name).lower() for name in player_names}
    if not wanted:
        return pd.DataFrame(columns=players_df.columns)

    return players_df[
        players_df[player_col].astype(str).str.strip().str.lower().isin(wanted)
    ].copy()


def build_roster_rows(player_names):
    roster_df = get_player_rows(player_names)
    if roster_df.empty:
        return pd.DataFrame(
            columns=["Player Name", "Position", "Team", "Projection", "ADP", "_position_rank"]
        )

    out = pd.DataFrame(
        {
            "Player Name": roster_df[player_col].astype(str),
            "Position": roster_df[pos_col].astype(str).str.upper(),
            "Team": roster_df[team_col].astype(str),
            "Projection": roster_df[proj_col],
            "ADP": roster_df[adp_col],
        }
    )
    out["_position_rank"] = out["Position"].map(position_rank).fillna(len(position_order))
    return out.sort_values(["_position_rank", "Projection"], ascending=[True, False])


def starter_projection(roster_df):
    if roster_df.empty:
        return 0.0

    used_indexes = set()
    total = 0.0

    for position in ["QB", "RB", "WR", "TE", "DST", "K"]:
        target = int(roster_settings.get(position, 0))
        if target <= 0:
            continue

        pos_df = roster_df[roster_df["Position"] == position].sort_values(
            "Projection",
            ascending=False,
        )

        for idx, row in pos_df.head(target).iterrows():
            total += float(row["Projection"])
            used_indexes.add(idx)

    flex_target = int(roster_settings.get("FLEX", 0))
    if flex_target > 0:
        flex_df = roster_df[
            roster_df["Position"].isin(["RB", "WR", "TE"])
            & ~roster_df.index.isin(used_indexes)
        ].sort_values("Projection", ascending=False)

        total += float(flex_df.head(flex_target)["Projection"].sum())

    return round(total, 2)


def score_roster(team_name, player_names, kind):
    roster_df = build_roster_rows(player_names)

    if roster_df.empty:
        return {
            "Team": team_name,
            "Type": kind,
            "Players": 0,
            "Projected Points": 0.0,
            "Starter Projection": 0.0,
            "Safe": 0.0,
            "Boom/Bust": 0.0,
            "Risk": 0.0,
            "Depth": 0.0,
        }

    projections = roster_df["Projection"].astype(float)
    adp_values = pd.to_numeric(roster_df["ADP"], errors="coerce").fillna(
        roster_df["ADP"].median()
    )

    total_projection = round(float(projections.sum()), 2)
    starter_points = starter_projection(roster_df)
    average_adp = float(adp_values.mean()) if len(adp_values) else 100.0
    projection_spread = float(projections.std()) if len(projections) > 1 else 0.0
    average_projection = float(projections.mean()) if len(projections) else 0.0

    risk = min(100.0, max(0.0, (average_adp / 2.0) + (projection_spread / 3.0)))
    safe = max(0.0, min(100.0, 100.0 - risk))
    boom_bust = min(100.0, max(0.0, (projection_spread / max(average_projection, 1.0)) * 100))

    position_depth_scores = []
    for position in ["QB", "RB", "WR", "TE"]:
        target = max(int(roster_settings.get(position, 0)), 1)
        have = int((roster_df["Position"] == position).sum())
        position_depth_scores.append(min(have / target, 1.5))

    depth = round((sum(position_depth_scores) / len(position_depth_scores)) * 100, 1)

    return {
        "Team": team_name,
        "Type": kind,
        "Players": len(roster_df),
        "Projected Points": total_projection,
        "Starter Projection": starter_points,
        "Safe": round(safe, 1),
        "Boom/Bust": round(boom_bust, 1),
        "Risk": round(risk, 1),
        "Depth": depth,
    }


def build_estimated_league():
    league_size = max(get_league_size(), 2)
    opponent_count = league_size - 1
    target_roster_size = max(len(my_team), 1)
    my_names = {clean_name(name).lower() for name in my_team}

    other_drafted = [
        clean_name(entry.get("player_name"))
        for entry in st.session_state.get("draft_log", [])
        if entry.get("action_type") == "drafted"
    ]
    other_drafted = [name for name in other_drafted if name and name.lower() not in my_names]

    unavailable = my_names | {name.lower() for name in other_drafted}
    fill_pool = players_df[
        ~players_df[player_col].astype(str).str.strip().str.lower().isin(unavailable)
    ].sort_values([adp_col, proj_col], ascending=[True, False])

    opponent_rosters = [[] for _ in range(opponent_count)]

    for idx, player_name in enumerate(other_drafted):
        opponent_rosters[idx % opponent_count].append(player_name)

    for _, row in fill_pool.iterrows():
        if all(len(roster) >= target_roster_size for roster in opponent_rosters):
            break

        next_index = min(range(opponent_count), key=lambda i: len(opponent_rosters[i]))
        opponent_rosters[next_index].append(clean_name(row[player_col]))

    rows = [score_roster("My Team", my_team, "Actual")]

    for idx, roster in enumerate(opponent_rosters, start=1):
        rows.append(score_roster(f"Leaguemate {idx}", roster, "Estimated"))

    comparison_df = pd.DataFrame(rows)

    for metric in ["Projected Points", "Starter Projection", "Safe", "Boom/Bust", "Risk", "Depth"]:
        rank_col = f"{metric} Rank"
        ascending = metric == "Risk"
        comparison_df[rank_col] = comparison_df[metric].rank(
            ascending=ascending,
            method="min",
        ).astype(int)

    return comparison_df


roster_df = build_roster_rows(my_team)
comparison_df = build_estimated_league()
my_row = comparison_df[comparison_df["Team"] == "My Team"].iloc[0]
league_average = comparison_df[comparison_df["Team"] != "My Team"].mean(numeric_only=True)

st.caption(
    "Opponent rosters are estimated from drafted non-my-team players and the remaining ranked player pool."
)

top_cols = st.columns(5)
top_cols[0].metric(
    "Projected Points",
    my_row["Projected Points"],
    round(my_row["Projected Points"] - league_average.get("Projected Points", 0), 2),
)
top_cols[1].metric(
    "Starter Projection",
    my_row["Starter Projection"],
    round(my_row["Starter Projection"] - league_average.get("Starter Projection", 0), 2),
)
top_cols[2].metric("Safe", my_row["Safe"], f"Rank {my_row['Safe Rank']}")
top_cols[3].metric("Risk", my_row["Risk"], f"Rank {my_row['Risk Rank']}")
top_cols[4].metric("Depth", my_row["Depth"], f"Rank {my_row['Depth Rank']}")

profile_cols = st.columns(4)
profile_cols[0].metric("Profile", team_profile["profile"])
profile_cols[1].metric("Volatility", team_profile["volatility"])
profile_cols[2].metric("Avg Injury Risk", team_profile["average_injury_risk"])
profile_cols[3].metric("Avg Durability", team_profile["average_durability"])

st.divider()

left, right = st.columns([1.4, 1])

with left:
    st.subheader("League Comparison")
    league_cols = [
        "Team",
        "Type",
        "Players",
        "Projected Points",
        "Starter Projection",
        "Safe",
        "Boom/Bust",
        "Risk",
        "Depth",
    ]
    st.dataframe(
        comparison_df[league_cols].sort_values(
            "Starter Projection",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = comparison_df.set_index("Team")[["Starter Projection"]]
    st.bar_chart(chart_df, use_container_width=True)

with right:
    st.subheader("Roster Construction")

    breakdown_df = pd.DataFrame(
        [
            {
                "Position": position,
                "Count": position_counts.get(position, 0),
                "Target": int(roster_settings.get(position, 0)),
            }
            for position in ["QB", "RB", "WR", "TE", "DST", "K"]
        ]
    )
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    st.subheader("Profile Read")
    profile_lines = []
    profile_lines.append(f"Recommendation engine classifies this roster as {team_profile['profile']}.")

    if my_row["Starter Projection Rank"] <= max(math.ceil(get_league_size() / 3), 1):
        profile_lines.append("High-end projected scoring profile.")
    elif my_row["Starter Projection Rank"] > math.ceil(get_league_size() * 2 / 3):
        profile_lines.append("Projected points trail the league estimate.")
    else:
        profile_lines.append("Projected points sit near the middle of the league.")

    if my_row["Risk Rank"] <= max(math.ceil(get_league_size() / 3), 1):
        profile_lines.append("Lower-risk build compared with the room.")
    elif my_row["Risk Rank"] > math.ceil(get_league_size() * 2 / 3):
        profile_lines.append("Higher-risk build; upside depends on volatile profiles hitting.")

    if my_row["Depth Rank"] <= max(math.ceil(get_league_size() / 3), 1):
        profile_lines.append("Depth is a relative strength.")
    elif my_row["Depth Rank"] > math.ceil(get_league_size() * 2 / 3):
        profile_lines.append("Depth is a roster need.")

    for line in profile_lines:
        st.write(f"- {line}")

st.divider()

st.subheader("Current Roster")

display_cols = ["Player Name", "Position", "Team", "Projection", "ADP"]
st.dataframe(
    roster_df[display_cols],
    use_container_width=True,
    hide_index=True,
)

st.markdown("#### Remove Player")

for index, row in roster_df.iterrows():
    player_name = row["Player Name"]

    with st.container(border=True):
        info_col, button_col = st.columns([4, 1])

        with info_col:
            st.write(
                f"**{player_name}** ({row['Position']}, {row['Team']}) - "
                f"Proj: {row['Projection']} | ADP: {row['ADP']}"
            )

        with button_col:
            if st.button(
                "Remove",
                key=f"remove_my_team_{index}_{player_name}",
                use_container_width=True,
            ):
                remove_from_my_team(player_name, make_available=True)
                st.rerun()

st.divider()

with st.expander("Scoring Notes", expanded=False):
    st.write("Projected Points uses available player projections.")
    st.write("Starter Projection uses current roster settings and FLEX if configured.")
    st.write("Safe and Risk are derived from ADP and projection spread.")
    st.write("Boom/Bust is derived from projection volatility within the roster.")
    st.write("Depth compares roster counts against configured roster settings.")
    st.write("Profile, volatility, injury risk, and durability use draft_analysis.get_team_profile().")
