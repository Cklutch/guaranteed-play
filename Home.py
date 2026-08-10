"""
Top 300 rankings -- the whole app.

Deliberately a single page with no navigation, no draft-state controls, and
no draft actions. The other pages (Draft Mode, Player Cards, Team Outlook,
Draft Lab, Player Compare, Tier Desperation, Component Audit) and the
original landing page are parked in pages_archive/ rather than deleted;
moving any of them back into a pages/ directory restores the multipage nav
exactly as it was.

Scoring is unchanged -- this reads the same draftkit engine and the same
data/processed/master_players.csv that Draft Mode used. Because there is no
live draft state here, the board is evaluated at its draft-start position
(empty roster, pick 1), which is what a static preseason ranking list
should show.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from draftkit.age_context import age_note
from draftkit.archetypes import archetype_label, risk_profile
from draftkit.data_access import load_players_df
from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import init_session_state

TOP_N = 300

st.set_page_config(page_title="Top 300 Rankings", layout="wide")

# Still needed: the scoring engine reads roster_settings / drafted_players /
# current_pick_number from session state. init_session_state() seeds the
# defaults (empty roster, pick 1) that define the draft-start board.
init_session_state()


def _scoring_version():
    """
    Cache key covering everything the board depends on.

    st.cache_data hashes only the decorated function's own body, so edits to
    draft_analysis.py (where the scoring actually lives) or to the data files
    do NOT invalidate it -- the app kept serving a pre-fix board through
    repeated reloads. Keying on the mtimes of the real inputs fixes that.
    """
    watched = [
        Path("draftkit/draft_analysis.py"),
        Path("data/processed/master_players.csv"),
        Path("research/validation_v1/data/positional_tier_curve.csv"),
    ]
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in watched)


@st.cache_data(show_spinner="Building rankings...", max_entries=4)
def _rankings(_version):
    df = build_recommendation_rankings_df()
    if df.empty:
        return df

    # The scoring pipeline doesn't carry archetype_primary through, so join
    # it back from the source frame. (draft_analysis does emit a column
    # literally named `archetype`, but that's the legacy STEADY/RISKY bucket
    # -- empty for ~87% of players and derived from scores that are no
    # longer populated. Not the usage-derived archetypes.)
    source = load_players_df()
    if "archetype_primary" in source.columns and "player_name" in source.columns:
        arch = source[["player_name", "archetype_primary"]].drop_duplicates("player_name")
        df = df.merge(arch, on="player_name", how="left")
        df["Archetype"] = df["archetype_primary"].apply(archetype_label)
        df["Scoring Type"] = df["archetype_primary"].apply(
            lambda a: {"event": "TD / big play", "volume": "Volume", "mixed": "Mixed"}.get(
                risk_profile(a), ""
            )
        )

    # Age context -- display only. Age failed as a predictor (see
    # draftkit/age_context.py), so it never touches final_score.
    if "age" in source.columns and "player_name" in source.columns:
        ages = source[["player_name", "age"]].drop_duplicates("player_name")
        df = df.merge(ages, on="player_name", how="left", suffixes=("", "_src"))
        age_col = "age" if "age" in df.columns else "age_src"
        df["Age"] = pd.to_numeric(df[age_col], errors="coerce")
        df["Age Note"] = [age_note(p, a) for p, a in zip(df["position"], df["Age"])]
    return df


rankings_df = _rankings(_scoring_version())

if rankings_df.empty:
    st.error("No rankings available -- check that data/processed/master_players.csv exists.")
    st.stop()

# Restrict to players who are actually draftable BEFORE ranking.
#
# The source pool is Sleeper's full database (~3,941 rows), which includes
# practice-squad, retired, and deceased players. Only ~678 carry a real ADP
# and ~513 a real projection. Without this filter every player with neither
# lands on the same replacement-level fallback score (an exact tie at
# 35.70), so the board sorted them ALPHABETICALLY -- ranks 84-300 came back
# as Aaron Dykes, Aaron Green, Aaron Hernandez, Aaron Peck, Abram Smith...
# A "top 300" that is 217 alphabetized non-players is worse than useless.
#
# The market pricing a player, or a projection service publishing a real
# number for him, is the available evidence that he is draftable at all.
has_adp = pd.to_numeric(rankings_df.get("adp"), errors="coerce").notna()
has_real_projection = rankings_df.get("projection_source", pd.Series(dtype=object)) == "real"
rankings_df = rankings_df[has_adp | has_real_projection].copy()

if rankings_df.empty:
    st.error("No draftable players found (none have a real ADP or projection).")
    st.stop()

st.title("Top 300 Rankings")

position_filter = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE"], key="pos_filter")

board = rankings_df.copy()
# Rank is assigned BEFORE filtering so a player keeps his true overall rank
# when you narrow to one position -- filtering to RB should show RB1 as
# overall #1, not renumber him.
board.insert(0, "Rank", range(1, len(board) + 1))

# Where the market has this player, and how far we differ.
#
# ADP rank is computed over the SAME draftable pool as Rank, not from raw ADP
# values -- otherwise the two would be on different scales (raw ADP counts
# kickers and defenses this board excludes) and the delta would be junk.
#
# Sign: POSITIVE = we rank him EARLIER than the market (we're higher on him).
# Negative = we're lower. Named "vs ADP" with that convention stated in the
# column tooltip, since either reading is plausible to a fresh eye.
_adp_rank = pd.to_numeric(board["adp"], errors="coerce").rank(method="first")
board["ADP Rk"] = _adp_rank
board["vs ADP"] = _adp_rank - board["Rank"]

board = board.head(TOP_N)

if position_filter != "All":
    board = board[board["position"].astype(str).str.upper() == position_filter]

DISPLAY_COLUMNS = [
    "Rank", "player_name", "position", "team", "Age",
    "projection_points", "adp", "ADP Rk", "vs ADP", "final_score",
    "Archetype", "Scoring Type", "Age Note", "projection_source",
]
display_df = board[[c for c in DISPLAY_COLUMNS if c in board.columns]].rename(columns={
    "player_name": "Player",
    "position": "Pos",
    "team": "Team",
    "projection_points": "Proj",
    "adp": "ADP",
    "final_score": "Score",
    "projection_source": "Proj Source",
})

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    height=min(38 * len(display_df) + 38, 900),
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Proj": st.column_config.NumberColumn(format="%.1f"),
        "ADP": st.column_config.NumberColumn(format="%.1f"),
        "ADP Rk": st.column_config.NumberColumn(
            format="%d", width="small",
            help="Market rank within this same draftable pool (not raw ADP order).",
        ),
        "vs ADP": st.column_config.NumberColumn(
            format="%+d", width="small",
            help="Positive = we rank him EARLIER than the market. Negative = later.",
        ),
        "Score": st.column_config.NumberColumn(format="%.1f"),
    },
)

# Replacement-level rows are scored off a fallback floor rather than a real
# FantasyPros projection; surfacing the count keeps that from reading as a
# real forecast.
if "projection_source" in board.columns:
    fallback = int((board["projection_source"] == "replacement_fallback").sum())
    if fallback:
        st.caption(
            f"{fallback} of these {len(board)} players have no real projection and are "
            "scored at replacement level (see Proj Source)."
        )
