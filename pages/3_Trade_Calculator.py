"""Trade calculator -- judge a proposed trade in in-season value.

Values every player by the position-adjusted Week-N Base Value score from
draftkit.in_season (blended actuals + preseason, re-scored through the same
engine as the live board), which is the right currency for an in-season
trade -- preseason draft value has already gone stale on risers and busts.

Link a Sleeper league to load each team's real roster into the two sides,
or fall back to searching the whole player pool by name. Logic lives in
draftkit/trade.py; this page is wiring + rendering only.
"""

import pandas as pd
import streamlit as st

from draftkit.draft_analysis import build_recommendation_rankings_df
from draftkit.draft_state import init_session_state
from draftkit.in_season import build_in_season_board
from draftkit.trade import build_value_index, evaluate_trade, fetch_league_teams
from draftkit.ui_helpers import render_tool_nav


@st.cache_data(show_spinner="Loading in-season values...", ttl=1800)
def _value_index():
    board = build_in_season_board(build_recommendation_rankings_df(), scoring="half_ppr")
    idx = build_value_index(board)
    meta = dict(board.attrs.get("in_season_meta", {}))
    return idx, meta


@st.cache_data(show_spinner="Reading league from Sleeper...", ttl=300)
def _league_teams(league_id):
    return fetch_league_teams(league_id)


st.set_page_config(page_title="Trade Calculator", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .stApp, header[data-testid="stHeader"] { background: #0d0f10 !important; }
      section[data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none !important; }
      .block-container { padding-top: 1.5rem; max-width: 960px; }
      body, .stApp p, .stApp div, .stApp span { color: #eef1ec; }
      div[data-testid="stVerticalBlockBorderWrapper"] { background: #141718; border-color: #23282b !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_session_state()
render_tool_nav("Trade Calculator")

value_index, meta = _value_index()
wk = meta.get("current_week", "?")

st.title("Trade calculator")
st.caption(
    f"Week {wk}: players valued by position-adjusted in-season Base Value "
    "(real production blended with the preseason projection). A scarce RB is "
    "worth more than a WR with equal points -- that's what makes a cross-"
    "position verdict honest."
)

# name -> id, for the whole pool (manual search fallback and label building).
name_to_id = {}
id_to_label = {}
for pid, rec in value_index.items():
    label = f"{rec['name']} ({rec['position']}{('-' + rec['team']) if rec['team'] else ''}) · {rec['score']:.0f}"
    name_to_id[rec["name"]] = pid
    id_to_label[pid] = label


def _ids_from_names(names):
    return [name_to_id[n] for n in names if n in name_to_id]


# --- Optional Sleeper league link: populate each side from real rosters ---
league_input = st.text_input(
    "Sleeper league (optional)",
    key="trade_league_id",
    placeholder="league id or https://sleeper.com/leagues/……",
    help="Loads each team's real roster into the dropdowns. Leave blank to search all players.",
)

team_players = None
if league_input.strip():
    teams = _league_teams(league_input)
    if not teams:
        st.warning("Couldn't read that league (bad id, private, or Sleeper unavailable). Falling back to full-pool search below.")
    else:
        # Only players our value index knows get shown (skips K/DST/deep names).
        def _roster_names(t):
            out = []
            for pid in t["player_ids"]:
                rec = value_index.get(pid)
                if rec:
                    out.append(rec["name"])
            return sorted(out)

        labeled = {f"{t['owner']} ({len(_roster_names(t))})": _roster_names(t) for t in teams}
        team_players = labeled
        st.success(f"Loaded {len(teams)} teams.")

col_a, col_b = st.columns(2)


def _side_picker(col, title, key):
    col.subheader(title)
    if team_players:
        team_label = col.selectbox("Team", list(team_players.keys()), key=f"{key}_team")
        options = team_players.get(team_label, [])
    else:
        options = sorted(name_to_id.keys())
    picks = col.multiselect("Players", options, key=f"{key}_players")
    return picks


give_names = _side_picker(col_a, "You give", "give")
get_names = _side_picker(col_b, "You get", "get")

st.divider()

if not give_names or not get_names:
    st.info("Pick at least one player on each side to evaluate the trade.")
else:
    result = evaluate_trade(
        _ids_from_names(give_names), _ids_from_names(get_names), value_index,
        label_a="You give", label_b="You get",
    )

    verdict = result["verdict"]
    if verdict.startswith("Fair"):
        st.success(f"⚖️  {verdict}  ·  {result['gap_pct']}% apart")
    else:
        st.warning(f"{verdict}  ·  {result['gap_pct']}% apart")

    mcol = st.columns(2)
    mcol[0].metric("You give — total value", f"{result['a_total']:.0f}")
    mcol[1].metric("You get — total value", f"{result['b_total']:.0f}",
                   delta=f"{result['gap']:+.0f} vs give")

    if result["uneven_count"]:
        st.caption(
            "Uneven player counts: consolidating several bodies into fewer, "
            "better players carries roster-spot value this total doesn't score "
            "— worth a small edge to the side receiving fewer players."
        )

    bcol = st.columns(2)
    for col, players, tot in (
        (bcol[0], result["a_players"], result["a_total"]),
        (bcol[1], result["b_players"], result["b_total"]),
    ):
        rows = [
            {"Player": p["name"], "Pos": p["position"],
             "Rank": p["rank"] or "—", "Value": round(p["score"], 1)}
            for p in players
        ]
        col.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
