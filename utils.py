from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "guaranteed_play.db"


@st.cache_resource
def get_connection():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


@st.cache_data
def load_players_df(db_last_modified=None):
    conn = get_connection()
    query = "SELECT * FROM players"
    df = pd.read_sql_query(query, conn)
    return df


def init_session_state():
    defaults = {
        "my_team": [],
        "drafted_players": [],
        "draft_history": [],
        "draft_pick_number": 1,
        "user_draft_slot": 1,
        "roster_settings": {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "FLEX": 1
        },
        "scoring_profile": {
            "ppr": 0.0,
            "te_premium": 0.0
        },
        "sleeper_league_id": "",
        "sleeper_last_import_status": "",
        "sleeper_raw_roster_positions": [],
        "sleeper_last_league_data": {},
        "last_action_message": "",
        "_sidebar_qb": 1,
        "_sidebar_rb": 2,
        "_sidebar_wr": 3,
        "_sidebar_te": 1,
        "_sidebar_flex": 1
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def keep_session_state_alive():
    pass


def fetch_sleeper_league(league_id: str):
    url = f"https://api.sleeper.app/v1/league/{league_id}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def parse_sleeper_roster_positions(roster_positions):
    parsed = {
        "QB": 0,
        "RB": 0,
        "WR": 0,
        "TE": 0,
        "FLEX": 0
    }

    for pos in roster_positions:
        if pos == "QB":
            parsed["QB"] += 1
        elif pos == "RB":
            parsed["RB"] += 1
        elif pos == "WR":
            parsed["WR"] += 1
        elif pos == "TE":
            parsed["TE"] += 1
        elif pos in ["FLEX", "WRRB_FLEX", "REC_FLEX", "RB_WR", "WR_TE", "SUPER_FLEX", "OP"]:
            parsed["FLEX"] += 1

    return parsed


def parse_sleeper_scoring_settings(scoring_settings):
    if not scoring_settings:
        return {"ppr": 0.0, "te_premium": 0.0}

    rec = float(scoring_settings.get("rec", 0) or 0)
    te_rec = float(scoring_settings.get("bonus_rec_te", 0) or 0)

    return {
        "ppr": rec,
        "te_premium": te_rec
    }


def import_league_settings_from_sleeper():
    league_id = st.session_state.get("sleeper_league_id", "").strip()

    if not league_id:
        st.session_state["sleeper_last_import_status"] = "Enter a Sleeper league ID first."
        return

    try:
        league_data = fetch_sleeper_league(league_id)
        st.session_state["sleeper_last_league_data"] = league_data

        scoring_settings = league_data.get("scoring_settings", {})
        st.session_state["scoring_profile"] = parse_sleeper_scoring_settings(scoring_settings)

        roster_positions = league_data.get("roster_positions", [])
        parsed = parse_sleeper_roster_positions(roster_positions)

        st.session_state["roster_settings"] = parsed
        st.session_state["sleeper_raw_roster_positions"] = roster_positions

        st.session_state["_sidebar_qb"] = parsed["QB"]
        st.session_state["_sidebar_rb"] = parsed["RB"]
        st.session_state["_sidebar_wr"] = parsed["WR"]
        st.session_state["_sidebar_te"] = parsed["TE"]
        st.session_state["_sidebar_flex"] = parsed["FLEX"]

        total_rosters = league_data.get("total_rosters")
        if total_rosters:
            try:
                draft_slot = st.session_state.get("user_draft_slot", 1)
                draft_slot = max(1, min(int(draft_slot), int(total_rosters)))
                st.session_state["user_draft_slot"] = draft_slot
            except (TypeError, ValueError):
                pass

        st.session_state["sleeper_last_import_status"] = (
            f"Imported from Sleeper: "
            f"{parsed['QB']} QB, "
            f"{parsed['RB']} RB, "
            f"{parsed['WR']} WR, "
            f"{parsed['TE']} TE, "
            f"{parsed['FLEX']} FLEX"
        )

    except Exception as e:
        st.session_state["sleeper_last_import_status"] = f"Sleeper import failed: {e}"


def render_league_settings_sidebar():
    st.sidebar.markdown("## League Settings")

    current = st.session_state.get(
        "roster_settings",
        {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "FLEX": 1
        }
    )

    st.session_state["_sidebar_qb"] = current.get("QB", 1)
    st.session_state["_sidebar_rb"] = current.get("RB", 2)
    st.session_state["_sidebar_wr"] = current.get("WR", 3)
    st.session_state["_sidebar_te"] = current.get("TE", 1)
    st.session_state["_sidebar_flex"] = current.get("FLEX", 1)

    def save_roster_settings():
        st.session_state["roster_settings"] = {
            "QB": st.session_state["_sidebar_qb"],
            "RB": st.session_state["_sidebar_rb"],
            "WR": st.session_state["_sidebar_wr"],
            "TE": st.session_state["_sidebar_te"],
            "FLEX": st.session_state["_sidebar_flex"]
        }

    st.sidebar.text_input("Sleeper League ID", key="sleeper_league_id")

    st.sidebar.button(
        "Import from Sleeper",
        on_click=import_league_settings_from_sleeper,
        use_container_width=True
    )

    if st.session_state.get("sleeper_last_import_status"):
        st.sidebar.caption(st.session_state["sleeper_last_import_status"])

    st.sidebar.number_input(
        "Your draft slot",
        min_value=1,
        max_value=20,
        step=1,
        key="user_draft_slot"
    )

    st.sidebar.number_input("Starting QB", min_value=0, max_value=3, step=1, key="_sidebar_qb", on_change=save_roster_settings)
    st.sidebar.number_input("Starting RB", min_value=0, max_value=5, step=1, key="_sidebar_rb", on_change=save_roster_settings)
    st.sidebar.number_input("Starting WR", min_value=0, max_value=5, step=1, key="_sidebar_wr", on_change=save_roster_settings)
    st.sidebar.number_input("Starting TE", min_value=0, max_value=3, step=1, key="_sidebar_te", on_change=save_roster_settings)
    st.sidebar.number_input("FLEX spots", min_value=0, max_value=4, step=1, key="_sidebar_flex", on_change=save_roster_settings)

    st.sidebar.caption(
        f"Current format: "
        f"{st.session_state['roster_settings']['QB']} QB, "
        f"{st.session_state['roster_settings']['RB']} RB, "
        f"{st.session_state['roster_settings']['WR']} WR, "
        f"{st.session_state['roster_settings']['TE']} TE, "
        f"{st.session_state['roster_settings']['FLEX']} FLEX"
    )

    profile = st.session_state.get("scoring_profile", {"ppr": 0.0, "te_premium": 0.0})
    st.sidebar.caption(
        f"Scoring: PPR={profile['ppr']}, TE premium={profile['te_premium']}"
    )

    with st.sidebar.expander("Sleeper import details", expanded=False):
        st.write("Raw roster positions:")
        st.json(st.session_state.get("sleeper_raw_roster_positions", []), expanded=True)

        st.write("Mapped roster settings:")
        st.json(st.session_state.get("roster_settings", {}), expanded=True)

        league_data = st.session_state.get("sleeper_last_league_data", {})
        if league_data:
            st.write("Selected league fields:")
            st.json(
                {
                    "name": league_data.get("name"),
                    "season": league_data.get("season"),
                    "total_rosters": league_data.get("total_rosters"),
                    "roster_positions": league_data.get("roster_positions"),
                    "scoring_settings": league_data.get("scoring_settings", {})
                },
                expanded=False
            )


def safe_col(df, candidates, default=None):
    for col in candidates:
        if col in df.columns:
            return col
    return default


def _get_player_record_value(player, field):
    if isinstance(player, dict):
        return player.get(field)

    player_name = str(player)
    df = load_players_df()

    if df.empty:
        return None

    player_col = safe_col(df, ["player_name", "Player", "player", "name", "full_name"])
    if player_col is None:
        return None

    match = df[df[player_col].astype(str) == player_name]
    if match.empty:
        return None

    row = match.iloc[0]

    if field == "player":
        return str(row[player_col])

    if field == "pos":
        pos_col = safe_col(df, ["position", "pos", "Pos", "Position", "POS"])
        return row[pos_col] if pos_col else None

    if field == "projection":
        proj_col = safe_col(df, ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"])
        return row[proj_col] if proj_col else None

    return None


def get_roster_counts():
    counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}

    for player in st.session_state.get("my_team", []):
        pos = _get_player_record_value(player, "pos")
        pos = str(pos).upper() if pos is not None else None
        if pos in counts:
            counts[pos] += 1

    return counts


def get_roster_targets():
    return st.session_state.get(
        "roster_settings",
        {
            "QB": 1,
            "RB": 2,
            "WR": 3,
            "TE": 1,
            "FLEX": 1
        }
    )


def normalize_series(series):
    s = pd.to_numeric(series, errors="coerce")
    s = pd.Series(s)

    if s.isna().all():
        return pd.Series([0.0] * len(s), index=s.index)

    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series([0.0] * len(s), index=s.index)

    return (s - min_val) / (max_val - min_val)


def get_position_need_weights():
    counts = get_roster_counts()
    targets = get_roster_targets()
    profile = st.session_state.get("scoring_profile", {"ppr": 0.0, "te_premium": 0.0})

    need_weights = {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0}

    for pos in need_weights.keys():
        base_target = targets.get(pos, 0)
        drafted = counts.get(pos, 0)
        shortage = max(base_target - drafted, 0)

        if shortage >= 2:
            need_weights[pos] = 1.00
        elif shortage == 1:
            need_weights[pos] = 0.65
        else:
            need_weights[pos] = 0.15

    flex_slots = targets.get("FLEX", 0)
    if flex_slots > 0:
        rb_surplus_gap = max((targets.get("RB", 0) + flex_slots) - counts.get("RB", 0), 0)
        wr_surplus_gap = max((targets.get("WR", 0) + flex_slots) - counts.get("WR", 0), 0)

        if rb_surplus_gap > 0:
            need_weights["RB"] += 0.20
        if wr_surplus_gap > 0:
            need_weights["WR"] += 0.20

    if profile.get("ppr", 0) >= 0.5:
        need_weights["WR"] += 0.10

    if profile.get("ppr", 0) >= 1.0:
        need_weights["WR"] += 0.05
        need_weights["RB"] += 0.05

    if profile.get("te_premium", 0) > 0:
        need_weights["TE"] += 0.15

    return need_weights


def get_available_players_df():
    df = load_players_df(DB_PATH.stat().st_mtime).copy()

    player_col = safe_col(df, ["player_name", "Player", "player"])
    if player_col is None:
        return df

    drafted = set(st.session_state.get("drafted_players", []))
    if drafted:
        df = df[~df[player_col].astype(str).isin([str(name) for name in drafted])]

    return df.reset_index(drop=True)


def add_best_available_score(df):
    working_df = df.copy()

    proj_col = safe_col(working_df, ["projection_points", "Projection", "projected_points", "FPTS"])
    if proj_col is None:
        working_df["best_available_score"] = 0.0
        return working_df

    working_df["best_available_score"] = normalize_series(working_df[proj_col])
    return working_df.sort_values("best_available_score", ascending=False)


def add_best_value_score(df):
    working_df = df.copy()

    proj_col = safe_col(working_df, ["projection_points", "Projection", "projected_points", "FPTS"])
    adp_col = safe_col(working_df, ["adp", "ADP"])

    if proj_col is None:
        working_df["best_value_score"] = 0.0
        return working_df

    projection_score = normalize_series(working_df[proj_col])

    if adp_col is not None:
        adp_discount = 1 - normalize_series(working_df[adp_col])
    else:
        adp_discount = pd.Series([0.0] * len(working_df), index=working_df.index)

    working_df["best_value_score"] = 0.65 * projection_score + 0.35 * adp_discount
    return working_df.sort_values("best_value_score", ascending=False)


def add_best_fit_score(df):
    working_df = df.copy()

    proj_col = safe_col(working_df, ["projection_points", "Projection", "projected_points", "FPTS"])
    pos_col = safe_col(working_df, ["position", "pos", "Pos", "Position"])
    adp_col = safe_col(working_df, ["adp", "ADP"])

    if proj_col is None:
        working_df["best_fit_score"] = 0.0
        return working_df

    projection_score = normalize_series(working_df[proj_col])

    if adp_col is not None:
        adp_score = 1 - normalize_series(working_df[adp_col])
    else:
        adp_score = pd.Series([0.0] * len(working_df), index=working_df.index)

    need_weights = get_position_need_weights()

    if pos_col is not None:
        pos_series = working_df[pos_col].astype(str)
        fit_score = pos_series.map(lambda x: need_weights.get(x, 0.0)).fillna(0.0)
    else:
        fit_score = pd.Series([0.0] * len(working_df), index=working_df.index)

    working_df["best_fit_score"] = (
        0.55 * projection_score
        + 0.20 * adp_score
        + 0.25 * fit_score
    )

    return working_df.sort_values("best_fit_score", ascending=False)


def draft_player(player_name):
    df = get_available_players_df()

    player_col = safe_col(df, ["player_name", "Player", "player"])
    pos_col = safe_col(df, ["position", "pos", "Pos", "Position"])
    team_col = safe_col(df, ["team", "Team"])
    adp_col = safe_col(df, ["adp", "ADP"])
    proj_col = safe_col(df, ["projection_points", "Projection", "projected_points", "FPTS"])

    if player_col is None:
        return False

    selected = df[df[player_col].astype(str) == str(player_name)]
    if selected.empty:
        return False

    row = selected.iloc[0]
    current_pick = st.session_state["draft_pick_number"]

    record = {
        "player": str(row[player_col]),
        "pos": str(row[pos_col]) if pos_col else "Unknown",
        "team": str(row[team_col]) if team_col else "Unknown",
        "adp": row[adp_col] if adp_col else None,
        "projection": row[proj_col] if proj_col else None
    }

    history_record = {
        "pick_number": current_pick,
        "player": record["player"],
        "pos": record["pos"],
        "team": record["team"],
        "adp": record["adp"],
        "projection": record["projection"]
    }

    st.session_state["my_team"].append(record)
    st.session_state["draft_history"].append(history_record)

    if str(player_name) not in st.session_state["drafted_players"]:
        st.session_state["drafted_players"].append(str(player_name))

    st.session_state["draft_pick_number"] += 1
    return True


def undo_last_pick():
    if len(st.session_state["my_team"]) == 0:
        return False

    last_player = st.session_state["my_team"].pop()
    last_name = str(_get_player_record_value(last_player, "player") or last_player)

    if st.session_state.get("draft_history"):
        st.session_state["draft_history"].pop()

    if last_name in st.session_state["drafted_players"]:
        st.session_state["drafted_players"].remove(last_name)

    st.session_state["draft_pick_number"] = max(1, st.session_state["draft_pick_number"] - 1)
    return True


def remove_player_by_name(player_name):
    player_name = str(player_name)
    my_team = st.session_state["my_team"]

    removed_player = None
    updated_team = []

    for p in my_team:
        record_name = str(_get_player_record_value(p, "player") or p)
        if removed_player is None and record_name == player_name:
            removed_player = p
        else:
            updated_team.append(p)

    removed_any = removed_player is not None

    if removed_any:
        st.session_state["my_team"] = updated_team

        if player_name in st.session_state["drafted_players"]:
            st.session_state["drafted_players"].remove(player_name)

        history = st.session_state.get("draft_history", [])
        updated_history = [h for h in history if str(h.get("player")) != player_name]
        st.session_state["draft_history"] = updated_history

        st.session_state["draft_pick_number"] = max(1, len(st.session_state["my_team"]) + 1)

    return removed_any


def get_recommended_next_position():
    need_weights = get_position_need_weights()

    sorted_positions = sorted(
        need_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_pos, top_weight = sorted_positions[0]

    if top_weight <= 0.20:
        return "Best value / depth"

    return top_pos


def get_total_team_projection():
    total = 0.0

    for player in st.session_state.get("my_team", []):
        projection = _get_player_record_value(player, "projection")
        if projection is not None:
            try:
                total += float(projection)
            except (TypeError, ValueError):
                pass

    return round(total, 1)


def build_player_comparison_df(player_names):
    df = get_available_players_df()

    player_col = safe_col(df, ["player_name", "Player", "player"])
    pos_col = safe_col(df, ["position", "pos", "Pos", "Position"])
    team_col = safe_col(df, ["team", "Team"])
    adp_col = safe_col(df, ["adp", "ADP"])
    proj_col = safe_col(df, ["projection_points", "Projection", "projected_points", "FPTS"])

    if player_col is None or not player_names:
        return pd.DataFrame()

    player_names = [str(name) for name in player_names]
    compare_df = df[df[player_col].astype(str).isin(player_names)].copy()

    fit_df = add_best_fit_score(compare_df)
    value_df = add_best_value_score(compare_df)
    available_df = add_best_available_score(compare_df)

    fit_scores = {}
    value_scores = {}
    available_scores = {}

    if player_col in fit_df.columns and "best_fit_score" in fit_df.columns:
        fit_scores = {
            str(name): float(score)
            for name, score in zip(
                fit_df[player_col].tolist(),
                fit_df["best_fit_score"].tolist()
            )
        }

    if player_col in value_df.columns and "best_value_score" in value_df.columns:
        value_scores = {
            str(name): float(score)
            for name, score in zip(
                value_df[player_col].tolist(),
                value_df["best_value_score"].tolist()
            )
        }

    if player_col in available_df.columns and "best_available_score" in available_df.columns:
        available_scores = {
            str(name): float(score)
            for name, score in zip(
                available_df[player_col].tolist(),
                available_df["best_available_score"].tolist()
            )
        }

    rows = []
    for _, row in compare_df.iterrows():
        player_name = str(row[player_col])

        rows.append({
            "Player": player_name,
            "Position": str(row[pos_col]) if pos_col else "Unknown",
            "Team": str(row[team_col]) if team_col else "Unknown",
            "ADP": row[adp_col] if adp_col else None,
            "Projection": row[proj_col] if proj_col else None,
            "Best Fit Score": round(fit_scores.get(player_name, 0.0), 3),
            "Best Value Score": round(value_scores.get(player_name, 0.0), 3),
            "Best Available Score": round(available_scores.get(player_name, 0.0), 3),
        })

    return pd.DataFrame(rows).sort_values("Best Fit Score", ascending=False).reset_index(drop=True)


def build_roster_needs_df():
    counts = get_roster_counts()
    targets = get_roster_targets()

    rows = []
    for pos in ["QB", "RB", "WR", "TE"]:
        drafted = counts.get(pos, 0)
        target = targets.get(pos, 0)
        shortage = max(target - drafted, 0)

        if shortage >= 2:
            need_level = "High"
        elif shortage == 1:
            need_level = "Medium"
        else:
            need_level = "Low"

        rows.append(
            {
                "Position": pos,
                "Drafted": drafted,
                "Target": target,
                "Shortage": shortage,
                "Need": need_level
            }
        )

    return pd.DataFrame(rows)


def build_live_rankings_df(position_filter="All"):
    available_df = get_available_players_df()

    player_col = safe_col(available_df, ["player_name", "Player", "player"])
    pos_col = safe_col(available_df, ["position", "pos", "Pos", "Position"])
    team_col = safe_col(available_df, ["team", "Team"])
    adp_col = safe_col(available_df, ["adp", "ADP"])
    proj_col = safe_col(available_df, ["projection_points", "Projection", "projected_points", "FPTS"])

    if available_df.empty or player_col is None:
        return pd.DataFrame()

    working_df = available_df.copy()

    if pos_col is not None and position_filter != "All":
        working_df = working_df[working_df[pos_col].astype(str) == str(position_filter)].copy()

    fit_df = add_best_fit_score(working_df)
    value_df = add_best_value_score(working_df)
    available_scored_df = add_best_available_score(working_df)

    fit_scores = {}
    value_scores = {}
    available_scores = {}

    if player_col in fit_df.columns and "best_fit_score" in fit_df.columns:
        fit_scores = {
            str(name): float(score)
            for name, score in zip(
                fit_df[player_col].tolist(),
                fit_df["best_fit_score"].tolist()
            )
        }

    if player_col in value_df.columns and "best_value_score" in value_df.columns:
        value_scores = {
            str(name): float(score)
            for name, score in zip(
                value_df[player_col].tolist(),
                value_df["best_value_score"].tolist()
            )
        }

    if player_col in available_scored_df.columns and "best_available_score" in available_scored_df.columns:
        available_scores = {
            str(name): float(score)
            for name, score in zip(
                available_scored_df[player_col].tolist(),
                available_scored_df["best_available_score"].tolist()
            )
        }

    rows = []
    for _, row in working_df.iterrows():
        player_name = str(row[player_col])

        rows.append({
            "Player": player_name,
            "Position": str(row[pos_col]) if pos_col else "Unknown",
            "Team": str(row[team_col]) if team_col else "Unknown",
            "ADP": row[adp_col] if adp_col else None,
            "Projection": row[proj_col] if proj_col else None,
            "Best Fit Score": round(fit_scores.get(player_name, 0.0), 3),
            "Best Value Score": round(value_scores.get(player_name, 0.0), 3),
            "Best Available Score": round(available_scores.get(player_name, 0.0), 3),
        })

    rankings_df = pd.DataFrame(rows)

    if rankings_df.empty:
        return rankings_df

    return rankings_df.sort_values(
        ["Best Fit Score", "Best Value Score", "Best Available Score"],
        ascending=False
    ).reset_index(drop=True)


def build_position_scarcity_df(top_n=12):
    available_df = get_available_players_df()

    player_col = safe_col(available_df, ["player_name", "Player", "player"])
    pos_col = safe_col(available_df, ["position", "pos", "Pos", "Position"])
    proj_col = safe_col(available_df, ["projection_points", "Projection", "projected_points", "FPTS"])
    adp_col = safe_col(available_df, ["adp", "ADP"])

    if available_df.empty or player_col is None or pos_col is None or proj_col is None:
        return pd.DataFrame()

    rows = []

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = available_df[available_df[pos_col].astype(str) == pos].copy()

        if pos_df.empty:
            rows.append({
                "Position": pos,
                "Players Considered": 0,
                "Top Projection": 0,
                "Projection at Cutoff": 0,
                "Dropoff": 0,
                "Avg Top ADP": None,
                "Scarcity Score": 0
            })
            continue

        pos_df[proj_col] = pd.to_numeric(pos_df[proj_col], errors="coerce")
        pos_df = pos_df.dropna(subset=[proj_col]).sort_values(proj_col, ascending=False).reset_index(drop=True)
        top_slice = pos_df.head(top_n).copy()

        top_projection = float(top_slice.iloc[0][proj_col]) if not top_slice.empty else 0.0
        cutoff_projection = float(top_slice.iloc[-1][proj_col]) if not top_slice.empty else 0.0
        dropoff = top_projection - cutoff_projection

        if adp_col is not None and adp_col in top_slice.columns:
            avg_top_adp = pd.to_numeric(top_slice[adp_col], errors="coerce").mean()
            avg_top_adp = round(float(avg_top_adp), 2) if pd.notna(avg_top_adp) else None
        else:
            avg_top_adp = None

        scarcity_score = round(dropoff, 2)

        rows.append({
            "Position": pos,
            "Players Considered": len(top_slice),
            "Top Projection": round(top_projection, 2),
            "Projection at Cutoff": round(cutoff_projection, 2),
            "Dropoff": round(dropoff, 2),
            "Avg Top ADP": avg_top_adp,
            "Scarcity Score": scarcity_score
        })

    scarcity_df = pd.DataFrame(rows)

    if scarcity_df.empty:
        return scarcity_df

    return scarcity_df.sort_values("Scarcity Score", ascending=False).reset_index(drop=True)


def build_position_cliff_df(window_size=3):
    available_df = get_available_players_df()

    player_col = safe_col(available_df, ["player_name", "Player", "player"])
    pos_col = safe_col(available_df, ["position", "pos", "Pos", "Position"])
    proj_col = safe_col(available_df, ["projection_points", "Projection", "projected_points", "FPTS"])
    adp_col = safe_col(available_df, ["adp", "ADP"])

    if available_df.empty or player_col is None or pos_col is None or proj_col is None:
        return pd.DataFrame()

    roster_targets = get_roster_targets()
    roster_counts = get_roster_counts()

    rows = []

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = available_df[available_df[pos_col].astype(str) == pos].copy()

        if pos_df.empty:
            rows.append({
                "Position": pos,
                "Current Best": None,
                "Current Projection": 0.0,
                "Projection After Wait": 0.0,
                "Drop If Wait": 0.0,
                "Cliff Level": "Unknown",
                "Urgency Score": 0.0,
                "Need Weight": 0.0,
                "Draft Signal": "No data"
            })
            continue

        pos_df[proj_col] = pd.to_numeric(pos_df[proj_col], errors="coerce")
        pos_df = pos_df.dropna(subset=[proj_col]).sort_values(proj_col, ascending=False).reset_index(drop=True)

        if pos_df.empty:
            continue

        best_row = pos_df.iloc[0]
        current_best_name = str(best_row[player_col])
        current_projection = float(best_row[proj_col])

        wait_index = min(window_size, len(pos_df) - 1)
        after_wait_projection = float(pos_df.iloc[wait_index][proj_col]) if len(pos_df) > 1 else current_projection
        drop_if_wait = current_projection - after_wait_projection

        target = roster_targets.get(pos, 0)
        drafted = roster_counts.get(pos, 0)
        shortage = max(target - drafted, 0)

        if shortage >= 2:
            need_weight = 1.0
        elif shortage == 1:
            need_weight = 0.7
        else:
            need_weight = 0.2

        urgency_score = round(drop_if_wait * need_weight, 2)

        if drop_if_wait >= 25:
            cliff_level = "Immediate"
        elif drop_if_wait >= 15:
            cliff_level = "Near"
        elif drop_if_wait >= 8:
            cliff_level = "Soon"
        else:
            cliff_level = "Stable"

        if cliff_level == "Immediate" and need_weight >= 0.7:
            draft_signal = f"Draft {pos} now"
        elif cliff_level in ["Immediate", "Near"] and need_weight >= 0.2:
            draft_signal = f"Strongly consider {pos}"
        elif cliff_level == "Soon":
            draft_signal = f"Monitor {pos} closely"
        else:
            draft_signal = f"You can likely wait on {pos}"

        avg_top_adp = None
        if adp_col is not None and adp_col in pos_df.columns:
            sample = pos_df.head(min(5, len(pos_df))).copy()
            sample[adp_col] = pd.to_numeric(sample[adp_col], errors="coerce")
            avg_top_adp = sample[adp_col].mean()
            avg_top_adp = round(float(avg_top_adp), 2) if pd.notna(avg_top_adp) else None

        rows.append({
            "Position": pos,
            "Current Best": current_best_name,
            "Current Projection": round(current_projection, 2),
            "Projection After Wait": round(after_wait_projection, 2),
            "Drop If Wait": round(drop_if_wait, 2),
            "Cliff Level": cliff_level,
            "Urgency Score": urgency_score,
            "Need Weight": round(need_weight, 2),
            "Avg Top ADP": avg_top_adp,
            "Draft Signal": draft_signal
        })

    cliff_df = pd.DataFrame(rows)

    if cliff_df.empty:
        return cliff_df

    return cliff_df.sort_values(
        ["Urgency Score", "Drop If Wait"],
        ascending=False
    ).reset_index(drop=True)


def get_falloff_recommendation(window_size=3):
    cliff_df = build_position_cliff_df(window_size=window_size)

    if cliff_df.empty:
        return {
            "headline": "No fall-off signal available",
            "message": "Not enough player data to estimate a draft cliff yet."
        }

    top_row = cliff_df.iloc[0]

    position = top_row["Position"]
    signal = top_row["Draft Signal"]
    drop = top_row["Drop If Wait"]
    cliff = top_row["Cliff Level"]
    player = top_row["Current Best"]

    headline = f"Fall-off watch: {position}"
    message = (
        f"{signal}. Top remaining {position} is {player}, and waiting about "
        f"{window_size} picks could cost roughly {drop} projected points. "
        f"Current cliff level: {cliff}."
    )

    return {
        "headline": headline,
        "message": message
    }


def get_league_size():
    league_data = st.session_state.get("sleeper_last_league_data", {})
    total_rosters = league_data.get("total_rosters")

    try:
        return int(total_rosters) if total_rosters else 12
    except (TypeError, ValueError):
        return 12


def get_user_draft_slot():
    slot = st.session_state.get("user_draft_slot", 1)
    try:
        return int(slot)
    except (TypeError, ValueError):
        return 1


def get_round_and_pick_in_round(overall_pick, league_size):
    round_number = ((overall_pick - 1) // league_size) + 1
    pick_in_round = ((overall_pick - 1) % league_size) + 1
    return round_number, pick_in_round


def get_next_pick_distance():
    overall_pick = int(st.session_state.get("draft_pick_number", 1))
    league_size = get_league_size()
    draft_slot = get_user_draft_slot()

    if league_size <= 1:
        return 1

    round_number, _ = get_round_and_pick_in_round(overall_pick, league_size)

    if round_number % 2 == 1:
        current_slot_on_clock = ((overall_pick - 1) % league_size) + 1
    else:
        current_slot_on_clock = league_size - (((overall_pick - 1) % league_size))

    if current_slot_on_clock != draft_slot:
        return 1

    next_pick_distance = (2 * league_size) - 1
    return next_pick_distance


def build_turn_aware_cliff_df():
    wait_picks = get_next_pick_distance()
    cliff_df = build_position_cliff_df(window_size=wait_picks).copy()

    if cliff_df.empty:
        return cliff_df

    cliff_df["Wait Picks"] = wait_picks
    return cliff_df

def build_position_tiers_df(drop_threshold=12.0):
    available_df = get_available_players_df()

    player_col = safe_col(available_df, ["player_name", "Player", "player"])
    pos_col = safe_col(available_df, ["position", "pos", "Pos", "Position"])
    proj_col = safe_col(available_df, ["projection_points", "Projection", "projected_points", "FPTS"])
    adp_col = safe_col(available_df, ["adp", "ADP"])
    team_col = safe_col(available_df, ["team", "Team"])

    if available_df.empty or player_col is None or pos_col is None or proj_col is None:
        return pd.DataFrame()

    all_rows = []

    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = available_df[available_df[pos_col].astype(str) == pos].copy()

        if pos_df.empty:
            continue

        pos_df[proj_col] = pd.to_numeric(pos_df[proj_col], errors="coerce")
        pos_df = pos_df.dropna(subset=[proj_col]).sort_values(proj_col, ascending=False).reset_index(drop=True)

        if pos_df.empty:
            continue

        tier_number = 1
        previous_projection = None

        for idx, row in pos_df.iterrows():
            current_projection = float(row[proj_col])

            if previous_projection is not None:
                gap = previous_projection - current_projection
                if gap >= float(drop_threshold):
                    tier_number += 1

            all_rows.append({
                "Position": pos,
                "Tier": tier_number,
                "Player": str(row[player_col]),
                "Team": str(row[team_col]) if team_col else "Unknown",
                "Projection": round(current_projection, 2),
                "ADP": row[adp_col] if adp_col else None,
                "Tier Rank": idx + 1
            })

            previous_projection = current_projection

    tiers_df = pd.DataFrame(all_rows)

    if tiers_df.empty:
        return tiers_df

    return tiers_df.reset_index(drop=True)


def build_tier_summary_df(drop_threshold=12.0):
    tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)

    if tiers_df.empty:
        return pd.DataFrame()

    summary_rows = []

    grouped = tiers_df.groupby(["Position", "Tier"], dropna=False)

    for (position, tier), group in grouped:
        group = group.reset_index(drop=True)
        players_left = len(group)
        best_player = group.iloc[0]["Player"]
        worst_player = group.iloc[-1]["Player"]
        best_projection = float(group.iloc[0]["Projection"])
        worst_projection = float(group.iloc[-1]["Projection"])

        if players_left <= 2:
            tier_status = "Thin"
        elif players_left <= 4:
            tier_status = "Shrinking"
        else:
            tier_status = "Healthy"

        summary_rows.append({
            "Position": position,
            "Tier": tier,
            "Players Left In Tier": players_left,
            "Tier Status": tier_status,
            "Best Player": best_player,
            "Last Player In Tier": worst_player,
            "Top Projection": round(best_projection, 2),
            "Bottom Projection": round(worst_projection, 2)
        })

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        return summary_df

    summary_df = summary_df.sort_values(
        ["Position", "Tier"],
        ascending=[True, True]
    ).reset_index(drop=True)

    return summary_df


def get_tier_warning(drop_threshold=12.0):
    summary_df = build_tier_summary_df(drop_threshold=drop_threshold)

    if summary_df.empty:
        return {
            "headline": "No tier warning available",
            "message": "Not enough player data to build tiers."
        }

    thin_df = summary_df[summary_df["Tier Status"].isin(["Thin", "Shrinking"])].copy()

    if thin_df.empty:
        return {
            "headline": "Tier board looks stable",
            "message": "No position currently has an obviously thin top tier."
        }

    priority_map = {"Thin": 2, "Shrinking": 1}
    thin_df["priority"] = thin_df["Tier Status"].map(priority_map)

    top_row = thin_df.sort_values(
        ["priority", "Tier"],
        ascending=[False, True]
    ).iloc[0]

    position = top_row["Position"]
    tier = top_row["Tier"]
    players_left = top_row["Players Left In Tier"]
    best_player = top_row["Best Player"]
    worst_player = top_row["Last Player In Tier"]

    headline = f"Tier watch: {position}"
    message = (
        f"Tier {tier} at {position} is {top_row['Tier Status'].lower()} with "
        f"{players_left} player(s) left. The tier currently runs from {best_player} "
        f"to {worst_player}, so you may want to draft from this group before the next drop."
    )

    return {
        "headline": headline,
        "message": message
    }
def build_desperation_targets_df(drop_threshold=12.0):
    tiers_df = build_position_tiers_df(drop_threshold=drop_threshold)
    tier_summary_df = build_tier_summary_df(drop_threshold=drop_threshold)
    cliff_df = build_turn_aware_cliff_df()

    if tiers_df.empty or tier_summary_df.empty:
        return pd.DataFrame()

    roster_needs = get_position_need_weights()

    cliff_lookup = {}
    if not cliff_df.empty:
        for _, row in cliff_df.iterrows():
            cliff_lookup[str(row["Position"])] = {
                "Drop If Wait": float(row["Drop If Wait"]),
                "Urgency Score": float(row["Urgency Score"]),
                "Cliff Level": str(row["Cliff Level"]),
                "Draft Signal": str(row["Draft Signal"]),
            }

    rows = []

    priority_summary = tier_summary_df[
        tier_summary_df["Tier Status"].isin(["Thin", "Shrinking"])
    ].copy()

    for _, summary_row in priority_summary.iterrows():
        position = str(summary_row["Position"])
        tier = int(summary_row["Tier"])
        tier_status = str(summary_row["Tier Status"])
        players_left = int(summary_row["Players Left In Tier"])

        tier_players = tiers_df[
            (tiers_df["Position"] == position) &
            (tiers_df["Tier"] == tier)
        ].copy().reset_index(drop=True)

        if tier_players.empty:
            continue

        target_row = tier_players.iloc[-1]
        player_name = str(target_row["Player"])
        team = str(target_row["Team"])
        projection = float(target_row["Projection"])
        adp = target_row["ADP"]

        next_tier_players = tiers_df[
            (tiers_df["Position"] == position) &
            (tiers_df["Tier"] == tier + 1)
        ].copy().reset_index(drop=True)

        next_tier_player = None
        next_tier_projection = None
        tier_drop = 0.0

        if not next_tier_players.empty:
            next_tier_player = str(next_tier_players.iloc[0]["Player"])
            next_tier_projection = float(next_tier_players.iloc[0]["Projection"])
            tier_drop = round(projection - next_tier_projection, 2)

        need_weight = float(roster_needs.get(position, 0.0))

        if tier_status == "Thin":
            thinness_score = 1.0
        else:
            thinness_score = 0.7

        if players_left <= 1:
            thinness_score += 0.25
        elif players_left == 2:
            thinness_score += 0.10

        cliff_info = cliff_lookup.get(position, {})
        drop_if_wait = float(cliff_info.get("Drop If Wait", 0.0))
        urgency_score = float(cliff_info.get("Urgency Score", 0.0))
        cliff_level = str(cliff_info.get("Cliff Level", "Unknown"))
        draft_signal = str(cliff_info.get("Draft Signal", "Monitor"))

        desperation_score = round(
            (thinness_score * 35)
            + (need_weight * 25)
            + min(tier_drop, 25.0)
            + min(drop_if_wait, 25.0),
            2
        )

        rows.append({
            "Position": position,
            "Tier": tier,
            "Tier Status": tier_status,
            "Players Left In Tier": players_left,
            "Player": player_name,
            "Team": team,
            "Projection": projection,
            "ADP": adp,
            "Next Tier Player": next_tier_player,
            "Next Tier Projection": next_tier_projection,
            "Tier Drop": tier_drop,
            "Need Weight": round(need_weight, 2),
            "Drop If Wait": round(drop_if_wait, 2),
            "Urgency Score": round(urgency_score, 2),
            "Cliff Level": cliff_level,
            "Draft Signal": draft_signal,
            "Desperation Score": desperation_score
        })

    desperation_df = pd.DataFrame(rows)

    if desperation_df.empty:
        return desperation_df

    return desperation_df.sort_values(
        ["Desperation Score", "Tier Drop", "Drop If Wait"],
        ascending=False
    ).reset_index(drop=True)

def get_best_desperation_target(drop_threshold=12.0):
    desperation_df = build_desperation_targets_df(drop_threshold=drop_threshold)

    if desperation_df.empty:
        return None

    row = desperation_df.iloc[0].to_dict()
    return row
def handle_draft_action(player_name):
    drafted = draft_player(player_name)
    if drafted:
        st.session_state["last_action_message"] = f"Drafted {player_name}"



def get_turn_aware_falloff_recommendation():
    wait_picks = get_next_pick_distance()
    cliff_df = build_position_cliff_df(window_size=wait_picks)

    if cliff_df.empty:
        return {
            "headline": "No turn-based fall-off signal available",
            "message": "Not enough player data to estimate the drop before your next pick."
        }

    top_row = cliff_df.iloc[0]

    position = top_row["Position"]
    signal = top_row["Draft Signal"]
    drop = top_row["Drop If Wait"]
    cliff = top_row["Cliff Level"]
    player = top_row["Current Best"]

    headline = f"Before your next pick: {position}"
    message = (
        f"{signal}. If you wait until your next turn, about {wait_picks} picks away, "
        f"the best remaining {position} could fall from {player} by roughly {drop} projected points. "
        f"Current cliff level: {cliff}."
    )

    return {
        "headline": headline,
        "message": message
    }
