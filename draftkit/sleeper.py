import requests
import streamlit as st


def fetch_sleeper_league(league_id):
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
        "FLEX": 0,
        "SUPER_FLEX": 0
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
        elif pos in ["FLEX", "WRRB_FLEX", "REC_FLEX", "RB_WR", "WR_TE"]:
            parsed["FLEX"] += 1
        elif pos in ["SUPER_FLEX", "OP"]:
            parsed["SUPER_FLEX"] += 1

    return parsed


def parse_sleeper_scoring(scoring_settings):
    if not scoring_settings:
        return {
            "ppr": 0.0,
            "te_premium": 0.0,
            "pass_td": 4.0
        }

    return {
        "ppr": float(scoring_settings.get("rec", 0.0)),
        "te_premium": float(scoring_settings.get("bonus_rec_te", 0.0)),
        "pass_td": float(scoring_settings.get("pass_td", 4.0))
    }


def import_league_settings_from_sleeper(league_id):
    try:
        league_data = fetch_sleeper_league(league_id)
        st.session_state["sleeper_last_league_data"] = league_data

        roster_positions = league_data.get("roster_positions", [])
        scoring_settings = league_data.get("scoring_settings", {})
        total_rosters = int(league_data.get("total_rosters", 12))

        parsed_roster = parse_sleeper_roster_positions(roster_positions)
        parsed_scoring = parse_sleeper_scoring(scoring_settings)

        st.session_state["league_size"] = total_rosters
        st.session_state["roster_settings"] = parsed_roster
        st.session_state["scoring_profile"] = parsed_scoring
        st.session_state["sleeper_raw_roster_positions"] = roster_positions
        st.session_state["sleeper_last_import_status"] = (
            f"Imported league settings from {league_data.get('name', 'Sleeper league')}"
        )

        st.session_state["_sidebar_qb"] = parsed_roster.get("QB", 1)
        st.session_state["_sidebar_rb"] = parsed_roster.get("RB", 2)
        st.session_state["_sidebar_wr"] = parsed_roster.get("WR", 3)
        st.session_state["_sidebar_te"] = parsed_roster.get("TE", 1)
        st.session_state["_sidebar_flex"] = parsed_roster.get("FLEX", 1)
        st.session_state["_sidebar_super_flex"] = parsed_roster.get("SUPER_FLEX", 0)
        st.session_state["_sidebar_ppr"] = parsed_scoring.get("ppr", 0.0)
        st.session_state["_sidebar_te_premium"] = parsed_scoring.get("te_premium", 0.0)
        st.session_state["_sidebar_pass_td"] = parsed_scoring.get("pass_td", 4.0)
        st.session_state["_sidebar_league_size"] = total_rosters

        return True

    except Exception as e:
        st.session_state["sleeper_last_import_status"] = f"Import failed: {e}"
        return False