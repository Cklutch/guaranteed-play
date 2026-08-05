import pandas as pd
import streamlit as st

from draftkit.championship_equity import build_championship_equity_df
from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _normalize_name
from draftkit.common import safe_float as _safe_float
from draftkit.construction_pressure import calculate_construction_pressure
from draftkit.data_access import load_players_df, safe_col
from draftkit.draft_analysis import build_master_recommendations_df
from draftkit.draft_state import get_current_pick_number, get_my_team_positions, init_session_state
from draftkit.signal_trust import build_signal_trust_report


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
ADP_COLS = ["adp", "ADP", "consensus_adp"]
INJURY_RISK_COLS = ["injury_risk", "Injury Risk", "injury_score", "risk_score"]
BOOM_COLS = ["boom_score", "Boom Score"]
BUST_COLS = ["bust_score", "Bust Score"]

COMPONENT_WEIGHTS = {
    "pick_value_score": 0.30,
    "recommendation_alignment_score": 0.25,
    "construction_score": 0.20,
    "risk_score": 0.10,
    "championship_equity_score": 0.15,
}


def _grade_from_score(score):
    score = _safe_float(score, 0.0)
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 60:
        return "D"
    return "F"


def _lookup_by_player(df, player_name):
    if df is None or df.empty or "player_name" not in df.columns:
        return {}

    player_key = _normalize_name(player_name)
    match = df[df["player_name"].astype(str).str.lower().str.strip() == player_key]
    if match.empty:
        return {}

    return match.iloc[0].to_dict()


def _load_player_context():
    players_df = load_players_df()
    player_col = safe_col(players_df, PLAYER_COLS) if not players_df.empty else None
    if players_df.empty or player_col is None:
        player_lookup = {}
    else:
        player_lookup = {
            _normalize_name(row[player_col]): row.to_dict()
            for _, row in players_df.iterrows()
        }

    try:
        equity_df = build_championship_equity_df()
    except Exception:
        equity_df = pd.DataFrame()

    try:
        trust_df = build_signal_trust_report()
    except Exception:
        trust_df = pd.DataFrame()

    return {
        "players_df": players_df,
        "player_lookup": player_lookup,
        "equity_df": equity_df,
        "trust_df": trust_df,
    }


def _player_position(player_name, player_row=None):
    player_row = player_row or {}
    for col in POSITION_COLS:
        if col in player_row and not pd.isna(player_row[col]):
            return str(player_row[col]).upper()

    position_map = st.session_state.get("player_position_map", {})
    return str(position_map.get(player_name, "")).upper()


def _recommendation_lookup(recommendations_df=None):
    recommendations = (
        build_master_recommendations_df()
        if recommendations_df is None
        else recommendations_df.copy()
    )
    if recommendations.empty or "player_name" not in recommendations.columns:
        return {}, recommendations

    lookup = {
        _normalize_name(row["player_name"]): row.to_dict()
        for _, row in recommendations.reset_index(drop=True).iterrows()
    }
    return lookup, recommendations


def _resolve_pick_record(pick_record=None):
    init_session_state()
    if pick_record:
        return dict(pick_record)

    history = st.session_state.get("recommendation_history", [])
    if history:
        return dict(history[-1])

    my_picks = [
        entry
        for entry in st.session_state.get("draft_log", [])
        if entry.get("action_type") == "my_pick"
    ]
    if my_picks:
        latest = my_picks[-1]
        return {
            "pick_number": latest.get("pick_number"),
            "selected_player": latest.get("player_name"),
            "recommended_player": None,
            "recommendation_score": None,
            "recommendation_followed": False,
        }

    return {
        "pick_number": get_current_pick_number(),
        "selected_player": None,
        "recommended_player": None,
        "recommendation_score": None,
        "recommendation_followed": False,
    }


def calculate_pick_value(player_selected, pick_number=None, recommendations_df=None, pick_record=None):
    recommendation_lookup, recommendations = _recommendation_lookup(recommendations_df)
    selected = recommendation_lookup.get(_normalize_name(player_selected), {})
    top_score = _safe_float(
        pick_record.get("recommendation_score") if pick_record else None,
        _safe_float(recommendations.iloc[0].get("final_score"), 75.0)
        if not recommendations.empty
        else 75.0,
    )
    selected_score = _safe_float(selected.get("final_score"), None)

    if selected_score is None:
        if pick_record and pick_record.get("recommendation_followed"):
            selected_score = top_score
        else:
            selected_score = 55.0

    score_gap = top_score - selected_score
    score = selected_score
    if score_gap <= 3:
        score += 12
    elif score_gap <= 8:
        score += 5
    elif score_gap >= 20:
        score -= 18
    elif score_gap >= 12:
        score -= 10

    adp = _safe_float(selected.get("adp"), None)
    if adp is not None and pick_number is not None:
        pick = _safe_float(pick_number, 1.0)
        if adp - pick >= 15:
            score += 8
        elif pick - adp >= 20:
            score -= 10

    return _clip_score(score)


def calculate_recommendation_alignment(player_selected, pick_record=None, recommendations_df=None):
    pick_record = pick_record or {}
    recommended_player = pick_record.get("recommended_player")

    if recommended_player and _normalize_name(player_selected) == _normalize_name(recommended_player):
        return 100.0

    recommendation_lookup, recommendations = _recommendation_lookup(recommendations_df)
    selected = recommendation_lookup.get(_normalize_name(player_selected), {})
    if not selected:
        return 45.0

    selected_score = _safe_float(selected.get("final_score"), 0.0)
    top_score = _safe_float(
        pick_record.get("recommendation_score"),
        _safe_float(recommendations.iloc[0].get("final_score"), selected_score)
        if not recommendations.empty
        else selected_score,
    )
    gap = top_score - selected_score

    if gap <= 3:
        return 90.0
    if gap <= 8:
        return 78.0
    if gap <= 15:
        return 62.0
    return 40.0


def calculate_construction_impact(player_selected, player_row=None):
    position = _player_position(player_selected, player_row)
    pressure = calculate_construction_pressure()
    position_pressure = pressure.get("position_pressure", {}).get(position, {})
    level = position_pressure.get("pressure_level", "NONE")

    level_scores = {
        "SEVERE": 95.0,
        "HIGH": 88.0,
        "MODERATE": 74.0,
        "LOW": 62.0,
        "NONE": 52.0,
    }
    score = level_scores.get(level, 52.0)

    counts = get_my_team_positions(load_players_df())
    current = counts.get(position, 0)
    target = st.session_state.get("roster_settings", {}).get(position, 0)
    if target and current > target + 2:
        score -= 12

    return _clip_score(score)


def calculate_risk_impact(player_selected, player_row=None, trust_row=None):
    player_row = player_row or {}
    trust_row = trust_row or {}
    injury_risk = 30.0
    bust_score = 40.0

    for col in INJURY_RISK_COLS:
        if col in player_row:
            injury_risk = _safe_float(player_row.get(col), injury_risk)
            break

    for col in BUST_COLS:
        if col in player_row:
            bust_score = _safe_float(player_row.get(col), bust_score)
            break

    signal_trust = _safe_float(trust_row.get("signal_trust_score"), 75.0)
    score = 100.0 - max(injury_risk, bust_score * 0.8)
    if signal_trust < 50:
        score -= 12

    return _clip_score(score)


def calculate_championship_equity_impact(player_selected, equity_row=None):
    equity_row = equity_row or {}
    return _clip_score(equity_row.get("championship_equity_score", 50.0))


def _weighted_grade_score(components):
    total_weight = sum(COMPONENT_WEIGHTS.values())
    if total_weight <= 0:
        return 0.0

    score = sum(
        _safe_float(components.get(component), 50.0) * weight
        for component, weight in COMPONENT_WEIGHTS.items()
    ) / total_weight
    return _clip_score(score)


def _grade_explanation(components, pick_record):
    positives = []
    negatives = []

    if components["recommendation_alignment_score"] >= 90:
        positives.append("Selected the recommended player or a very close alternative.")
    elif components["recommendation_alignment_score"] < 55:
        negatives.append("Pick was not closely aligned with the recommendation board.")

    if components["pick_value_score"] >= 80:
        positives.append("Strong value relative to the recommendation score.")
    elif components["pick_value_score"] < 55:
        negatives.append("Potential reach versus available recommendation value.")

    if components["construction_score"] >= 80:
        positives.append("Improved an important roster construction need.")
    elif components["construction_score"] < 55:
        negatives.append("Limited construction benefit at this point in the draft.")

    if components["risk_score"] >= 75:
        positives.append("Risk profile is acceptable.")
    elif components["risk_score"] < 55:
        negatives.append("Risk profile meaningfully lowers the grade.")

    if components["championship_equity_score"] >= 70:
        positives.append("Adds championship equity or upside.")
    elif components["championship_equity_score"] < 45:
        negatives.append("Limited championship equity signal.")

    if not positives:
        positives.append("Pick has some defensible value.")
    if not negatives:
        negatives.append("No major grade penalties detected.")

    return {
        "why": "Grade combines value, recommendation alignment, construction, risk, and championship equity.",
        "biggest_positives": positives,
        "biggest_negatives": negatives,
    }


def grade_pick(pick_record=None, recommendations_df=None):
    init_session_state()
    pick_record = _resolve_pick_record(pick_record)
    player_selected = pick_record.get("selected_player") or pick_record.get("player_selected")
    if not player_selected:
        return {}

    context = _load_player_context()
    player_row = context["player_lookup"].get(_normalize_name(player_selected), {})
    equity_row = _lookup_by_player(context["equity_df"], player_selected)
    trust_row = _lookup_by_player(context["trust_df"], player_selected)
    pick_number = pick_record.get("pick_number")

    components = {
        "pick_value_score": calculate_pick_value(
            player_selected,
            pick_number=pick_number,
            recommendations_df=recommendations_df,
            pick_record=pick_record,
        ),
        "construction_score": calculate_construction_impact(player_selected, player_row),
        "risk_score": calculate_risk_impact(player_selected, player_row, trust_row),
        "championship_equity_score": calculate_championship_equity_impact(
            player_selected,
            equity_row,
        ),
        "recommendation_alignment_score": calculate_recommendation_alignment(
            player_selected,
            pick_record=pick_record,
            recommendations_df=recommendations_df,
        ),
    }
    grade_score = _weighted_grade_score(components)
    explanation = _grade_explanation(components, pick_record)

    return {
        "pick_number": pick_number,
        "player_selected": player_selected,
        "recommended_player": pick_record.get("recommended_player"),
        "recommendation_followed": bool(pick_record.get("recommendation_followed", False)),
        "grade": _grade_from_score(grade_score),
        "grade_score": grade_score,
        **components,
        "why": explanation["why"],
        "biggest_positives": explanation["biggest_positives"],
        "biggest_negatives": explanation["biggest_negatives"],
    }


def build_pick_grade_report(pick_records=None, recommendations_df=None):
    init_session_state()
    records = pick_records
    if records is None:
        records = st.session_state.get("recommendation_history", [])

    rows = []
    for record in records:
        grade = grade_pick(record, recommendations_df=recommendations_df)
        if grade:
            rows.append(grade)

    return pd.DataFrame(rows)


def build_draft_review(pick_records=None, recommendations_df=None):
    report_df = build_pick_grade_report(
        pick_records=pick_records,
        recommendations_df=recommendations_df,
    )
    if report_df.empty:
        return {
            "best_pick": None,
            "worst_pick": None,
            "highest_upside_pick": None,
            "highest_risk_pick": None,
            "largest_reach": None,
            "largest_value_pick": None,
        }

    def row_max(column):
        return report_df.loc[report_df[column].idxmax()].to_dict()

    def row_min(column):
        return report_df.loc[report_df[column].idxmin()].to_dict()

    return {
        "best_pick": row_max("grade_score"),
        "worst_pick": row_min("grade_score"),
        "highest_upside_pick": row_max("championship_equity_score"),
        "highest_risk_pick": row_min("risk_score"),
        "largest_reach": row_min("pick_value_score"),
        "largest_value_pick": row_max("pick_value_score"),
    }


def get_draft_grade_debug_info(pick_records=None, recommendations_df=None):
    report_df = build_pick_grade_report(
        pick_records=pick_records,
        recommendations_df=recommendations_df,
    )
    if report_df.empty:
        return {
            "graded_pick_count": 0,
            "grade_distribution": {},
            "average_grade_score": 0.0,
            "sample_grades": [],
            "draft_review": build_draft_review(pick_records, recommendations_df),
        }

    return {
        "graded_pick_count": int(len(report_df)),
        "grade_distribution": report_df["grade"].value_counts().to_dict(),
        "average_grade_score": round(float(report_df["grade_score"].mean()), 2),
        "sample_grades": report_df.head(10).to_dict("records"),
        "draft_review": build_draft_review(pick_records, recommendations_df),
    }
