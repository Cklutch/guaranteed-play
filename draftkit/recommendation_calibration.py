import pandas as pd
import streamlit as st

from draftkit.draft_analysis import (
    DEFAULT_MASTER_COMPONENT_WEIGHTS,
    build_master_recommendations_df,
)
from draftkit.draft_state import get_my_team_positions, init_session_state


POSITIONS_TO_CALIBRATE = ["QB", "RB", "WR", "TE"]

CALIBRATION_COMPONENTS = {
    "overall_score": "final_score",
    "need_score": "position_need_component_score",
    "tier_score": "tier_urgency_component_score",
    "projection_score": "projection_component_score",
    "opponent_score": "opponent_component_score",
    "risk_score": "risk_component_score",
    "team_fit_score": "team_fit_component_score",
    "availability_score": "availability_component_score",
    "adp_score": "adp_value_component_score",
}

ENGINE_WEIGHT_KEYS = {
    "projection_score": "projection",
    "need_score": "position_need",
    "adp_score": "adp_value",
    "tier_score": "tier_urgency",
    "team_fit_score": "team_fit",
}


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_recommendations_df(recommendations_df=None):
    if recommendations_df is not None:
        return recommendations_df.copy()

    return build_master_recommendations_df()


def _component_score(row, component_name):
    source_col = CALIBRATION_COMPONENTS.get(component_name)
    if not source_col:
        return 50.0

    if source_col in row:
        return _safe_float(row.get(source_col), 50.0)

    if component_name == "risk_score":
        return _derive_risk_score(row)

    if component_name == "availability_score":
        return _derive_availability_score(row)

    return 50.0


def _derive_risk_score(row):
    injury_risk = _safe_float(row.get("injury_risk"), 0.0)
    bust_score = _safe_float(row.get("bust_score"), 0.0)
    stability_score = _safe_float(row.get("stability_score"), 50.0)

    risk_penalty = max(injury_risk, bust_score, 100.0 - stability_score)
    if risk_penalty <= 0:
        return 50.0

    return round(max(0.0, min(100.0, 100.0 - risk_penalty)), 2)


def _derive_availability_score(row):
    fall_risk = str(row.get("fall_risk", "")).strip().upper()
    if fall_risk == "HIGH":
        return 85.0
    if fall_risk == "MEDIUM":
        return 65.0
    if fall_risk == "LOW":
        return 40.0

    return 50.0


def _engine_weights():
    total = sum(DEFAULT_MASTER_COMPONENT_WEIGHTS.values())
    if total <= 0:
        return DEFAULT_MASTER_COMPONENT_WEIGHTS.copy()

    return {
        key: value / total
        for key, value in DEFAULT_MASTER_COMPONENT_WEIGHTS.items()
    }


def _weighted_score_without_component(row, component_name):
    weights = _engine_weights()
    excluded_weight_key = ENGINE_WEIGHT_KEYS.get(component_name)

    weighted_total = 0.0
    active_weight = 0.0
    for score_name, weight_key in ENGINE_WEIGHT_KEYS.items():
        if weight_key == excluded_weight_key:
            continue

        weight = weights.get(weight_key, 0.0)
        weighted_total += _component_score(row, score_name) * weight
        active_weight += weight

    if active_weight <= 0:
        return _safe_float(row.get("final_score"), 0.0)

    return round(weighted_total / active_weight, 2)


def _rank_impact_df(recommendations_df, component_name):
    if recommendations_df.empty:
        return pd.DataFrame()

    working_df = recommendations_df.copy().reset_index(drop=True)
    working_df["final_rank"] = working_df.index + 1
    working_df[f"score_without_{component_name}"] = working_df.apply(
        lambda row: _weighted_score_without_component(row, component_name),
        axis=1,
    )
    working_df[f"rank_without_{component_name}"] = (
        working_df[f"score_without_{component_name}"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    working_df[f"{component_name}_rank_delta"] = (
        working_df[f"rank_without_{component_name}"] - working_df["final_rank"]
    )

    return working_df


def _primary_driver(row):
    driver_scores = {
        "Projection": abs(_component_score(row, "projection_score") - 50.0),
        "Need": abs(_component_score(row, "need_score") - 50.0),
        "ADP": abs(_component_score(row, "adp_score") - 50.0),
        "Tier": abs(_component_score(row, "tier_score") - 50.0),
        "Team Fit": abs(_component_score(row, "team_fit_score") - 50.0),
        "Risk": abs(_component_score(row, "risk_score") - 50.0),
        "Availability": abs(_component_score(row, "availability_score") - 50.0),
    }

    return max(driver_scores, key=driver_scores.get)


def _impact_summary(impact_df, component_name, top_n=5):
    delta_col = f"{component_name}_rank_delta"
    score_col = CALIBRATION_COMPONENTS.get(component_name)

    if impact_df.empty or delta_col not in impact_df.columns:
        return {
            "average_rank_change": 0.0,
            "players_promoted_count": 0,
            "players_penalized_count": 0,
            "promoted_players": [],
            "penalized_players": [],
            "distribution": {},
        }

    distribution = (
        impact_df[delta_col]
        .describe()
        .round(2)
        .to_dict()
    )

    cols = [
        col for col in [
            "player_name",
            "position",
            "final_score",
            score_col,
            delta_col,
        ]
        if col and col in impact_df.columns
    ]

    promoted = (
        impact_df[impact_df[delta_col] > 0]
        .sort_values(delta_col, ascending=False)
        .head(top_n)
    )
    penalized = (
        impact_df[impact_df[delta_col] < 0]
        .sort_values(delta_col, ascending=True)
        .head(top_n)
    )

    return {
        "average_rank_change": round(
            _safe_float(impact_df[delta_col].abs().mean()),
            2,
        ),
        "players_promoted_count": int((impact_df[delta_col] > 0).sum()),
        "players_penalized_count": int((impact_df[delta_col] < 0).sum()),
        "promoted_players": promoted[cols].to_dict("records"),
        "penalized_players": penalized[cols].to_dict("records"),
        "distribution": distribution,
    }


def analyze_score_components(recommendations_df=None, top_n=20):
    recommendations_df = _get_recommendations_df(recommendations_df)
    if recommendations_df.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in recommendations_df.head(top_n).reset_index(drop=True).iterrows():
        output_row = {
            "rank": idx + 1,
            "player_name": row.get("player_name"),
            "position": row.get("position"),
        }

        for component_name in CALIBRATION_COMPONENTS:
            output_row[component_name] = round(_component_score(row, component_name), 2)

        output_row["primary_driver"] = _primary_driver(row)
        rows.append(output_row)

    return pd.DataFrame(rows)


def analyze_need_impact(recommendations_df=None, top_n=5):
    recommendations_df = _get_recommendations_df(recommendations_df)
    impact_df = _rank_impact_df(recommendations_df, "need_score")
    return _impact_summary(impact_df, "need_score", top_n=top_n)


def analyze_adp_impact(recommendations_df=None, top_n=5):
    recommendations_df = _get_recommendations_df(recommendations_df)
    impact_df = _rank_impact_df(recommendations_df, "adp_score")
    return _impact_summary(impact_df, "adp_score", top_n=top_n)


def analyze_risk_impact(recommendations_df=None, top_n=5):
    recommendations_df = _get_recommendations_df(recommendations_df)
    if recommendations_df.empty:
        return _impact_summary(pd.DataFrame(), "risk_score", top_n=top_n)

    working_df = recommendations_df.copy()
    working_df["risk_component_score"] = working_df.apply(_derive_risk_score, axis=1)
    working_df["final_rank"] = range(1, len(working_df) + 1)
    working_df["risk_rank"] = working_df["risk_component_score"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    working_df["risk_score_rank_delta"] = working_df["risk_rank"] - working_df["final_rank"]

    high_risk = working_df[working_df["risk_component_score"] < 45].copy()
    high_risk_cols = [
        col for col in [
            "player_name",
            "position",
            "final_score",
            "risk_component_score",
            "injury_risk",
            "bust_score",
            "stability_score",
        ]
        if col in high_risk.columns
    ]

    summary = _impact_summary(working_df, "risk_score", top_n=top_n)
    summary["high_risk_recommendations"] = (
        high_risk.sort_values("final_score", ascending=False)
        .head(top_n)[high_risk_cols]
        .to_dict("records")
        if high_risk_cols
        else []
    )
    return summary


def analyze_tier_impact(recommendations_df=None, top_n=5):
    recommendations_df = _get_recommendations_df(recommendations_df)
    impact_df = _rank_impact_df(recommendations_df, "tier_score")
    return _impact_summary(impact_df, "tier_score", top_n=top_n)


def analyze_construction_pressure(players_df=None):
    init_session_state()
    counts = get_my_team_positions(players_df)
    roster_settings = st.session_state.get("roster_settings", {})

    position_reports = []
    total_scorable_players = sum(
        _safe_int(counts.get(position))
        for position in POSITIONS_TO_CALIBRATE
    )
    total_required = sum(
        _safe_int(roster_settings.get(position))
        for position in POSITIONS_TO_CALIBRATE
    )

    max_overfill = 0
    for position in POSITIONS_TO_CALIBRATE:
        target = _safe_int(roster_settings.get(position), 0)
        current = _safe_int(counts.get(position), 0)
        max_overfill = max(max_overfill, current - target)

    for position in POSITIONS_TO_CALIBRATE:
        target = _safe_int(roster_settings.get(position), 0)
        current = _safe_int(counts.get(position), 0)
        missing = max(target - current, 0)
        overfilled = max(current - target, 0)

        if target <= 0:
            pressure_score = 0.0
        elif missing > 0:
            pressure_score = 60.0 + min(15.0, missing * 7.5)
            if current == 0:
                pressure_score += 5.0
        elif overfilled > 0:
            pressure_score = 15.0
        else:
            pressure_score = 35.0

        if missing > 0 and max_overfill >= 2:
            pressure_score += 10.0
        if missing > 0 and max_overfill >= 3:
            pressure_score += 10.0
        if missing > 0 and max_overfill >= 4:
            pressure_score += 15.0

        pressure_score = round(max(0.0, min(100.0, pressure_score)), 2)

        if pressure_score >= 90:
            pressure_label = "severe"
        elif pressure_score >= 70:
            pressure_label = "increasing"
        elif pressure_score >= 45:
            pressure_label = "moderate"
        else:
            pressure_label = "low"

        share = (
            current / total_scorable_players
            if total_scorable_players > 0
            else 0.0
        )

        position_reports.append({
            "position": position,
            "current_count": current,
            "target_count": target,
            "missing_count": missing,
            "overfilled_count": overfilled,
            "roster_share": round(share, 3),
            "pressure_score": pressure_score,
            "pressure_label": pressure_label,
            "message": f"{position} pressure {pressure_label}",
        })

    return {
        "roster_counts": {
            position: _safe_int(counts.get(position), 0)
            for position in POSITIONS_TO_CALIBRATE
        },
        "roster_targets": {
            position: _safe_int(roster_settings.get(position), 0)
            for position in POSITIONS_TO_CALIBRATE
        },
        "total_scorable_players": total_scorable_players,
        "total_required_starters": total_required,
        "position_pressure": position_reports,
    }


def build_calibration_report(recommendations_df=None, top_n=10):
    recommendations_df = _get_recommendations_df(recommendations_df)
    score_breakdown = analyze_score_components(
        recommendations_df,
        top_n=top_n,
    )

    component_averages = {}
    if not score_breakdown.empty:
        component_averages = {
            component_name: round(
                _safe_float(score_breakdown[component_name].mean()),
                2,
            )
            for component_name in CALIBRATION_COMPONENTS
            if component_name in score_breakdown.columns
        }

    influence_averages = {}
    for component_name in [
        "need_score",
        "tier_score",
        "projection_score",
        "team_fit_score",
        "adp_score",
        "risk_score",
        "availability_score",
        "opponent_score",
    ]:
        if score_breakdown.empty or component_name not in score_breakdown.columns:
            influence_averages[component_name] = 0.0
        else:
            influence_averages[component_name] = round(
                _safe_float((score_breakdown[component_name] - 50.0).abs().mean()),
                2,
            )

    return {
        "recommendation_count": int(len(recommendations_df)),
        "top_recommendations": score_breakdown.to_dict("records"),
        "component_averages": component_averages,
        "average_component_influence": influence_averages,
        "primary_driver_counts": (
            score_breakdown["primary_driver"].value_counts().to_dict()
            if "primary_driver" in score_breakdown.columns
            else {}
        ),
        "construction_pressure": analyze_construction_pressure(),
        "adp_influence": analyze_adp_impact(recommendations_df),
        "need_influence": analyze_need_impact(recommendations_df),
        "risk_influence": analyze_risk_impact(recommendations_df),
        "tier_influence": analyze_tier_impact(recommendations_df),
    }


def get_calibration_debug_info(recommendations_df=None):
    recommendations_df = _get_recommendations_df(recommendations_df)

    available_columns = set(recommendations_df.columns) if not recommendations_df.empty else set()
    missing_component_columns = [
        source_col
        for source_col in CALIBRATION_COMPONENTS.values()
        if source_col not in available_columns
    ]

    return {
        "recommendation_rows": int(len(recommendations_df)),
        "component_columns_available": {
            component_name: source_col in available_columns
            for component_name, source_col in CALIBRATION_COMPONENTS.items()
        },
        "missing_component_columns": missing_component_columns,
        "engine_weights": DEFAULT_MASTER_COMPONENT_WEIGHTS.copy(),
        "construction_pressure": analyze_construction_pressure(),
    }
