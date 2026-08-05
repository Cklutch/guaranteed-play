import pandas as pd

from draftkit.championship_equity import build_championship_equity_df
from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _normalize_name
from draftkit.common import safe_float as _safe_float
from draftkit.draft_analysis import build_master_recommendations_df
from draftkit.market_disagreement import build_market_disagreement_df


COMPONENT_COLUMNS = {
    "need_score": "position_need_component_score",
    "tier_score": "tier_urgency_component_score",
    "projection_score": "projection_component_score",
    "opponent_score": "opponent_component_score",
    "risk_score": "risk_component_score",
    "availability_score": "availability_component_score",
    "team_fit_score": "team_fit_component_score",
    "adp_score": "adp_value_component_score",
    "championship_equity_score": "championship_equity_score",
    "market_disagreement_score": "market_disagreement_score",
}

AUDIT_COLUMNS = [
    "player_name",
    "position",
    "overall_score",
    "recommendation_rank",
    "need_score",
    "tier_score",
    "projection_score",
    "opponent_score",
    "risk_score",
    "availability_score",
    "team_fit_score",
    "adp_score",
    "championship_equity_score",
    "market_disagreement_score",
]


def _derive_risk_score(row):
    if "risk_component_score" in row and not pd.isna(row.get("risk_component_score")):
        return _safe_float(row.get("risk_component_score"), 50.0)

    injury_risk = _safe_float(row.get("injury_risk"), 0.0)
    bust_score = _safe_float(row.get("bust_score"), 0.0)
    stability_score = _safe_float(row.get("stability_score"), 50.0)
    risk_penalty = max(injury_risk, bust_score, 100.0 - stability_score)

    if risk_penalty <= 0:
        return 50.0

    return _clip_score(100.0 - risk_penalty)


def _derive_availability_score(row):
    if "availability_component_score" in row and not pd.isna(row.get("availability_component_score")):
        return _safe_float(row.get("availability_component_score"), 50.0)

    fall_risk = str(row.get("fall_risk", "")).strip().upper()
    if fall_risk == "HIGH":
        return 85.0
    if fall_risk == "MEDIUM":
        return 65.0
    if fall_risk == "LOW":
        return 40.0

    return 50.0


def _normalize_market_disagreement(value):
    value = _safe_float(value, 0.0)
    return _clip_score(50.0 + value)


def _get_recommendations_df(recommendations_df=None):
    if recommendations_df is not None:
        return recommendations_df.copy()

    return build_master_recommendations_df()


def _build_signal_enriched_recommendations(recommendations_df=None):
    recommendations = _get_recommendations_df(recommendations_df)
    if recommendations.empty:
        return recommendations

    if (
        "recommendation_rank" in recommendations.columns
        and "market_disagreement_score" in recommendations.columns
        and "championship_equity_score" in recommendations.columns
    ):
        return recommendations.copy()

    out = recommendations.copy().reset_index(drop=True)
    out["recommendation_rank"] = out.index + 1
    out["player_key"] = out["player_name"].apply(_normalize_name)

    try:
        equity_df = build_championship_equity_df()
    except Exception:
        equity_df = pd.DataFrame()

    if not equity_df.empty and "player_name" in equity_df.columns:
        equity_subset = equity_df[["player_name", "championship_equity_score"]].copy()
        equity_subset["player_key"] = equity_subset["player_name"].apply(_normalize_name)
        equity_subset = equity_subset.drop_duplicates("player_key", keep="first")
        out = out.merge(
            equity_subset[["player_key", "championship_equity_score"]],
            on="player_key",
            how="left",
        )
    elif "championship_equity_score" not in out.columns:
        out["championship_equity_score"] = pd.NA

    try:
        market_df = build_market_disagreement_df()
    except Exception:
        market_df = pd.DataFrame()

    if not market_df.empty and "player_name" in market_df.columns:
        market_subset = market_df[
            ["player_name", "market_disagreement", "market_confidence", "sportsbook_projection"]
        ].copy()
        market_subset["player_key"] = market_subset["player_name"].apply(_normalize_name)
        market_subset = market_subset.drop_duplicates("player_key", keep="first")
        out = out.merge(
            market_subset[
                [
                    "player_key",
                    "market_disagreement",
                    "market_confidence",
                    "sportsbook_projection",
                ]
            ],
            on="player_key",
            how="left",
        )
    else:
        for column in ["market_disagreement", "market_confidence", "sportsbook_projection"]:
            if column not in out.columns:
                out[column] = pd.NA

    for column in ["market_disagreement", "market_confidence", "sportsbook_projection"]:
        if column not in out.columns:
            out[column] = pd.NA

    out["market_disagreement_score"] = out["market_disagreement"].apply(
        _normalize_market_disagreement
    )
    out["risk_score"] = out.apply(_derive_risk_score, axis=1)
    out["availability_score"] = out.apply(_derive_availability_score, axis=1)

    if "championship_equity_score" not in out.columns:
        out["championship_equity_score"] = pd.NA

    return out


def _component_value(row, component_name):
    if component_name == "risk_score":
        return _safe_float(row.get("risk_score"), 50.0)
    if component_name == "availability_score":
        return _safe_float(row.get("availability_score"), 50.0)

    column = COMPONENT_COLUMNS.get(component_name)
    if column and column in row:
        return _safe_float(row.get(column), 50.0)

    return 50.0


def _component_breakdown(row):
    breakdown = {
        "need_score": _component_value(row, "need_score"),
        "tier_score": _component_value(row, "tier_score"),
        "projection_score": _component_value(row, "projection_score"),
        "opponent_score": _component_value(row, "opponent_score"),
        "risk_score": _component_value(row, "risk_score"),
        "availability_score": _component_value(row, "availability_score"),
        "team_fit_score": _component_value(row, "team_fit_score"),
        "adp_score": _component_value(row, "adp_score"),
        "championship_equity_score": _safe_float(
            row.get("championship_equity_score"),
            50.0,
        ),
        "market_disagreement_score": _safe_float(
            row.get("market_disagreement_score"),
            50.0,
        ),
    }
    return {key: round(value, 2) for key, value in breakdown.items()}


def _component_contributions(row):
    breakdown = _component_breakdown(row)
    return {
        component: round(score - 50.0, 2)
        for component, score in breakdown.items()
    }


def _largest_items(values, positive=True, limit=5):
    items = [
        {"component": key, "impact": round(_safe_float(value), 2)}
        for key, value in values.items()
        if (_safe_float(value) > 0 if positive else _safe_float(value) < 0)
    ]
    return sorted(
        items,
        key=lambda item: item["impact"],
        reverse=positive,
    )[:limit]


def audit_recommendation(player_name, recommendations_df=None):
    recommendations = _build_signal_enriched_recommendations(recommendations_df)
    if recommendations.empty:
        return {}

    player_key = _normalize_name(player_name)
    match = recommendations[recommendations["player_key"] == player_key]
    if match.empty:
        return {}

    row = match.iloc[0]
    contributions = _component_contributions(row)
    return {
        "player_name": row.get("player_name"),
        "position": row.get("position"),
        "overall_score": round(_safe_float(row.get("final_score")), 2),
        "recommendation_rank": int(row.get("recommendation_rank")),
        "component_breakdown": _component_breakdown(row),
        "rank_change_analysis": _rank_change_for_row(row),
        "largest_bonuses": _largest_items(contributions, positive=True),
        "largest_penalties": _largest_items(contributions, positive=False),
        "anomalies": _anomalies_for_row(row),
    }


def compare_players(player_a, player_b, recommendations_df=None):
    recommendations = _build_signal_enriched_recommendations(recommendations_df)
    if recommendations.empty:
        return {}

    player_a_key = _normalize_name(player_a)
    player_b_key = _normalize_name(player_b)
    a_match = recommendations[recommendations["player_key"] == player_a_key]
    b_match = recommendations[recommendations["player_key"] == player_b_key]

    if a_match.empty or b_match.empty:
        return {
            "player_a": player_a,
            "player_b": player_b,
            "error": "One or both players were not found in recommendation output.",
        }

    a = a_match.iloc[0]
    b = b_match.iloc[0]
    a_breakdown = _component_breakdown(a)
    b_breakdown = _component_breakdown(b)
    diffs = {
        component: round(a_breakdown[component] - b_breakdown[component], 2)
        for component in a_breakdown
    }

    return {
        "player_a": a.get("player_name"),
        "player_b": b.get("player_name"),
        "score_difference": round(
            _safe_float(a.get("final_score")) - _safe_float(b.get("final_score")),
            2,
        ),
        "rank_difference": int(a.get("recommendation_rank")) - int(b.get("recommendation_rank")),
        "component_differences": diffs,
        "largest_contributing_factors": _largest_items(diffs, positive=True),
        "largest_penalties": _largest_items(diffs, positive=False),
        "largest_bonuses": _largest_items(_component_contributions(a), positive=True),
    }


def build_score_component_report(recommendations_df=None, limit=None):
    recommendations = _build_signal_enriched_recommendations(recommendations_df)
    if recommendations.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    rows = []
    for _, row in recommendations.iterrows():
        output_row = {
            "player_name": row.get("player_name"),
            "position": row.get("position"),
            "overall_score": round(_safe_float(row.get("final_score")), 2),
            "recommendation_rank": int(row.get("recommendation_rank")),
        }
        output_row.update(_component_breakdown(row))
        rows.append(output_row)

    report_df = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if limit:
        return report_df.head(limit)
    return report_df


def _rank_change_for_row(row):
    contributions = _component_contributions(row)
    items = [
        {
            "component": component,
            "rank_impact_estimate": impact,
        }
        for component, impact in contributions.items()
        if abs(impact) >= 1.0
    ]
    return sorted(
        items,
        key=lambda item: abs(item["rank_impact_estimate"]),
        reverse=True,
    )


def build_rank_change_report(recommendations_df=None, limit=20):
    recommendations = _build_signal_enriched_recommendations(recommendations_df)
    if recommendations.empty:
        return []

    rows = []
    for _, row in recommendations.head(limit).iterrows():
        rows.append({
            "player_name": row.get("player_name"),
            "recommendation_rank": int(row.get("recommendation_rank")),
            "overall_score": round(_safe_float(row.get("final_score")), 2),
            "rank_change_analysis": _rank_change_for_row(row),
        })

    return rows


def _anomalies_for_row(row):
    anomalies = []

    adp_score = _component_value(row, "adp_score")
    need_score = _component_value(row, "need_score")
    risk_score = _component_value(row, "risk_score")

    if adp_score >= 85 or adp_score <= 20:
        anomalies.append("High ADP influence")
    if need_score >= 85:
        anomalies.append("High need influence")
    if risk_score <= 40:
        anomalies.append("High risk influence")
    if pd.isna(row.get("projection_points")) and pd.isna(row.get("projection_component_score")):
        anomalies.append("Missing projection inputs")
    if pd.isna(row.get("sportsbook_projection")):
        anomalies.append("Missing sportsbook inputs")
    if pd.isna(row.get("championship_equity_score")):
        anomalies.append("Missing championship equity signal")
    if pd.isna(row.get("market_disagreement")):
        anomalies.append("Missing market disagreement signal")

    return anomalies


def find_score_anomalies(recommendations_df=None, limit=50):
    recommendations = _build_signal_enriched_recommendations(recommendations_df)
    if recommendations.empty:
        return []

    rows = []
    for _, row in recommendations.iterrows():
        anomalies = _anomalies_for_row(row)
        if anomalies:
            rows.append({
                "player_name": row.get("player_name"),
                "recommendation_rank": int(row.get("recommendation_rank")),
                "overall_score": round(_safe_float(row.get("final_score")), 2),
                "anomalies": anomalies,
            })

    return rows[:limit]


def build_recommendation_audit_report(recommendations_df=None, limit=20):
    recommendations = _build_signal_enriched_recommendations(recommendations_df)
    component_report = build_score_component_report(recommendations, limit=limit)

    if recommendations.empty:
        return {
            "recommendation_count": 0,
            "score_component_report": [],
            "rank_change_report": [],
            "anomalies": [],
            "largest_score_drivers": [],
            "largest_penalties": [],
        }

    all_contributions = []
    for _, row in recommendations.iterrows():
        for component, impact in _component_contributions(row).items():
            all_contributions.append({
                "player_name": row.get("player_name"),
                "component": component,
                "impact": impact,
            })

    contribution_df = pd.DataFrame(all_contributions)
    largest_drivers = (
        contribution_df[contribution_df["impact"] > 0]
        .sort_values("impact", ascending=False)
        .head(10)
        .to_dict("records")
    )
    largest_penalties = (
        contribution_df[contribution_df["impact"] < 0]
        .sort_values("impact", ascending=True)
        .head(10)
        .to_dict("records")
    )

    return {
        "recommendation_count": int(len(recommendations)),
        "score_component_report": component_report.to_dict("records"),
        "rank_change_report": build_rank_change_report(recommendations, limit=limit),
        "anomalies": find_score_anomalies(recommendations, limit=limit),
        "largest_score_drivers": largest_drivers,
        "largest_penalties": largest_penalties,
    }


def get_audit_debug_info(recommendations_df=None):
    recommendations = _build_signal_enriched_recommendations(recommendations_df)
    if recommendations.empty:
        return {
            "recommendation_count": 0,
            "component_coverage": {},
            "missing_signals": [],
            "largest_score_drivers": [],
        }

    component_report = build_score_component_report(recommendations)
    component_coverage = {}
    missing_signals = []

    for component in COMPONENT_COLUMNS:
        if component not in component_report.columns:
            component_coverage[component] = 0.0
            missing_signals.append(component)
            continue

        coverage = round(float(component_report[component].notna().mean() * 100.0), 2)
        component_coverage[component] = coverage
        if coverage <= 0:
            missing_signals.append(component)

    report = build_recommendation_audit_report(recommendations, limit=10)
    return {
        "recommendation_count": int(len(recommendations)),
        "component_coverage": component_coverage,
        "missing_signals": missing_signals,
        "largest_score_drivers": report["largest_score_drivers"],
        "largest_penalties": report["largest_penalties"],
    }
