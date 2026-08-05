import pandas as pd

from draftkit.common import name_key as _normalize_player_name
from draftkit.common import safe_float as _safe_float
from draftkit.draft_analysis import (
    build_master_recommendations_df,
    build_position_urgency_df,
    get_best_pick_recommendation,
    get_position_need_weights,
    get_team_profile,
    get_tier_cliff_recommendation,
)
from draftkit.draft_simulation import calculate_availability_probability
from draftkit.opponent_model import (
    build_position_demand_forecast,
    get_position_threat_report,
)
from draftkit.projection_enrichment import (
    build_projection_enriched_dataset,
    get_projection_enrichment_debug_info,
)


SCORE_FIELDS = [
    "overall_score",
    "need_score",
    "tier_score",
    "projection_score",
    "opponent_score",
    "risk_score",
    "team_fit_score",
    "availability_score",
]
COMPONENT_COLUMNS = {
    "need_score": "position_need_component_score",
    "tier_score": "tier_urgency_component_score",
    "projection_score": "projection_component_score",
    "team_fit_score": "team_fit_component_score",
}
RISK_ARCHETYPES = {"RISKY", "BOOM"}
_CACHE = {}


def clear_explainer_cache():
    _CACHE.clear()


def prime_recommendation_explainer_cache(
    recommendations_df=None,
    availability_df=None,
    position_forecast_df=None,
    projection_enriched_df=None,
):
    if recommendations_df is not None:
        _CACHE["recommendations_df"] = recommendations_df.copy()
    if availability_df is not None:
        _CACHE["availability_df"] = availability_df.copy()
    if position_forecast_df is not None:
        _CACHE["position_forecast_df"] = position_forecast_df.copy()
    if projection_enriched_df is not None:
        _CACHE["projection_enriched_df"] = projection_enriched_df.copy()


def _get_recommendations_df():
    if "recommendations_df" not in _CACHE:
        _CACHE["recommendations_df"] = build_master_recommendations_df()
    return _CACHE["recommendations_df"]


def _get_recommendation_row(player_name=None, recommendations_df=None):
    recommendations_df = _get_recommendations_df() if recommendations_df is None else recommendations_df
    if recommendations_df.empty:
        return None, recommendations_df

    if player_name:
        match = recommendations_df[
            recommendations_df["player_name"].astype(str).str.lower()
            == _normalize_player_name(player_name)
        ]
        if not match.empty:
            return match.iloc[0].to_dict(), recommendations_df

    return recommendations_df.iloc[0].to_dict(), recommendations_df


def _get_projection_row(player_name):
    if "projection_enriched_df" not in _CACHE:
        _CACHE["projection_enriched_df"] = build_projection_enriched_dataset()
    enriched_df = _CACHE["projection_enriched_df"]
    if enriched_df.empty or not player_name:
        return None

    match = enriched_df[
        enriched_df["player_name"].astype(str).str.lower()
        == _normalize_player_name(player_name)
    ]
    if match.empty:
        return None

    return match.iloc[0].to_dict()


def _get_availability_row(player_name):
    if "availability_df" not in _CACHE:
        _CACHE["availability_df"] = calculate_availability_probability(
            num_simulations=75,
            seed=17,
        )
    availability_df = _CACHE["availability_df"]
    if availability_df.empty or not player_name:
        return None

    match = availability_df[
        availability_df["player_name"].astype(str).str.lower()
        == _normalize_player_name(player_name)
    ]
    if match.empty:
        return None

    return match.iloc[0].to_dict()


def _get_position_forecast(position):
    if "position_forecast_df" not in _CACHE:
        _CACHE["position_forecast_df"] = build_position_demand_forecast()
    forecast_df = _CACHE["position_forecast_df"]
    if forecast_df.empty:
        return None

    match = forecast_df[forecast_df["position"].astype(str).str.upper() == str(position).upper()]
    if match.empty:
        return None

    return match.iloc[0].to_dict()


def _score_from_opponent_forecast(position):
    forecast = _get_position_forecast(position)
    if not forecast:
        return 0.0

    expected_picks = _safe_float(forecast.get("expected_picks_before_next_turn"), 0.0)
    tier_pressure = _safe_float(forecast.get("tier_pressure"), 1.0)
    score = min((expected_picks * 28.0) + (tier_pressure * 12.0), 100.0)
    return round(score, 2)


def _risk_score_from_row(row):
    injury_risk = _safe_float(row.get("injury_risk"), 50.0)
    archetype = str(row.get("archetype", "")).upper()
    durability = _safe_float(row.get("durability_grade"), 70.0)
    volatility_penalty = 18.0 if archetype in RISK_ARCHETYPES else 0.0
    durability_penalty = max(70.0 - durability, 0.0) * 0.40
    raw_risk = injury_risk + volatility_penalty + durability_penalty
    return round(max(0.0, min(raw_risk, 100.0)), 2)


def _availability_score(player_name):
    availability = _get_availability_row(player_name)
    if not availability:
        return 0.0

    drafted_probability = _safe_float(
        availability.get("drafted_before_next_pick_probability"),
        0.0,
    )
    return round(max(0.0, min(drafted_probability, 100.0)), 2)


def build_score_breakdown(player_name=None, recommendations_df=None):
    """
    Return the standardized score breakdown used by the explanation layer.
    """
    row, _ = _get_recommendation_row(player_name, recommendations_df=recommendations_df)
    if row is None:
        return {field: None for field in SCORE_FIELDS}

    risk_score = _risk_score_from_row(row)
    breakdown = {
        "overall_score": _safe_float(row.get("final_score"), 0.0),
        "need_score": _safe_float(row.get(COMPONENT_COLUMNS["need_score"]), 0.0),
        "tier_score": _safe_float(row.get(COMPONENT_COLUMNS["tier_score"]), 0.0),
        "projection_score": _safe_float(row.get(COMPONENT_COLUMNS["projection_score"]), 0.0),
        "opponent_score": _score_from_opponent_forecast(row.get("position")),
        "risk_score": risk_score,
        "team_fit_score": _safe_float(row.get(COMPONENT_COLUMNS["team_fit_score"]), 0.0),
        "availability_score": _availability_score(row.get("player_name")),
    }

    positive_components = [
        key for key in [
            "need_score",
            "tier_score",
            "projection_score",
            "opponent_score",
            "team_fit_score",
            "availability_score",
        ]
        if breakdown[key] >= 65.0
    ]
    risk_components = ["risk_score"] if risk_score >= 65.0 else []

    return {
        **breakdown,
        "player_name": row.get("player_name"),
        "position": row.get("position"),
        "top_positive_components": sorted(
            positive_components,
            key=lambda key: breakdown[key],
            reverse=True,
        )[:3],
        "risk_components": risk_components,
        "raw_inputs": {
            "projection_points": row.get("projection_points"),
            "position_value_score": row.get("position_value_score"),
            "need_bonus": row.get("need_bonus"),
            "value_score": row.get("value_score"),
            "urgency_score": row.get("urgency_score"),
            "team_fit_bonus": row.get("team_fit_bonus"),
            "injury_risk": row.get("injury_risk"),
            "durability_grade": row.get("durability_grade"),
            "archetype": row.get("archetype"),
        },
    }


def explain_position_need(player_or_row):
    row = player_or_row if isinstance(player_or_row, dict) else _get_recommendation_row(player_or_row)[0]
    if row is None:
        return {"supported": False, "message": "No player data available."}

    position = str(row.get("position", "")).upper()
    need_weight = _safe_float(get_position_need_weights().get(position), 1.0)

    if need_weight >= 1.35:
        reason = "Fills largest remaining roster need"
        message = f"{position} is a priority roster need."
    elif need_weight > 1.0:
        reason = "Addresses a roster need"
        message = f"{position} still improves roster construction."
    elif need_weight >= 0.75:
        reason = None
        message = f"{position} is close to filled."
    else:
        reason = None
        message = f"{position} is a lower roster need right now."

    return {
        "supported": reason is not None,
        "position": position,
        "need_weight": round(need_weight, 3),
        "reason": reason,
        "message": message,
    }


def explain_tier_pressure(player_or_row):
    row = player_or_row if isinstance(player_or_row, dict) else _get_recommendation_row(player_or_row)[0]
    if row is None:
        return {"supported": False, "message": "No player data available."}

    position = str(row.get("position", "")).upper()
    if "tier_urgency_df" not in _CACHE:
        _CACHE["tier_urgency_df"] = build_position_urgency_df()
    urgency_df = _CACHE["tier_urgency_df"]
    if urgency_df.empty:
        return {
            "supported": False,
            "position": position,
            "message": "Tier pressure is unavailable.",
        }

    match = urgency_df[urgency_df["Position"].astype(str).str.upper() == position]
    if match.empty:
        return {
            "supported": False,
            "position": position,
            "message": f"No immediate {position} tier cliff detected.",
        }

    tier_row = match.iloc[0].to_dict()
    urgency_score = _safe_float(tier_row.get("Urgency Score"), 0.0)
    urgency_label = str(tier_row.get("Urgency Label", "Low")).upper()
    supported = urgency_score >= 45.0

    return {
        "supported": supported,
        "position": position,
        "reason": "Tier cliff expected before next selection" if supported else None,
        "threat_level": urgency_label,
        "urgency_score": urgency_score,
        "players_left_in_tier": int(_safe_float(tier_row.get("Players Left In Tier"), 0.0)),
        "tier_dropoff": _safe_float(tier_row.get("Tier Dropoff"), 0.0),
        "message": (
            f"{position} tier pressure is {urgency_label.lower()} with "
            f"{tier_row.get('Players Left In Tier')} player(s) left in the tier."
        ),
    }


def explain_opponent_pressure(player_or_row):
    row = player_or_row if isinstance(player_or_row, dict) else _get_recommendation_row(player_or_row)[0]
    if row is None:
        return {"supported": False, "message": "No player data available."}

    position = str(row.get("position", "")).upper()
    forecast = _get_position_forecast(position)
    if "position_threat_report" not in _CACHE:
        _CACHE["position_threat_report"] = get_position_threat_report()
    threat_report = _CACHE["position_threat_report"]

    if not forecast:
        return {
            "supported": False,
            "position": position,
            "message": "Opponent pressure is unavailable.",
        }

    expected_picks = _safe_float(forecast.get("expected_picks_before_next_turn"), 0.0)
    threat_level = str(forecast.get("threat_level", "LOW")).upper()
    likely_losses = []
    for threat in threat_report:
        if str(threat.get("position", "")).upper() == position:
            likely_losses = threat.get("likely_player_losses", [])
            break

    supported = threat_level in {"HIGH", "CRITICAL"} or expected_picks >= 1.25
    return {
        "supported": supported,
        "position": position,
        "reason": "Position under heavy opponent demand" if supported else None,
        "threat_level": threat_level,
        "expected_picks_before_next_turn": expected_picks,
        "position_demand": forecast.get("position_demand"),
        "likely_player_losses": likely_losses,
        "message": (
            f"Opponent model expects about {expected_picks} {position} pick(s) "
            f"before your next turn."
        ),
    }


def explain_team_fit(player_or_row):
    row = player_or_row if isinstance(player_or_row, dict) else _get_recommendation_row(player_or_row)[0]
    if row is None:
        return {"supported": False, "message": "No player data available."}

    if "team_profile" not in _CACHE:
        _CACHE["team_profile"] = get_team_profile()
    team_profile = _CACHE["team_profile"]
    team_fit_bonus = _safe_float(row.get("team_fit_bonus"), 1.0)
    archetype = str(row.get("archetype", "")).upper()

    if team_fit_bonus >= 1.05:
        reason = "Improves roster balance"
        message = f"{archetype} profile improves the current {team_profile.get('profile')} roster build."
    elif team_fit_bonus <= 0.95:
        reason = None
        message = f"{archetype} profile may add roster construction risk."
    else:
        reason = None
        message = f"{archetype} profile is a neutral team fit."

    return {
        "supported": reason is not None,
        "reason": reason,
        "team_profile": team_profile,
        "player_archetype": archetype,
        "team_fit_bonus": team_fit_bonus,
        "message": message,
    }


def explain_projection_advantage(player_or_row):
    row = player_or_row if isinstance(player_or_row, dict) else _get_recommendation_row(player_or_row)[0]
    if row is None:
        return {"supported": False, "message": "No player data available."}

    projection_row = _get_projection_row(row.get("player_name"))
    if projection_row is None or pd.isna(projection_row.get("market_projection")):
        return {
            "supported": False,
            "projection_source": None,
            "message": "Sportsbook projection support is not available yet.",
        }

    confidence = _safe_float(projection_row.get("projection_confidence"), 0.0)
    market_projection = projection_row.get("market_projection")
    supported = confidence >= 40.0

    return {
        "supported": supported,
        "reason": "Market projection advantage" if supported else None,
        "projection_source": projection_row.get("projection_source"),
        "market_projection": market_projection,
        "projection_confidence": confidence,
        "projection_variance": projection_row.get("projection_variance"),
        "message": (
            f"Sportsbook markets project {market_projection} fantasy points "
            f"with {confidence}% confidence."
        ),
    }


def explain_risk_impact(player_or_row):
    row = player_or_row if isinstance(player_or_row, dict) else _get_recommendation_row(player_or_row)[0]
    if row is None:
        return {
            "risks": ["No player data available for risk explanation."],
            "risk_score": None,
        }

    risks = []
    position_need = explain_position_need(row)
    team_fit = explain_team_fit(row)
    injury_risk = _safe_float(row.get("injury_risk"), 50.0)
    archetype = str(row.get("archetype", "")).upper()
    if "team_profile" not in _CACHE:
        _CACHE["team_profile"] = get_team_profile()
    team_profile = _CACHE["team_profile"]

    if archetype in RISK_ARCHETYPES:
        risks.append("Increases roster volatility")
    if injury_risk >= 65.0:
        risks.append("Injury risk concentration")
    if position_need.get("need_weight", 1.0) < 0.75:
        risks.append("Creates positional imbalance")
    if team_fit.get("team_fit_bonus", 1.0) <= 0.95:
        risks.append("May not improve current team construction")
    if team_profile.get("volatility") == "High" and archetype in RISK_ARCHETYPES:
        risks.append("Adds volatility to an already volatile roster")

    return {
        "risk_score": _risk_score_from_row(row),
        "risks": risks,
        "message": "No major risk flags." if not risks else "; ".join(risks),
    }


def _key_reason_for_row(row):
    position_need = explain_position_need(row)
    tier_pressure = explain_tier_pressure(row)
    opponent_pressure = explain_opponent_pressure(row)

    for explanation in [position_need, tier_pressure, opponent_pressure]:
        if explanation.get("supported") and explanation.get("reason"):
            return explanation["reason"]

    value_score = _safe_float(row.get("adp_value_component_score"), 0.0)
    if value_score >= 65.0:
        return "Strong value relative to ranking"

    return "Strong overall score"


def _top_alternatives(player_name, recommendations_df, limit=3):
    if recommendations_df.empty:
        return []

    alternatives_df = recommendations_df.copy()
    if player_name:
        alternatives_df = alternatives_df[
            alternatives_df["player_name"].astype(str).str.lower()
            != _normalize_player_name(player_name)
        ]

    alternatives = []
    for _, alt_row in alternatives_df.head(limit).iterrows():
        row = alt_row.to_dict()
        alternatives.append({
            "player_name": row.get("player_name"),
            "overall_score": _safe_float(row.get("final_score"), 0.0),
            "key_reason": _key_reason_for_row(row),
        })

    return alternatives


def generate_recommendation_reasons(player_name=None, recommendations_df=None):
    row, _ = _get_recommendation_row(player_name, recommendations_df=recommendations_df)
    if row is None:
        return ["No recommendation is available."]

    explanations = [
        explain_position_need(row),
        explain_tier_pressure(row),
        explain_opponent_pressure(row),
        explain_projection_advantage(row),
        explain_team_fit(row),
    ]
    reasons = [
        explanation["reason"]
        for explanation in explanations
        if explanation.get("supported") and explanation.get("reason")
    ]

    value_score = _safe_float(row.get("adp_value_component_score"), 0.0)
    if value_score >= 65.0:
        reasons.append("Strong value relative to ranking")

    risk_impact = explain_risk_impact(row)
    if not risk_impact.get("risks") and str(row.get("archetype", "")).upper() in {"SAFE", "STEADY"}:
        reasons.append("Improves roster stability")

    return list(dict.fromkeys(reasons))


def build_player_decision_card(player_name=None, recommendations_df=None):
    row, recommendations_df = _get_recommendation_row(
        player_name,
        recommendations_df=recommendations_df,
    )
    if row is None:
        return {
            "recommended_player": None,
            "confidence": 0.0,
            "reasons": ["No recommendation is available."],
            "risks": [],
            "alternatives": [],
            "score_breakdown": {field: None for field in SCORE_FIELDS},
        }

    score_breakdown = build_score_breakdown(
        row.get("player_name"),
        recommendations_df=recommendations_df,
    )
    reasons = generate_recommendation_reasons(
        row.get("player_name"),
        recommendations_df=recommendations_df,
    )
    risk_impact = explain_risk_impact(row)
    alternatives = _top_alternatives(row.get("player_name"), recommendations_df)

    confidence_parts = [
        _safe_float(score_breakdown.get("overall_score"), 0.0),
        _safe_float(score_breakdown.get("need_score"), 0.0),
        _safe_float(score_breakdown.get("projection_score"), 0.0),
        _safe_float(score_breakdown.get("availability_score"), 0.0),
    ]
    confidence = round(sum(confidence_parts) / len(confidence_parts), 2)

    return {
        "recommended_player": {
            "player_name": row.get("player_name"),
            "position": row.get("position"),
            "team": row.get("team"),
        },
        "confidence": confidence,
        "reasons": reasons,
        "risks": risk_impact.get("risks", []),
        "alternatives": alternatives,
        "score_breakdown": score_breakdown,
        "position_need": explain_position_need(row),
        "tier_pressure": explain_tier_pressure(row),
        "opponent_pressure": explain_opponent_pressure(row),
        "team_fit": explain_team_fit(row),
        "projection_advantage": explain_projection_advantage(row),
        "risk_impact": risk_impact,
    }


def build_player_decision_cards(player_names, recommendations_df=None):
    """
    Build several decision cards while sharing the same recommendation dataframe.
    """
    if recommendations_df is not None:
        prime_recommendation_explainer_cache(recommendations_df=recommendations_df)

    return {
        player_name: build_player_decision_card(
            player_name,
            recommendations_df=recommendations_df,
        )
        for player_name in player_names
    }


def get_recommendation_explainer_debug_info():
    recommendations_df = _get_recommendations_df()
    sample_cards = []
    reason_counts = {}
    missing_dependencies = []

    if recommendations_df.empty:
        missing_dependencies.append("recommendation_engine")
    else:
        for player_name in recommendations_df.head(5)["player_name"].tolist():
            card = build_player_decision_card(player_name)
            sample_cards.append(card)
            for reason in card.get("reasons", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if "tier_urgency_df" not in _CACHE:
        _CACHE["tier_urgency_df"] = build_position_urgency_df()
    if "position_forecast_df" not in _CACHE:
        _CACHE["position_forecast_df"] = build_position_demand_forecast()

    tier_df = _CACHE["tier_urgency_df"]
    opponent_df = _CACHE["position_forecast_df"]
    projection_debug = get_projection_enrichment_debug_info()

    if tier_df.empty:
        missing_dependencies.append("tier_engine")
    if opponent_df.empty:
        missing_dependencies.append("opponent_model")
    if projection_debug.get("sportsbook_projection_row_count", 0) <= 0:
        missing_dependencies.append("sportsbook_projection_engine")

    component_availability = {}
    if not recommendations_df.empty:
        for score_name, column in COMPONENT_COLUMNS.items():
            component_availability[score_name] = {
                "column": column,
                "available": column in recommendations_df.columns,
                "non_null_count": int(recommendations_df[column].notna().sum())
                if column in recommendations_df.columns
                else 0,
            }
        component_availability["overall_score"] = {
            "column": "final_score",
            "available": "final_score" in recommendations_df.columns,
            "non_null_count": int(recommendations_df["final_score"].notna().sum())
            if "final_score" in recommendations_df.columns
            else 0,
        }

    return {
        "best_pick": get_best_pick_recommendation(),
        "recommendation_rows": int(len(recommendations_df)),
        "reason_counts": reason_counts,
        "missing_dependencies": missing_dependencies,
        "score_component_availability": component_availability,
        "sample_decision_cards": sample_cards,
        "tier_cliff": get_tier_cliff_recommendation(),
        "opponent_forecast_rows": int(len(opponent_df)),
        "projection_enrichment": projection_debug,
    }
