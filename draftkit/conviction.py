import pandas as pd

from draftkit.championship_equity import build_championship_equity_df
from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _normalize_name
from draftkit.common import safe_float as _safe_float
from draftkit.draft_analysis import build_master_recommendations_df


CONVICTION_LEVELS = ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]


def _conviction_level(score):
    score = _safe_float(score, 0.0)
    if score >= 82:
        return "VERY_HIGH"
    if score >= 68:
        return "HIGH"
    if score >= 48:
        return "MEDIUM"
    return "LOW"


def _conviction_message(level):
    messages = {
        "VERY_HIGH": "We strongly recommend this player.",
        "HIGH": "This player is clearly preferred.",
        "MEDIUM": "Several viable alternatives exist.",
        "LOW": "Multiple players are similarly valued.",
    }
    return messages.get(level, "Multiple players are similarly valued.")


def _get_recommendations_df(recommendations_df=None):
    if recommendations_df is not None:
        return recommendations_df.copy().reset_index(drop=True)

    return build_master_recommendations_df().reset_index(drop=True)


def _add_championship_equity(recommendations_df):
    if recommendations_df.empty:
        return recommendations_df.copy()

    out = recommendations_df.copy()
    if "championship_equity_score" in out.columns:
        return out

    try:
        equity_df = build_championship_equity_df()
    except Exception:
        equity_df = pd.DataFrame()

    if equity_df.empty or "player_name" not in equity_df.columns:
        out["championship_equity_score"] = pd.NA
        return out

    equity_subset = equity_df[["player_name", "championship_equity_score"]].copy()
    equity_subset["player_key"] = equity_subset["player_name"].apply(_normalize_name)
    equity_subset = equity_subset.drop_duplicates("player_key", keep="first")

    out["player_key"] = out["player_name"].apply(_normalize_name)
    out = out.merge(
        equity_subset[["player_key", "championship_equity_score"]],
        on="player_key",
        how="left",
    )
    return out.drop(columns=["player_key"])


def calculate_score_gap(recommendations_df, rank=0):
    if recommendations_df.empty or rank >= len(recommendations_df):
        return 0.0

    current_score = _safe_float(recommendations_df.iloc[rank].get("final_score"), 0.0)
    next_rank = rank + 1
    if next_rank >= len(recommendations_df):
        return 0.0

    next_score = _safe_float(recommendations_df.iloc[next_rank].get("final_score"), current_score)
    return round(max(current_score - next_score, 0.0), 2)


def calculate_alternative_strength(recommendations_df, rank=0, window=3):
    if recommendations_df.empty or rank >= len(recommendations_df):
        return 50.0

    current_score = _safe_float(recommendations_df.iloc[rank].get("final_score"), 0.0)
    alternatives = recommendations_df.iloc[rank + 1: rank + 1 + window]
    if alternatives.empty:
        return 50.0

    alt_scores = alternatives["final_score"].apply(_safe_float)
    average_gap = max(current_score - float(alt_scores.mean()), 0.0)
    # High score means alternatives are strong and conviction should fall.
    return _clip_score(100.0 - (average_gap * 8.0))


def calculate_position_gap(recommendations_df, rank=0):
    if recommendations_df.empty or rank >= len(recommendations_df):
        return 0.0

    row = recommendations_df.iloc[rank]
    position = str(row.get("position", "")).upper()
    current_score = _safe_float(row.get("final_score"), 0.0)
    same_position = recommendations_df[
        recommendations_df["position"].astype(str).str.upper() == position
    ].reset_index(drop=True)

    player_name = _normalize_name(row.get("player_name"))
    player_idx = None
    for idx, pos_row in same_position.iterrows():
        if _normalize_name(pos_row.get("player_name")) == player_name:
            player_idx = idx
            break

    if player_idx is None or player_idx + 1 >= len(same_position):
        return 0.0

    next_score = _safe_float(same_position.iloc[player_idx + 1].get("final_score"), current_score)
    return round(max(current_score - next_score, 0.0), 2)


def calculate_replacement_gap(recommendations_df, rank=0, replacement_rank=8):
    if recommendations_df.empty or rank >= len(recommendations_df):
        return 0.0

    current_score = _safe_float(recommendations_df.iloc[rank].get("final_score"), 0.0)
    replacement_idx = min(rank + replacement_rank, len(recommendations_df) - 1)
    replacement_score = _safe_float(
        recommendations_df.iloc[replacement_idx].get("final_score"),
        current_score,
    )
    return round(max(current_score - replacement_score, 0.0), 2)


def calculate_signal_confidence(recommendation_row):
    signal_trust = _safe_float(recommendation_row.get("signal_trust_score"), 75.0)
    championship_equity = _safe_float(
        recommendation_row.get("championship_equity_score"),
        50.0,
    )
    construction_pressure = _safe_float(
        recommendation_row.get("construction_pressure_score"),
        0.0,
    )
    trust_penalty = 0.0
    if recommendation_row.get("trust_adjustment_applied"):
        trust_penalty += 18.0

    signal_confidence = (
        (signal_trust * 0.60)
        + (championship_equity * 0.25)
        + (min(construction_pressure, 100.0) * 0.15)
        - trust_penalty
    )
    return _clip_score(signal_confidence)


def calculate_conviction_score(recommendations_df, rank=0):
    recommendations = _add_championship_equity(
        _get_recommendations_df(recommendations_df)
    )
    if recommendations.empty or rank >= len(recommendations):
        return 0.0

    row = recommendations.iloc[rank]
    score_gap = calculate_score_gap(recommendations, rank)
    alternative_strength = calculate_alternative_strength(recommendations, rank)
    position_gap = calculate_position_gap(recommendations, rank)
    replacement_gap = calculate_replacement_gap(recommendations, rank)
    signal_confidence = calculate_signal_confidence(row)

    gap_component = min(score_gap * 4.0, 36.0)
    alternative_component = max(0.0, 100.0 - alternative_strength) * 0.18
    position_component = min(position_gap * 1.5, 20.0)
    replacement_component = min(replacement_gap * 1.0, 20.0)
    signal_component = (signal_confidence - 50.0) * 0.45

    conviction = (
        15.0
        + gap_component
        + alternative_component
        + position_component
        + replacement_component
        + signal_component
    )
    conviction = _clip_score(conviction)

    if signal_confidence < 45:
        conviction = min(conviction, 65.0)
    if row.get("trust_adjustment_applied"):
        conviction = min(conviction, 72.0)

    return _clip_score(conviction)


def _primary_reasons(row):
    reasons = []
    if _safe_float(row.get("score_gap")) >= 6:
        reasons.append("Clear score separation from the next option")
    if _safe_float(row.get("position_gap")) >= 6:
        reasons.append("Meaningful positional drop after this player")
    if _safe_float(row.get("replacement_gap")) >= 10:
        reasons.append("Large value loss if skipped")
    if _safe_float(row.get("signal_confidence")) >= 75:
        reasons.append("Input signals are trustworthy")
    if _safe_float(row.get("construction_pressure_score")) >= 55:
        reasons.append("Addresses roster construction pressure")

    if not reasons:
        reasons.append(_conviction_message(row.get("conviction_level")))
    return reasons[:3]


def _secondary_reasons(row):
    reasons = []
    if _safe_float(row.get("alternative_strength")) >= 75:
        reasons.append("Nearby alternatives are strong")
    if _safe_float(row.get("signal_confidence")) < 55:
        reasons.append("Signal confidence is limited")
    if _safe_float(row.get("score_gap")) < 3:
        reasons.append("Small gap from the next recommendation")
    if row.get("trust_adjustment_applied"):
        reasons.append("Trust guardrail reduced this player's recommendation score")
    return reasons[:3]


def build_conviction_report(recommendations_df=None, limit=None):
    recommendations = _add_championship_equity(
        _get_recommendations_df(recommendations_df)
    )
    if recommendations.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in recommendations.reset_index(drop=True).iterrows():
        conviction_score = calculate_conviction_score(recommendations, idx)
        conviction_level = _conviction_level(conviction_score)
        output_row = {
            "rank": idx + 1,
            "player_name": row.get("player_name"),
            "position": row.get("position"),
            "team": row.get("team"),
            "recommendation_score": _safe_float(row.get("final_score"), 0.0),
            "conviction_score": conviction_score,
            "conviction_level": conviction_level,
            "conviction_message": _conviction_message(conviction_level),
            "score_gap": calculate_score_gap(recommendations, idx),
            "alternative_strength": calculate_alternative_strength(recommendations, idx),
            "position_gap": calculate_position_gap(recommendations, idx),
            "replacement_gap": calculate_replacement_gap(recommendations, idx),
            "signal_confidence": calculate_signal_confidence(row),
            "signal_trust_score": _safe_float(row.get("signal_trust_score"), 75.0),
            "championship_equity_score": _safe_float(
                row.get("championship_equity_score"),
                50.0,
            ),
            "construction_pressure_score": _safe_float(
                row.get("construction_pressure_score"),
                0.0,
            ),
            "trust_adjustment_applied": bool(row.get("trust_adjustment_applied", False)),
        }
        output_row["primary_reasons"] = _primary_reasons(output_row)
        output_row["secondary_reasons"] = _secondary_reasons(output_row)
        rows.append(output_row)

    report_df = pd.DataFrame(rows)
    if limit:
        return report_df.head(limit)
    return report_df


def get_top_recommendation_conviction(recommendations_df=None):
    report_df = build_conviction_report(recommendations_df, limit=1)
    if report_df.empty:
        return {
            "recommended_player": None,
            "conviction_level": "LOW",
            "conviction_score": 0.0,
            "primary_reasons": [],
            "secondary_reasons": [],
        }

    top = report_df.iloc[0].to_dict()
    return {
        "recommended_player": top.get("player_name"),
        "conviction_level": top.get("conviction_level"),
        "conviction_score": top.get("conviction_score"),
        "recommendation_score": top.get("recommendation_score"),
        "primary_reasons": top.get("primary_reasons", []),
        "secondary_reasons": top.get("secondary_reasons", []),
    }


def get_conviction_debug_info(recommendations_df=None):
    report_df = build_conviction_report(recommendations_df)
    if report_df.empty:
        return {
            "recommendation_count": 0,
            "average_conviction": 0.0,
            "highest_conviction_recommendations": [],
            "lowest_conviction_recommendations": [],
            "largest_score_gaps": [],
        }

    return {
        "recommendation_count": int(len(report_df)),
        "average_conviction": round(float(report_df["conviction_score"].mean()), 2),
        "highest_conviction_recommendations": report_df.sort_values(
            "conviction_score",
            ascending=False,
        ).head(10).to_dict("records"),
        "lowest_conviction_recommendations": report_df.sort_values(
            "conviction_score",
            ascending=True,
        ).head(10).to_dict("records"),
        "largest_score_gaps": report_df.sort_values(
            "score_gap",
            ascending=False,
        ).head(10).to_dict("records"),
    }
