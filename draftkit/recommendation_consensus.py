from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _name_key
from draftkit.common import safe_float as _safe_float


DEFAULT_CONSENSUS_WEIGHTS = {
    "base_score": 0.25,
    "value_score": 0.20,
    "roster_fit_score": 0.20,
    "future_pick_impact_score": 0.15,
    "championship_equity_score": 0.10,
    "risk_score": 0.05,
    "survival_probability": 0.05,
}

RECOMMENDATION_CATEGORIES = [
    "BEST_OVERALL",
    "BEST_VALUE",
    "BEST_UPSIDE",
    "BEST_FLOOR",
    "BEST_CONSTRUCTION",
    "BEST_LEAGUE_WINNER",
]


@dataclass
class ConsensusScore:
    player_name: str
    position: str = ""
    team: str = ""
    base_score: float = 50.0
    value_score: float = 50.0
    roster_fit_score: float = 50.0
    risk_score: float = 50.0
    championship_equity_score: float = 50.0
    championship_equity: float = 50.0
    equity_delta: float = 0.0
    league_winning_upside: float = 50.0
    portfolio_classification: str = ""
    future_pick_impact_score: float = 50.0
    survival_probability: float = 50.0
    consensus_score: float = 0.0
    consensus_rank: int = 0
    models_top_5_count: int = 0
    average_model_rank: float = 0.0
    disagreement_score: float = 0.0
    confidence: float = 0.0
    categories: List[str] = None
    why_models_agree: List[str] = None
    key_risks: List[str] = None
    model_ranks: Dict[str, Optional[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["categories"] = data["categories"] or []
        data["why_models_agree"] = data["why_models_agree"] or []
        data["key_risks"] = data["key_risks"] or []
        data["model_ranks"] = data["model_ranks"] or {}
        return data


def _normalize_keyed_lookup(df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if df is None or df.empty or "player_name" not in df.columns:
        return {}

    return {
        _name_key(row.get("player_name")): row.to_dict()
        for _, row in df.iterrows()
        if _name_key(row.get("player_name"))
    }


def _score(value: Any, default: float = 50.0) -> float:
    return _clip_score(value, default)


def _safety_score(row: Dict[str, Any]) -> float:
    injury_risk = _safe_float(row.get("injury_risk"), 50.0)
    durability = _safe_float(row.get("durability_grade"), 70.0)
    archetype = str(row.get("archetype", "")).upper()

    raw_risk = injury_risk
    if archetype in {"RISKY", "BOOM"}:
        raw_risk += 12.0
    raw_risk += max(70.0 - durability, 0.0) * 0.35

    return _score(100.0 - raw_risk)


def _future_impact_score(row: Dict[str, Any]) -> float:
    quality = _safe_float(row.get("future_pick_quality"), None)
    roster_value = _safe_float(row.get("expected_roster_value"), None)
    construction = _safe_float(row.get("expected_construction_score"), None)

    parts = [part for part in [quality, roster_value, construction] if part is not None]
    if not parts:
        return 50.0
    return _score(sum(parts) / len(parts))


def _equity_delta_score(value: Any) -> float:
    delta = _safe_float(value, 0.0)
    return _score(50.0 + (delta * 7.5))


def _championship_component_score(equity_row: Dict[str, Any], future_row: Dict[str, Any]) -> float:
    equity = _safe_float(
        equity_row.get(
            "championship_equity",
            equity_row.get(
                "championship_probability",
                equity_row.get(
                    "championship_equity_score",
                    future_row.get("expected_championship_equity", 50.0),
                ),
            ),
        ),
        50.0,
    )
    equity_delta = _safe_float(equity_row.get("equity_delta"), 0.0)
    upside = _safe_float(equity_row.get("league_winning_upside"), 50.0)

    if "equity_delta" in equity_row or "league_winning_upside" in equity_row:
        return _score(
            (_score(equity) * 0.55)
            + (_equity_delta_score(equity_delta) * 0.25)
            + (_score(upside) * 0.20)
        )

    return _score(equity)


def _rank_lookup(
    players: Iterable[str],
    score_lookup: Dict[str, float],
    reverse: bool = True,
) -> Dict[str, int]:
    ordered = sorted(
        list(players),
        key=lambda key: score_lookup.get(key, 50.0),
        reverse=reverse,
    )
    return {key: rank for rank, key in enumerate(ordered, start=1)}


def _rank_stddev(ranks: List[int]) -> float:
    if len(ranks) <= 1:
        return 0.0

    avg = sum(ranks) / len(ranks)
    variance = sum((rank - avg) ** 2 for rank in ranks) / len(ranks)
    return variance ** 0.5


def _confidence_from_signals(
    score: ConsensusScore,
    player_count: int,
    simulation_row: Optional[Dict[str, Any]] = None,
) -> float:
    if player_count <= 1:
        return 70.0

    agreement = 100.0 - score.disagreement_score

    consensus_rank = max(score.consensus_rank, 1)
    equity_rank = score.model_ranks.get("championship_equity_score") if score.model_ranks else None
    if equity_rank is None:
        equity_alignment = 62.0
    else:
        equity_alignment = 100.0 - min(abs(equity_rank - consensus_rank) / max(player_count - 1, 1), 1.0) * 100.0

    sim_stability = 60.0
    if simulation_row:
        simulations_run = _safe_float(simulation_row.get("simulations_run"), 0.0)
        if simulations_run >= 100:
            sim_stability += 12.0
        elif simulations_run >= 50:
            sim_stability += 7.0

        details = simulation_row.get("target_survival_details")
        if isinstance(details, list) and details:
            probabilities = [
                _safe_float(target.get("survival_probability"), None)
                for target in details
            ]
            probabilities = [value for value in probabilities if value is not None]
            if len(probabilities) >= 2:
                avg = sum(probabilities) / len(probabilities)
                volatility = (
                    sum((value - avg) ** 2 for value in probabilities) / len(probabilities)
                ) ** 0.5
                sim_stability -= min(volatility * 0.35, 18.0)

    confidence = (agreement * 0.48) + (equity_alignment * 0.27) + (sim_stability * 0.25)
    if score.models_top_5_count >= 3:
        confidence += 6.0
    elif score.models_top_5_count <= 1 and player_count >= 5:
        confidence -= 8.0

    return _score(confidence)


def _agreement_reasons(score: ConsensusScore) -> List[str]:
    reasons = []
    ranks = score.model_ranks or {}

    if score.models_top_5_count >= 4:
        reasons.append("Most models rank this player inside their top five.")
    elif score.models_top_5_count >= 2:
        reasons.append("Multiple models keep this player near the top of the board.")

    if score.base_score >= 70 and score.value_score >= 65:
        reasons.append("Base recommendation and value signals are both strong.")
    if score.roster_fit_score >= 70:
        reasons.append("Roster construction supports the pick.")
    if ranks.get("championship_equity_score") and ranks["championship_equity_score"] <= max(score.consensus_rank + 2, 3):
        reasons.append("Championship equity is aligned with the consensus rank.")
    if score.equity_delta > 0:
        reasons.append("V2 championship simulation improves title probability.")
    if score.league_winning_upside >= 70:
        reasons.append("League-winning upside is a clear positive signal.")
    if score.future_pick_impact_score >= 65:
        reasons.append("Future-pick simulations do not meaningfully punish the selection.")

    return reasons[:4] or ["Consensus is driven by the weighted blend of available model signals."]


def _key_risks(score: ConsensusScore) -> List[str]:
    risks = []

    if score.disagreement_score >= 45:
        risks.append("Models disagree on the player's rank.")
    if score.risk_score <= 40:
        risks.append("Risk model flags elevated volatility or durability concern.")
    if score.survival_probability >= 70:
        risks.append("Player may be available later, reducing urgency.")
    if score.future_pick_impact_score <= 40:
        risks.append("Monte Carlo paths suggest weaker future-board outcomes.")
    if score.championship_equity_score <= 40:
        risks.append("Championship equity signal is below the candidate pool average.")
    if score.equity_delta < 0:
        risks.append("V2 simulation lowers title probability versus the current baseline.")
    if score.portfolio_classification in {"AGGRESSIVE", "HIGH_VARIANCE"}:
        risks.append(f"Portfolio shifts to {score.portfolio_classification.replace('_', ' ').title()} after the pick.")

    return risks[:4] or ["No major consensus-level risk flags."]


def _apply_categories(rows: List[ConsensusScore]) -> None:
    if not rows:
        return

    category_winners = {
        "BEST_OVERALL": max(rows, key=lambda row: row.consensus_score),
        "BEST_VALUE": max(rows, key=lambda row: row.value_score),
        "BEST_UPSIDE": max(
            rows,
            key=lambda row: (
                row.league_winning_upside * 0.45
                + row.championship_equity_score * 0.35
                + row.future_pick_impact_score * 0.15
                + row.value_score * 0.10
            ),
        ),
        "BEST_FLOOR": max(
            rows,
            key=lambda row: (
                row.risk_score * 0.50
                + (100.0 - min(row.disagreement_score, 100.0)) * 0.30
                + row.base_score * 0.20
            ),
        ),
        "BEST_CONSTRUCTION": max(rows, key=lambda row: row.roster_fit_score),
        "BEST_LEAGUE_WINNER": max(
            rows,
            key=lambda row: (
                row.championship_equity_score * 0.45
                + row.league_winning_upside * 0.30
                + _equity_delta_score(row.equity_delta) * 0.15
                + row.base_score * 0.10
            ),
        ),
    }

    for category, winner in category_winners.items():
        winner.categories.append(category)


def _resolve_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    resolved = DEFAULT_CONSENSUS_WEIGHTS.copy()
    if weights:
        for key, value in weights.items():
            if key in resolved:
                resolved[key] = max(_safe_float(value, resolved[key]), 0.0)

    total = sum(resolved.values())
    if total <= 0:
        return DEFAULT_CONSENSUS_WEIGHTS.copy()
    return {key: value / total for key, value in resolved.items()}


def build_consensus_recommendations(
    recommendations_df: Optional[pd.DataFrame],
    availability_df: Optional[pd.DataFrame] = None,
    future_impact_df: Optional[pd.DataFrame] = None,
    championship_equity_df: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Combine the recommendation, construction, simulation, value, risk, and equity
    outputs into a structured top-level decision layer.
    """
    if recommendations_df is None or recommendations_df.empty:
        empty_df = pd.DataFrame()
        return {
            "top_recommendation": None,
            "categories": {category: None for category in RECOMMENDATION_CATEGORIES},
            "consensus_df": empty_df,
            "consensus_scores": [],
            "weights": _resolve_weights(weights),
        }

    source_df = recommendations_df.copy()
    if limit is not None:
        source_df = source_df.head(max(int(limit), 0)).copy()

    availability_lookup = _normalize_keyed_lookup(availability_df)
    future_lookup = _normalize_keyed_lookup(future_impact_df)
    equity_lookup = _normalize_keyed_lookup(championship_equity_df)
    resolved_weights = _resolve_weights(weights)

    rows: List[ConsensusScore] = []
    component_values: Dict[str, Dict[str, float]] = {
        key: {} for key in DEFAULT_CONSENSUS_WEIGHTS
    }

    for _, rec in source_df.iterrows():
        rec_row = rec.to_dict()
        player_name = str(rec_row.get("player_name", "")).strip()
        player_key = _name_key(player_name)
        if not player_key:
            continue

        availability_row = availability_lookup.get(player_key, {})
        future_row = future_lookup.get(player_key, {})
        equity_row = equity_lookup.get(player_key, {})

        base_score = _score(rec_row.get("final_score"))
        value_score = _score(
            rec_row.get("adp_value_component_score", rec_row.get("value_score"))
        )
        roster_fit_score = _score(
            rec_row.get(
                "construction_pressure_score",
                rec_row.get("team_fit_component_score", rec_row.get("team_fit_bonus")),
            )
        )
        risk_score = _safety_score(rec_row)
        championship_equity = _score(
            equity_row.get(
                "championship_equity",
                equity_row.get(
                    "championship_probability",
                    equity_row.get("championship_equity_score", 50.0),
                ),
            )
        )
        equity_delta = round(_safe_float(equity_row.get("equity_delta"), 0.0), 2)
        league_winning_upside = _score(equity_row.get("league_winning_upside", 50.0))
        championship_equity_score = _championship_component_score(equity_row, future_row)
        future_pick_impact_score = _future_impact_score(future_row)
        survival_probability = _score(
            availability_row.get(
                "availability_probability",
                future_row.get("survival_probability", 50.0),
            )
        )

        consensus_score = round(
            (base_score * resolved_weights["base_score"])
            + (value_score * resolved_weights["value_score"])
            + (roster_fit_score * resolved_weights["roster_fit_score"])
            + (risk_score * resolved_weights["risk_score"])
            + (championship_equity_score * resolved_weights["championship_equity_score"])
            + (future_pick_impact_score * resolved_weights["future_pick_impact_score"])
            + (survival_probability * resolved_weights["survival_probability"]),
            2,
        )

        score = ConsensusScore(
            player_name=player_name,
            position=str(rec_row.get("position", "")),
            team=str(rec_row.get("team", "")),
            base_score=base_score,
            value_score=value_score,
            roster_fit_score=roster_fit_score,
            risk_score=risk_score,
            championship_equity_score=championship_equity_score,
            championship_equity=championship_equity,
            equity_delta=equity_delta,
            league_winning_upside=league_winning_upside,
            portfolio_classification=str(equity_row.get("portfolio_classification", "")),
            future_pick_impact_score=future_pick_impact_score,
            survival_probability=survival_probability,
            consensus_score=consensus_score,
            categories=[],
            why_models_agree=[],
            key_risks=[],
            model_ranks={},
        )
        rows.append(score)

        for metric in component_values:
            component_values[metric][player_key] = getattr(score, metric)

    if not rows:
        return {
            "top_recommendation": None,
            "categories": {category: None for category in RECOMMENDATION_CATEGORIES},
            "consensus_df": pd.DataFrame(),
            "consensus_scores": [],
            "weights": resolved_weights,
        }

    keys_by_player = {_name_key(row.player_name): row for row in rows}
    rank_lookups = {
        metric: _rank_lookup(keys_by_player.keys(), values)
        for metric, values in component_values.items()
    }

    rows = sorted(rows, key=lambda row: row.consensus_score, reverse=True)
    for consensus_rank, score in enumerate(rows, start=1):
        score.consensus_rank = consensus_rank
        key = _name_key(score.player_name)
        ranks = {
            metric: lookup.get(key)
            for metric, lookup in rank_lookups.items()
        }
        valid_ranks = [rank for rank in ranks.values() if rank is not None]
        score.model_ranks = ranks
        score.models_top_5_count = sum(1 for rank in valid_ranks if rank <= 5)
        score.average_model_rank = round(sum(valid_ranks) / len(valid_ranks), 2) if valid_ranks else 0.0
        rank_spread = _rank_stddev(valid_ranks)
        score.disagreement_score = _score(
            (rank_spread / max(len(rows) - 1, 1)) * 100.0,
            default=0.0,
        )
        score.confidence = _confidence_from_signals(
            score,
            player_count=len(rows),
            simulation_row=future_lookup.get(key),
        )
        score.why_models_agree = _agreement_reasons(score)
        score.key_risks = _key_risks(score)

    _apply_categories(rows)

    score_dicts = [row.to_dict() for row in rows]
    categories = {
        category: next(
            (row.to_dict() for row in rows if category in row.categories),
            None,
        )
        for category in RECOMMENDATION_CATEGORIES
    }

    return {
        "top_recommendation": rows[0].to_dict(),
        "categories": categories,
        "consensus_df": pd.DataFrame(score_dicts),
        "consensus_scores": score_dicts,
        "weights": resolved_weights,
    }
