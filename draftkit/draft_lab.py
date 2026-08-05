from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd

from draftkit.championship_equity_v2 import TeamOutcomeSimulator, derive_player_distribution
from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _name_key
from draftkit.common import safe_float as _safe_float
from draftkit.common import safe_int as _safe_int


SUPPORTED_LEAGUE_SIZES = [8, 10, 12, 14]
OPPONENT_ARCHETYPES = [
    "ADP Drafter",
    "Value Drafter",
    "RB Heavy",
    "WR Heavy",
    "Hero RB",
    "Zero RB",
    "Early QB",
    "Balanced",
]
USER_STRATEGIES = OPPONENT_ARCHETYPES
SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position", "POS"]
TEAM_COLS = ["team", "Team", "team_abbr", "TEAM"]
PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "projection_points",
    "Projection",
    "projected_points",
    "FPTS",
    "proj_points",
]
ADP_COLS = ["adp", "ADP", "consensus_adp", "rank", "Rank"]
INJURY_RISK_COLS = ["injury_risk", "Injury Risk", "injury_score", "risk_score"]
DEFAULT_ROSTER_SETTINGS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "DST": 1,
    "K": 1,
    "BENCH": 6,
}
GRADE_WEIGHTS = {
    "roster_strength": 0.30,
    "positional_balance": 0.20,
    "championship_equity": 0.20,
    "value_gained": 0.15,
    "risk_profile": 0.15,
}


def _safe_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "player_name": _safe_col(df, PLAYER_COLS),
        "position": _safe_col(df, POSITION_COLS),
        "team": _safe_col(df, TEAM_COLS),
        "projection_points": _safe_col(df, PROJECTION_COLS),
        "adp": _safe_col(df, ADP_COLS),
        "injury_risk": _safe_col(df, INJURY_RISK_COLS),
    }


def _grade_letter(score: Any) -> str:
    score = _safe_float(score, 0.0)
    if score >= 93:
        return "A"
    if score >= 85:
        return "B"
    if score >= 75:
        return "C"
    if score >= 65:
        return "D"
    return "F"


def _normalize_league_size(league_size: Any) -> int:
    league_size = _safe_int(league_size, 12)
    if league_size in SUPPORTED_LEAGUE_SIZES:
        return league_size
    return min(SUPPORTED_LEAGUE_SIZES, key=lambda size: abs(size - league_size))


def _roster_size(roster_settings: Optional[Dict[str, Any]] = None) -> int:
    settings = {**DEFAULT_ROSTER_SETTINGS, **(roster_settings or {})}
    return max(sum(_safe_int(value, 0) for value in settings.values()), 1)


def prepare_draft_pool(players_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    prepared_cols = {
        "player_name",
        "position",
        "projection_points",
        "adp",
        "injury_risk",
        "projection_score",
        "value_score",
        "risk_score",
        "player_key",
    }
    if prepared_cols.issubset(set(players_df.columns)):
        return players_df.copy().reset_index(drop=True)

    columns = _resolve_columns(players_df)
    player_col = columns["player_name"]
    position_col = columns["position"]
    projection_col = columns["projection_points"]
    if not player_col or not position_col:
        return pd.DataFrame()

    pool = pd.DataFrame({
        "player_name": players_df[player_col].astype(str),
        "position": players_df[position_col].astype(str).str.upper(),
        "team": players_df[columns["team"]].astype(str) if columns["team"] else "",
        "projection_points": pd.to_numeric(
            players_df[projection_col],
            errors="coerce",
        ) if projection_col else 0.0,
        "adp": pd.to_numeric(players_df[columns["adp"]], errors="coerce")
        if columns["adp"] else None,
        "injury_risk": pd.to_numeric(players_df[columns["injury_risk"]], errors="coerce")
        if columns["injury_risk"] else 40.0,
    })
    pool = pool[pool["position"].isin(SCORABLE_POSITIONS)].copy()
    pool["projection_points"] = pool["projection_points"].fillna(0.0)
    pool["adp"] = pool["adp"].fillna(pd.Series(pool.index + 1, index=pool.index))
    pool["injury_risk"] = pool["injury_risk"].fillna(40.0)
    pool = pool.drop_duplicates(subset=["player_name"]).copy()
    if pool.empty:
        return pool

    pool = pool.sort_values(
        ["adp", "projection_points"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    pool["projection_rank"] = pool["projection_points"].rank(
        ascending=False,
        method="first",
    )
    pool["adp_rank"] = pool["adp"].rank(ascending=True, method="first")
    max_projection = max(float(pool["projection_points"].max()), 1.0)
    pool["projection_score"] = (pool["projection_points"] / max_projection * 100.0).clip(0.0, 100.0)
    pool["value_score"] = (50.0 + (pool["adp_rank"] - pool["projection_rank"]) * 2.0).clip(0.0, 100.0)
    pool["risk_score"] = (100.0 - pool["injury_risk"]).clip(0.0, 100.0)
    pool["player_key"] = pool["player_name"].map(_name_key)
    return pool.reset_index(drop=True)


def build_snake_pick_order(league_size: int, rounds: int) -> List[int]:
    league_size = _normalize_league_size(league_size)
    rounds = max(_safe_int(rounds, 1), 1)
    order = []
    for round_number in range(1, rounds + 1):
        if round_number % 2 == 1:
            order.extend(range(1, league_size + 1))
        else:
            order.extend(range(league_size, 0, -1))
    return order


def _position_counts(roster: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {position: 0 for position in SCORABLE_POSITIONS}
    for player in roster:
        position = str(player.get("position", "")).upper()
        if position in counts:
            counts[position] += 1
    return counts


def _need_score(position: str, roster: List[Dict[str, Any]], roster_settings: Optional[Dict[str, Any]]) -> float:
    settings = {**DEFAULT_ROSTER_SETTINGS, **(roster_settings or {})}
    counts = _position_counts(roster)
    position = str(position).upper()
    target = _safe_int(settings.get(position), 0)
    have = counts.get(position, 0)
    if target <= 0:
        return 35.0
    if have < target:
        return 84.0 + min((target - have) * 5.0, 12.0)
    if position in ["RB", "WR", "TE"] and counts.get("RB", 0) + counts.get("WR", 0) + counts.get("TE", 0) < (
        _safe_int(settings.get("RB"), 2)
        + _safe_int(settings.get("WR"), 2)
        + _safe_int(settings.get("TE"), 1)
        + _safe_int(settings.get("FLEX"), 1)
    ):
        return 68.0
    return max(42.0 - max(have - target, 0) * 8.0, 12.0)


def _strategy_bonus(position: str, roster: List[Dict[str, Any]], round_number: int, strategy: str) -> float:
    strategy = str(strategy or "Balanced")
    counts = _position_counts(roster)
    position = str(position).upper()

    if strategy == "RB Heavy":
        return 24.0 if position == "RB" and round_number <= 7 else 0.0
    if strategy == "WR Heavy":
        return 24.0 if position == "WR" and round_number <= 7 else 0.0
    if strategy == "Hero RB":
        if position == "RB" and counts.get("RB", 0) == 0 and round_number <= 3:
            return 30.0
        if position == "RB" and round_number <= 6:
            return -18.0
        if position == "WR":
            return 10.0
    if strategy == "Zero RB":
        if position == "RB" and round_number <= 5:
            return -28.0
        if position in ["WR", "TE"]:
            return 12.0
    if strategy == "Early QB":
        if position == "QB" and counts.get("QB", 0) == 0 and round_number <= 5:
            return 28.0
        if position == "QB" and counts.get("QB", 0) > 0:
            return -30.0
    if strategy == "Value Drafter":
        return 0.0
    if strategy == "ADP Drafter":
        return 0.0
    return 5.0 if _need_score(position, roster, DEFAULT_ROSTER_SETTINGS) >= 80 else 0.0


def _score_candidate(
    row: Dict[str, Any],
    roster: List[Dict[str, Any]],
    pick_number: int,
    league_size: int,
    strategy: str,
    roster_settings: Optional[Dict[str, Any]] = None,
) -> float:
    round_number = ((max(pick_number, 1) - 1) // max(league_size, 1)) + 1
    position = row.get("position")
    adp = _safe_float(row.get("adp"), pick_number)
    adp_fit = _clip_score(100.0 - abs(adp - pick_number) * 2.0)
    need = _need_score(position, roster, roster_settings)
    projection = _safe_float(row.get("projection_score"), 50.0)
    value = _safe_float(row.get("value_score"), 50.0)
    risk = _safe_float(row.get("risk_score"), 60.0)
    bonus = _strategy_bonus(position, roster, round_number, strategy)

    if strategy == "ADP Drafter":
        score = adp_fit * 0.65 + projection * 0.20 + need * 0.15
    elif strategy == "Value Drafter":
        score = value * 0.45 + projection * 0.30 + need * 0.20 + risk * 0.05
    else:
        score = projection * 0.38 + need * 0.28 + value * 0.20 + risk * 0.08 + adp_fit * 0.06 + bonus

    return round(float(score), 2)


def _recommendation_for_pick(
    available_df: pd.DataFrame,
    roster: List[Dict[str, Any]],
    pick_number: int,
    league_size: int,
    roster_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if available_df.empty:
        return {}

    scored = []
    for _, row in available_df.head(48).iterrows():
        row_dict = row.to_dict()
        score = (
            _score_candidate(row_dict, roster, pick_number, league_size, "Value Drafter", roster_settings) * 0.45
            + _score_candidate(row_dict, roster, pick_number, league_size, "Balanced", roster_settings) * 0.35
            + _safe_float(row_dict.get("risk_score"), 60.0) * 0.20
        )
        scored.append((score, row_dict))

    if not scored:
        return {}

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    distribution = derive_player_distribution(best, {
        "player": "player_name",
        "position": "position",
        "team": "team",
        "projection": "projection_points",
        "injury_risk": "injury_risk",
        "adp": "adp",
    })
    equity_proxy = _clip_score(
        best_score * 0.45
        + max(distribution.ceiling_outcome - distribution.median_outcome, 0.0) * 0.35
        + _safe_float(best.get("value_score"), 50.0) * 0.20
    )
    confidence = _clip_score(58.0 + min(best_score - 50.0, 28.0))
    return {
        "player_name": best.get("player_name"),
        "position": best.get("position"),
        "consensus_score": _clip_score(best_score),
        "championship_equity": equity_proxy,
        "confidence": confidence,
    }


def _choose_player(
    available_df: pd.DataFrame,
    roster: List[Dict[str, Any]],
    pick_number: int,
    league_size: int,
    strategy: str,
    roster_settings: Optional[Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    if available_df.empty:
        return {}

    scored = []
    candidate_window = available_df.head(min(42, len(available_df))).copy()
    for _, row in candidate_window.iterrows():
        row_dict = row.to_dict()
        score = _score_candidate(row_dict, roster, pick_number, league_size, strategy, roster_settings)
        score += rng.uniform(-4.0, 4.0)
        scored.append((score, row_dict))

    scored.sort(key=lambda item: item[0], reverse=True)
    return dict(scored[0][1]) if scored else {}


def _opponent_archetypes(league_size: int, seed: Optional[int] = None) -> Dict[int, str]:
    rng = random.Random(seed)
    archetypes = OPPONENT_ARCHETYPES.copy()
    rng.shuffle(archetypes)
    return {
        slot: archetypes[(slot - 1) % len(archetypes)]
        for slot in range(1, league_size + 1)
    }


def _remove_player(available_df: pd.DataFrame, player_name: Any) -> pd.DataFrame:
    key = _name_key(player_name)
    return available_df[available_df["player_key"] != key].reset_index(drop=True)


def simulate_complete_draft(
    players_df: Optional[pd.DataFrame],
    draft_slot: int = 1,
    league_size: int = 12,
    strategy: str = "Balanced",
    roster_settings: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    league_size = _normalize_league_size(league_size)
    draft_slot = max(1, min(_safe_int(draft_slot, 1), league_size))
    roster_settings = {**DEFAULT_ROSTER_SETTINGS, **(roster_settings or {})}
    rounds = _roster_size(roster_settings)
    pool = prepare_draft_pool(players_df)
    if pool.empty:
        return {
            "final_roster": [],
            "draft_history": [],
            "recommendation_history": [],
            "draft_grade": _empty_grade(),
            "recommendation_failures": [],
            "league_size": league_size,
            "draft_slot": draft_slot,
            "strategy": strategy,
        }

    rng = random.Random(seed)
    pick_order = build_snake_pick_order(league_size, rounds)
    opponent_strategy = _opponent_archetypes(league_size, seed)
    rosters = {slot: [] for slot in range(1, league_size + 1)}
    draft_history = []
    recommendation_history = []
    available_df = pool.copy()

    for pick_index, slot in enumerate(pick_order, start=1):
        if available_df.empty:
            break

        is_user_pick = slot == draft_slot
        active_strategy = strategy if is_user_pick else opponent_strategy.get(slot, "Balanced")
        roster = rosters[slot]
        recommendation = _recommendation_for_pick(
            available_df,
            roster,
            pick_index,
            league_size,
            roster_settings,
        ) if is_user_pick else {}
        selected = _choose_player(
            available_df,
            roster,
            pick_index,
            league_size,
            active_strategy,
            roster_settings,
            rng,
        )
        if not selected:
            break

        rosters[slot].append(selected)
        available_df = _remove_player(available_df, selected.get("player_name"))
        record = {
            "pick_number": pick_index,
            "round": ((pick_index - 1) // league_size) + 1,
            "slot": slot,
            "manager": "You" if is_user_pick else f"Team {slot}",
            "strategy": active_strategy,
            "player_name": selected.get("player_name"),
            "position": selected.get("position"),
            "adp": selected.get("adp"),
            "projection_points": selected.get("projection_points"),
        }
        draft_history.append(record)

        if is_user_pick:
            recommended_player = recommendation.get("player_name")
            followed = _name_key(recommended_player) == _name_key(selected.get("player_name"))
            recommendation_history.append({
                "pick_number": pick_index,
                "round": record["round"],
                "recommendation_selected": selected.get("player_name") if followed else None,
                "recommendation_passed": recommended_player if not followed else None,
                "recommended_player": recommended_player,
                "selected_player": selected.get("player_name"),
                "consensus_score": recommendation.get("consensus_score"),
                "championship_equity": recommendation.get("championship_equity"),
                "recommendation_confidence": recommendation.get("confidence"),
                "recommendation_followed": followed,
            })

    final_roster = rosters[draft_slot]
    grade = grade_draft(
        final_roster=final_roster,
        recommendation_history=recommendation_history,
        draft_history=draft_history,
        league_size=league_size,
        roster_settings=roster_settings,
        seed=seed,
    )
    for item in recommendation_history:
        item["final_roster_impact"] = grade.get("grade_score")

    failures = identify_recommendation_failures(
        final_roster=final_roster,
        recommendation_history=recommendation_history,
        draft_history=draft_history,
        roster_settings=roster_settings,
    )

    return {
        "final_roster": final_roster,
        "draft_history": draft_history,
        "recommendation_history": recommendation_history,
        "draft_grade": grade,
        "recommendation_failures": failures,
        "league_size": league_size,
        "draft_slot": draft_slot,
        "strategy": strategy,
    }


def _empty_grade() -> Dict[str, Any]:
    return {
        "grade": "F",
        "grade_score": 0.0,
        "roster_strength": 0.0,
        "positional_balance": 0.0,
        "championship_equity": 0.0,
        "value_gained": 0.0,
        "risk_profile": 0.0,
        "portfolio_classification": "BALANCED",
    }


def grade_draft(
    final_roster: Optional[List[Dict[str, Any]]] = None,
    recommendation_history: Optional[List[Dict[str, Any]]] = None,
    draft_history: Optional[List[Dict[str, Any]]] = None,
    league_size: int = 12,
    roster_settings: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    final_roster = final_roster or []
    recommendation_history = recommendation_history or []
    draft_history = draft_history or []
    if not final_roster:
        return _empty_grade()

    projections = [_safe_float(player.get("projection_points"), 0.0) for player in final_roster]
    roster_strength = _clip_score((sum(projections) / max(len(final_roster), 1)) / 3.0)
    counts = _position_counts(final_roster)
    settings = {**DEFAULT_ROSTER_SETTINGS, **(roster_settings or {})}
    concentration_penalty = 0.0
    missing_penalty = 0.0
    for position in SCORABLE_POSITIONS:
        target = _safe_int(settings.get(position), 0)
        have = counts.get(position, 0)
        missing_penalty += max(target - have, 0) * 12.0
        concentration_penalty += max(have - target - 2, 0) * 10.0
    positional_balance = _clip_score(100.0 - missing_penalty - concentration_penalty)

    risk_profile = _clip_score(
        sum(100.0 - _safe_float(player.get("injury_risk"), 40.0) for player in final_roster)
        / max(len(final_roster), 1)
    )
    user_picks = [
        record for record in draft_history
        if record.get("manager") == "You"
    ]
    value_points = []
    for record in user_picks:
        adp = _safe_float(record.get("adp"), record.get("pick_number"))
        value_points.append(adp - _safe_float(record.get("pick_number"), adp))
    value_gained = _clip_score(50.0 + (sum(value_points) / max(len(value_points), 1)) * 2.0)

    outcome = TeamOutcomeSimulator(
        current_roster=final_roster,
        candidate_player=None,
        draft_round=max(_safe_int(len(final_roster), 1), 1),
        projected_roster_construction=settings,
        player_columns={
            "player": "player_name",
            "position": "position",
            "team": "team",
            "projection": "projection_points",
            "injury_risk": "injury_risk",
            "adp": "adp",
        },
        league_size=league_size,
        num_simulations=250,
        seed=seed,
    ).simulate()
    championship_equity = _clip_score(outcome.championship_probability)
    distributions = [
        derive_player_distribution(player, {
            "player": "player_name",
            "position": "position",
            "team": "team",
            "projection": "projection_points",
            "injury_risk": "injury_risk",
            "adp": "adp",
        })
        for player in final_roster
    ]
    avg_volatility = sum(item.volatility for item in distributions) / max(len(distributions), 1)
    portfolio_classification = (
        "HIGH_VARIANCE" if avg_volatility >= 68 else
        "AGGRESSIVE" if avg_volatility >= 55 else
        "SAFE" if avg_volatility <= 38 else
        "BALANCED"
    )

    components = {
        "roster_strength": roster_strength,
        "positional_balance": positional_balance,
        "championship_equity": championship_equity,
        "value_gained": value_gained,
        "risk_profile": risk_profile,
    }
    grade_score = _clip_score(
        sum(components[key] * weight for key, weight in GRADE_WEIGHTS.items())
        / sum(GRADE_WEIGHTS.values())
    )

    return {
        "grade": _grade_letter(grade_score),
        "grade_score": grade_score,
        **components,
        "portfolio_classification": portfolio_classification,
        "recommendation_follow_rate": round(
            sum(1 for item in recommendation_history if item.get("recommendation_followed"))
            / max(len(recommendation_history), 1)
            * 100.0,
            2,
        ),
    }


def identify_recommendation_failures(
    final_roster: Optional[List[Dict[str, Any]]] = None,
    recommendation_history: Optional[List[Dict[str, Any]]] = None,
    draft_history: Optional[List[Dict[str, Any]]] = None,
    roster_settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    final_roster = final_roster or []
    recommendation_history = recommendation_history or []
    draft_history = draft_history or []
    settings = {**DEFAULT_ROSTER_SETTINGS, **(roster_settings or {})}
    failures = []

    selected_lookup = {
        _name_key(record.get("player_name")): record
        for record in draft_history
        if record.get("manager") == "You"
    }
    for item in recommendation_history:
        if _safe_float(item.get("recommendation_confidence"), 100.0) < 45:
            failures.append({
                "type": "low-confidence recommendations",
                "pick_number": item.get("pick_number"),
                "player": item.get("recommended_player"),
                "severity": "medium",
            })
        if item.get("recommendation_passed"):
            selected = selected_lookup.get(_name_key(item.get("selected_player")), {})
            selected_projection = _safe_float(selected.get("projection_points"), 0.0)
            recommended_equity = _safe_float(item.get("championship_equity"), 0.0)
            if recommended_equity - selected_projection / 4.0 >= 18:
                failures.append({
                    "type": "large equity drops",
                    "pick_number": item.get("pick_number"),
                    "player": item.get("selected_player"),
                    "severity": "high",
                })

    counts = _position_counts(final_roster)
    for position, count in counts.items():
        target = _safe_int(settings.get(position), 0)
        if target and count > target + 3:
            failures.append({
                "type": "excessive positional concentration",
                "position": position,
                "count": count,
                "severity": "high",
            })

    for record in draft_history:
        if record.get("manager") != "You":
            continue
        adp = _safe_float(record.get("adp"), record.get("pick_number"))
        if _safe_float(record.get("pick_number"), adp) - adp >= 20:
            failures.append({
                "type": "poor value decisions",
                "pick_number": record.get("pick_number"),
                "player": record.get("player_name"),
                "severity": "medium",
            })

    return failures


def run_simulation_batch(
    players_df: Optional[pd.DataFrame],
    draft_slot: int = 1,
    league_size: int = 12,
    strategy: str = "Balanced",
    batch_size: int = 10,
    roster_settings: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = 101,
) -> Dict[str, Any]:
    batch_size = _safe_int(batch_size, 10)
    if batch_size not in [10, 50, 100, 500]:
        batch_size = min([10, 50, 100, 500], key=lambda value: abs(value - batch_size))

    if players_df is None or players_df.empty:
        return {
            "simulations_run": 0,
            "average_draft_grade": 0.0,
            "average_championship_equity": 0.0,
            "most_common_roster_builds": [],
            "recommendation_performance": {},
            "failure_counts": {},
            "drafts": [],
        }

    rng = random.Random(seed)
    prepared_pool = prepare_draft_pool(players_df)
    drafts = []
    for index in range(batch_size):
        drafts.append(
            simulate_complete_draft(
                players_df=prepared_pool,
                draft_slot=draft_slot,
                league_size=league_size,
                strategy=strategy,
                roster_settings=roster_settings,
                seed=rng.randint(1, 10_000_000) + index,
            )
        )

    grades = [draft.get("draft_grade", {}) for draft in drafts]
    builds = Counter()
    failure_counts = Counter()
    recommendation_records = []
    for draft in drafts:
        counts = _position_counts(draft.get("final_roster", []))
        build_key = "-".join(f"{position}{counts.get(position, 0)}" for position in SCORABLE_POSITIONS)
        builds[build_key] += 1
        recommendation_records.extend(draft.get("recommendation_history", []))
        for failure in draft.get("recommendation_failures", []):
            failure_counts[failure.get("type", "unknown")] += 1

    follow_rate = (
        sum(1 for record in recommendation_records if record.get("recommendation_followed"))
        / max(len(recommendation_records), 1)
        * 100.0
    )
    passed_count = sum(1 for record in recommendation_records if record.get("recommendation_passed"))

    return {
        "simulations_run": len(drafts),
        "average_draft_grade": round(
            sum(_safe_float(grade.get("grade_score"), 0.0) for grade in grades)
            / max(len(grades), 1),
            2,
        ),
        "average_championship_equity": round(
            sum(_safe_float(grade.get("championship_equity"), 0.0) for grade in grades)
            / max(len(grades), 1),
            2,
        ),
        "most_common_roster_builds": [
            {"build": build, "count": count}
            for build, count in builds.most_common(8)
        ],
        "recommendation_performance": {
            "recommendation_count": len(recommendation_records),
            "follow_rate": round(follow_rate, 2),
            "passed_count": passed_count,
            "average_consensus_score": round(
                sum(_safe_float(record.get("consensus_score"), 0.0) for record in recommendation_records)
                / max(len(recommendation_records), 1),
                2,
            ),
        },
        "failure_counts": dict(failure_counts),
        "drafts": drafts,
    }
