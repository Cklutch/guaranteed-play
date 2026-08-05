from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from draftkit.common import clip_score as _clip_score
from draftkit.common import name_key as _name_key
from draftkit.common import safe_float as _safe_float
from draftkit.common import safe_int as _safe_int


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
VOLATILITY_COLS = ["volatility_score", "volatility", "std_dev", "projection_variance"]
INJURY_RISK_COLS = ["injury_risk", "Injury Risk", "injury_score", "risk_score"]
AGE_COLS = ["age", "Age"]
ROLE_UNCERTAINTY_COLS = [
    "role_uncertainty",
    "role_uncertainty_score",
    "uncertainty_score",
    "stability_score",
]
CEILING_COLS = ["ceiling_projection", "ceiling", "Ceiling"]
FLOOR_COLS = ["floor_projection", "floor", "Floor"]
ADP_COLS = ["adp", "ADP", "consensus_adp", "rank", "Rank"]
ARCHETYPE_COLS = ["archetype", "Archetype", "player_archetype", "profile"]

DEFAULT_ROSTER_SETTINGS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "BENCH": 6,
}

POSITION_REPLACEMENT_POINTS = {
    "QB": 250.0,
    "RB": 150.0,
    "WR": 155.0,
    "TE": 105.0,
}

POSITION_SCARCITY_MULTIPLIERS = {
    "QB": 0.85,
    "RB": 1.08,
    "WR": 1.02,
    "TE": 1.12,
}


def _safe_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


@dataclass
class PlayerOutcomeDistribution:
    player_name: str
    position: str
    projection: float
    floor_outcome: float
    median_outcome: float
    ceiling_outcome: float
    volatility: float
    injury_risk: float
    age_risk: float
    role_uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TeamOutcome:
    expected_season_points: float
    playoff_qualification_probability: float
    playoff_advancement_probability: float
    championship_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "player": _safe_col(df, PLAYER_COLS),
        "position": _safe_col(df, POSITION_COLS),
        "team": _safe_col(df, TEAM_COLS),
        "projection": _safe_col(df, PROJECTION_COLS),
        "volatility": _safe_col(df, VOLATILITY_COLS),
        "injury_risk": _safe_col(df, INJURY_RISK_COLS),
        "age": _safe_col(df, AGE_COLS),
        "role_uncertainty": _safe_col(df, ROLE_UNCERTAINTY_COLS),
        "ceiling": _safe_col(df, CEILING_COLS),
        "floor": _safe_col(df, FLOOR_COLS),
        "adp": _safe_col(df, ADP_COLS),
        "archetype": _safe_col(df, ARCHETYPE_COLS),
    }


def _row_value(row: Dict[str, Any], columns: Dict[str, Optional[str]], key: str, default=None):
    col = columns.get(key)
    if not col:
        return default
    value = row.get(col, default)
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _age_risk(position: str, age: Optional[float]) -> float:
    if age is None:
        return 35.0

    position = str(position).upper()
    if position == "RB":
        if age <= 25:
            return 18.0
        if age <= 27:
            return 34.0
        if age <= 29:
            return 58.0
        return 75.0
    if position == "WR":
        if age <= 27:
            return 18.0
        if age <= 30:
            return 36.0
        return 62.0
    if position == "TE":
        if age <= 29:
            return 22.0
        if age <= 32:
            return 40.0
        return 62.0
    if position == "QB":
        if age <= 32:
            return 16.0
        if age <= 36:
            return 35.0
        return 58.0
    return 35.0


def _role_uncertainty(row: Dict[str, Any], columns: Dict[str, Optional[str]]) -> float:
    raw = _row_value(row, columns, "role_uncertainty")
    if raw is None:
        archetype = str(_row_value(row, columns, "archetype", "")).upper()
        if archetype in {"RISKY", "BOOM", "UPSIDE"}:
            return 58.0
        if archetype in {"SAFE", "STEADY"}:
            return 28.0
        return 42.0

    score = _safe_float(raw, 42.0)
    stability_col = columns.get("role_uncertainty")
    if stability_col and "stability" in stability_col.lower():
        return _clip_score(100.0 - score)
    return _clip_score(score)


def derive_player_distribution(
    player_row: Dict[str, Any],
    columns: Optional[Dict[str, Optional[str]]] = None,
) -> PlayerOutcomeDistribution:
    columns = columns or {}
    player_name = str(_row_value(player_row, columns, "player", player_row.get("player_name", "")))
    position = str(_row_value(player_row, columns, "position", player_row.get("position", ""))).upper()
    projection = max(_safe_float(_row_value(player_row, columns, "projection", player_row.get("projection_points")), 0.0), 0.0)
    injury_risk = _clip_score(_row_value(player_row, columns, "injury_risk", player_row.get("injury_risk", 40.0)))
    age = _safe_float(_row_value(player_row, columns, "age"), None)
    age_risk = _age_risk(position, age)
    role_uncertainty = _role_uncertainty(player_row, columns)
    volatility = _safe_float(_row_value(player_row, columns, "volatility"), None)

    if volatility is None:
        volatility = 28.0 + (injury_risk * 0.22) + (role_uncertainty * 0.20) + (age_risk * 0.10)
    volatility = _clip_score(volatility)

    floor = _safe_float(_row_value(player_row, columns, "floor"), None)
    ceiling = _safe_float(_row_value(player_row, columns, "ceiling"), None)
    uncertainty_width = 0.16 + (volatility / 100.0 * 0.28)
    downside_width = uncertainty_width + (injury_risk / 100.0 * 0.16)
    upside_width = uncertainty_width + ((100.0 - role_uncertainty) / 100.0 * 0.08)

    if floor is None:
        floor = projection * max(0.42, 1.0 - downside_width)
    if ceiling is None:
        ceiling = projection * (1.0 + upside_width)

    median = projection * (1.0 - (injury_risk * 0.0015) - (age_risk * 0.0008))

    return PlayerOutcomeDistribution(
        player_name=player_name,
        position=position,
        projection=round(projection, 2),
        floor_outcome=round(max(floor, 0.0), 2),
        median_outcome=round(max(median, 0.0), 2),
        ceiling_outcome=round(max(ceiling, median, floor, 0.0), 2),
        volatility=round(volatility, 2),
        injury_risk=round(injury_risk, 2),
        age_risk=round(age_risk, 2),
        role_uncertainty=round(role_uncertainty, 2),
    )


def _sample_distribution(distribution: PlayerOutcomeDistribution, rng: random.Random) -> Dict[str, Any]:
    roll = rng.random()
    injury_roll = rng.random() * 100.0
    breakout_roll = rng.random() * 100.0
    underperformance_roll = rng.random() * 100.0

    injury_hit = injury_roll < distribution.injury_risk * 0.22
    breakout_hit = breakout_roll < max(5.0, 28.0 - distribution.role_uncertainty * 0.12)
    underperformance_hit = underperformance_roll < (distribution.volatility * 0.22 + distribution.role_uncertainty * 0.10)

    if injury_hit:
        outcome = distribution.floor_outcome * rng.uniform(0.65, 0.95)
    elif breakout_hit:
        outcome = rng.uniform(distribution.median_outcome, distribution.ceiling_outcome)
    elif underperformance_hit:
        outcome = rng.uniform(distribution.floor_outcome, distribution.median_outcome)
    elif roll < 0.18:
        outcome = rng.uniform(distribution.floor_outcome, distribution.median_outcome)
    elif roll > 0.84:
        outcome = rng.uniform(distribution.median_outcome, distribution.ceiling_outcome)
    else:
        center_noise = rng.uniform(-0.06, 0.06) * distribution.projection
        outcome = distribution.median_outcome + center_noise

    return {
        "points": max(outcome, 0.0),
        "injury_hit": injury_hit,
        "breakout_hit": breakout_hit,
        "underperformance_hit": underperformance_hit,
    }


def _counts_by_position(distributions: List[PlayerOutcomeDistribution]) -> Dict[str, int]:
    counts = {position: 0 for position in SCORABLE_POSITIONS}
    for distribution in distributions:
        if distribution.position in counts:
            counts[distribution.position] += 1
    return counts


def _projected_roster_points(
    distributions: List[PlayerOutcomeDistribution],
    roster_settings: Optional[Dict[str, Any]] = None,
    draft_round: int = 1,
) -> float:
    roster_settings = {**DEFAULT_ROSTER_SETTINGS, **(roster_settings or {})}
    counts = _counts_by_position(distributions)
    round_discount = max(0.55, 1.0 - max(draft_round - 1, 0) * 0.035)
    projected = 0.0

    for position in SCORABLE_POSITIONS:
        target = _safe_int(roster_settings.get(position), DEFAULT_ROSTER_SETTINGS.get(position, 1))
        have = counts.get(position, 0)
        remaining = max(target - have, 0)
        projected += remaining * POSITION_REPLACEMENT_POINTS.get(position, 140.0) * round_discount

    flex_slots = _safe_int(roster_settings.get("FLEX"), 1)
    projected += max(flex_slots, 0) * 135.0 * round_discount
    return projected


def _playoff_probability(points: float, league_size: int = 12) -> float:
    playoff_teams = max(round(league_size / 2), 4)
    baseline = 1180.0 + (league_size - playoff_teams) * 18.0
    return 1.0 / (1.0 + math.exp(-(points - baseline) / 95.0))


def _advancement_probability(points: float, playoff_probability: float) -> float:
    spike_factor = 1.0 / (1.0 + math.exp(-(points - 1275.0) / 110.0))
    return playoff_probability * (0.34 + spike_factor * 0.34)


def _championship_probability(points: float, playoff_probability: float, advancement_probability: float) -> float:
    title_strength = 1.0 / (1.0 + math.exp(-(points - 1350.0) / 120.0))
    return playoff_probability * advancement_probability * (0.18 + title_strength * 0.46)


class TeamOutcomeSimulator:
    def __init__(
        self,
        current_roster: Optional[List[Dict[str, Any]]] = None,
        candidate_player: Optional[Dict[str, Any]] = None,
        draft_round: int = 1,
        projected_roster_construction: Optional[Dict[str, Any]] = None,
        player_columns: Optional[Dict[str, Optional[str]]] = None,
        league_size: int = 12,
        num_simulations: int = 250,
        seed: Optional[int] = None,
    ):
        self.current_roster = current_roster or []
        self.candidate_player = candidate_player
        self.draft_round = max(_safe_int(draft_round, 1), 1)
        self.projected_roster_construction = projected_roster_construction or {}
        self.player_columns = player_columns or {}
        self.league_size = max(_safe_int(league_size, 12), 2)
        self.num_simulations = max(_safe_int(num_simulations, 250), 1)
        self.seed = seed

    def _distributions(self) -> List[PlayerOutcomeDistribution]:
        roster_rows = list(self.current_roster)
        if self.candidate_player:
            roster_rows.append(self.candidate_player)
        return [
            derive_player_distribution(row, self.player_columns)
            for row in roster_rows
        ]

    def simulate(self) -> TeamOutcome:
        distributions = self._distributions()
        if not distributions:
            projected_points = _projected_roster_points([], self.projected_roster_construction, self.draft_round)
            playoff = _playoff_probability(projected_points, self.league_size)
            advancement = _advancement_probability(projected_points, playoff)
            championship = _championship_probability(projected_points, playoff, advancement)
            return TeamOutcome(
                expected_season_points=round(projected_points, 2),
                playoff_qualification_probability=round(playoff * 100.0, 2),
                playoff_advancement_probability=round(advancement * 100.0, 2),
                championship_probability=round(championship * 100.0, 2),
            )

        rng = random.Random(self.seed)
        season_points = []
        playoff_probs = []
        advancement_probs = []
        championship_probs = []

        for _ in range(self.num_simulations):
            sampled_points = 0.0
            for distribution in distributions:
                sampled_points += _sample_distribution(distribution, rng)["points"]

            sampled_points += _projected_roster_points(
                distributions,
                self.projected_roster_construction,
                self.draft_round,
            )
            playoff = _playoff_probability(sampled_points, self.league_size)
            advancement = _advancement_probability(sampled_points, playoff)
            championship = _championship_probability(sampled_points, playoff, advancement)

            season_points.append(sampled_points)
            playoff_probs.append(playoff)
            advancement_probs.append(advancement)
            championship_probs.append(championship)

        return TeamOutcome(
            expected_season_points=round(sum(season_points) / len(season_points), 2),
            playoff_qualification_probability=round(sum(playoff_probs) / len(playoff_probs) * 100.0, 2),
            playoff_advancement_probability=round(sum(advancement_probs) / len(advancement_probs) * 100.0, 2),
            championship_probability=round(sum(championship_probs) / len(championship_probs) * 100.0, 2),
        )


def _roster_rows_from_names(players_df: pd.DataFrame, roster_names: Optional[List[Any]]) -> List[Dict[str, Any]]:
    if players_df.empty or not roster_names:
        return []

    columns = _resolve_columns(players_df)
    player_col = columns.get("player")
    if not player_col:
        return []

    roster_keys = {_name_key(name) for name in roster_names}
    roster_df = players_df[
        players_df[player_col].astype(str).map(_name_key).isin(roster_keys)
    ].copy()
    return roster_df.to_dict("records")


def _portfolio_classification(distributions: List[PlayerOutcomeDistribution]) -> str:
    if not distributions:
        return "BALANCED"

    avg_volatility = sum(player.volatility for player in distributions) / len(distributions)
    avg_injury = sum(player.injury_risk for player in distributions) / len(distributions)
    high_variance_count = sum(
        1 for player in distributions
        if player.volatility >= 65 or player.injury_risk >= 65 or player.role_uncertainty >= 68
    )
    high_variance_share = high_variance_count / len(distributions)

    if avg_volatility >= 68 or avg_injury >= 62 or high_variance_share >= 0.45:
        return "HIGH_VARIANCE"
    if avg_volatility >= 55 or high_variance_share >= 0.30:
        return "AGGRESSIVE"
    if avg_volatility <= 38 and avg_injury <= 38 and high_variance_share <= 0.15:
        return "SAFE"
    return "BALANCED"


def _league_winning_upside(
    distribution: PlayerOutcomeDistribution,
    roster_distributions: List[PlayerOutcomeDistribution],
    candidate_row: Dict[str, Any],
    columns: Dict[str, Optional[str]],
) -> float:
    replacement = POSITION_REPLACEMENT_POINTS.get(distribution.position, 145.0)
    ceiling_contribution = max(distribution.ceiling_outcome - replacement, 0.0)
    spike_potential = max(distribution.ceiling_outcome - distribution.median_outcome, 0.0)
    positional_advantage = max(distribution.median_outcome - replacement, 0.0)
    scarcity = POSITION_SCARCITY_MULTIPLIERS.get(distribution.position, 1.0)
    roster_same_position = sum(
        1 for player in roster_distributions
        if player.position == distribution.position
    )
    scarcity_bonus = scarcity * max(1.0, 2.6 - roster_same_position * 0.35)
    adp = _safe_float(_row_value(candidate_row, columns, "adp"), None)
    late_value_bonus = 6.0 if adp is not None and adp >= 60 else 0.0

    raw_score = (
        ceiling_contribution * 0.22
        + spike_potential * 0.28
        + positional_advantage * 0.18
        + scarcity_bonus * 12.0
        + late_value_bonus
    )
    return _clip_score(raw_score)


def build_championship_equity_v2_df(
    players_df: Optional[pd.DataFrame],
    candidates_df: Optional[pd.DataFrame] = None,
    current_roster: Optional[List[Any]] = None,
    draft_round: int = 1,
    projected_roster_construction: Optional[Dict[str, Any]] = None,
    league_size: int = 12,
    num_simulations: int = 250,
    seed: Optional[int] = 53,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    columns = _resolve_columns(players_df)
    player_col = columns.get("player")
    position_col = columns.get("position")
    if not player_col or not position_col:
        return pd.DataFrame()

    candidates = candidates_df.copy() if candidates_df is not None else players_df.copy()
    if candidates.empty:
        return pd.DataFrame()
    if limit is not None:
        candidates = candidates.head(max(_safe_int(limit, 0), 0)).copy()

    current_roster_rows = _roster_rows_from_names(players_df, current_roster)
    baseline_simulator = TeamOutcomeSimulator(
        current_roster=current_roster_rows,
        candidate_player=None,
        draft_round=draft_round,
        projected_roster_construction=projected_roster_construction,
        player_columns=columns,
        league_size=league_size,
        num_simulations=num_simulations,
        seed=seed,
    )
    baseline_outcome = baseline_simulator.simulate()
    current_team_equity = baseline_outcome.championship_probability
    roster_distributions = [
        derive_player_distribution(row, columns)
        for row in current_roster_rows
    ]

    player_lookup = {
        _name_key(row.get(player_col)): row.to_dict()
        for _, row in players_df.iterrows()
        if _name_key(row.get(player_col))
    }

    rows = []
    for index, candidate in candidates.iterrows():
        candidate_row = candidate.to_dict()
        candidate_name = str(candidate_row.get("player_name", candidate_row.get(player_col, "")))
        canonical_row = player_lookup.get(_name_key(candidate_name), candidate_row)
        canonical_row = {**canonical_row, **candidate_row}
        distribution = derive_player_distribution(canonical_row, columns)

        simulator = TeamOutcomeSimulator(
            current_roster=current_roster_rows,
            candidate_player=canonical_row,
            draft_round=draft_round,
            projected_roster_construction=projected_roster_construction,
            player_columns=columns,
            league_size=league_size,
            num_simulations=num_simulations,
            seed=None if seed is None else seed + int(index) + 1,
        )
        outcome = simulator.simulate()
        after_distributions = roster_distributions + [distribution]
        equity_delta = outcome.championship_probability - current_team_equity

        rows.append({
            "player_name": distribution.player_name or candidate_name,
            "position": distribution.position,
            "team": str(_row_value(canonical_row, columns, "team", candidate_row.get("team", ""))),
            "floor_outcome": distribution.floor_outcome,
            "median_outcome": distribution.median_outcome,
            "ceiling_outcome": distribution.ceiling_outcome,
            "expected_season_points": outcome.expected_season_points,
            "playoff_qualification_probability": outcome.playoff_qualification_probability,
            "playoff_advancement_probability": outcome.playoff_advancement_probability,
            "championship_probability": outcome.championship_probability,
            "championship_equity": _clip_score(outcome.championship_probability),
            "championship_equity_score": _clip_score(outcome.championship_probability),
            "current_team_equity": round(current_team_equity, 2),
            "equity_delta": round(equity_delta, 2),
            "league_winning_upside": _league_winning_upside(
                distribution,
                after_distributions,
                canonical_row,
                columns,
            ),
            "portfolio_classification": _portfolio_classification(after_distributions),
            "volatility": distribution.volatility,
            "injury_risk": distribution.injury_risk,
            "age_risk": distribution.age_risk,
            "role_uncertainty": distribution.role_uncertainty,
            "simulations_run": max(_safe_int(num_simulations, 250), 1),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["championship_equity", "equity_delta", "league_winning_upside"],
        ascending=False,
    ).reset_index(drop=True)


def get_championship_equity_v2_debug_info(
    players_df: Optional[pd.DataFrame] = None,
    candidates_df: Optional[pd.DataFrame] = None,
    current_roster: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    if players_df is None or players_df.empty:
        return {
            "row_count": 0,
            "candidate_count": 0,
            "missing_inputs": ["player_data"],
            "sample_equity": [],
        }

    columns = _resolve_columns(players_df)
    missing_inputs = [key for key, col in columns.items() if col is None]
    equity_df = build_championship_equity_v2_df(
        players_df=players_df,
        candidates_df=candidates_df,
        current_roster=current_roster,
        num_simulations=250,
        limit=5,
    )

    return {
        "row_count": int(len(players_df)),
        "candidate_count": int(len(candidates_df)) if candidates_df is not None else int(len(players_df)),
        "detected_columns": columns,
        "missing_inputs": missing_inputs,
        "sample_equity": equity_df.head(5).to_dict("records") if not equity_df.empty else [],
    }
