import pandas as pd

from draftkit.common import clip_score as _clip_score
from draftkit.common import safe_float as _safe_float
from draftkit.data_access import load_players_df, safe_col


SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]

PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
ADP_COLS = ["adp", "ADP", "consensus_adp"]
ADP_RANK_COLS = ["adp_rank", "ADP Rank", "rank", "Rank"]
PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "projection_points",
    "Projection",
    "projected_points",
    "FPTS",
]
PROJECTION_RANK_COLS = ["projection_rank", "Projection Rank"]
AGE_COLS = ["age", "Age"]
CEILING_COLS = ["ceiling_projection", "ceiling", "Ceiling"]
FLOOR_COLS = ["floor_projection", "floor", "Floor"]
BOOM_COLS = ["boom_score", "Boom Score"]
BUST_COLS = ["bust_score", "Bust Score"]
STABILITY_COLS = ["stability_score", "Stability Score"]
ARCHETYPE_COLS = ["archetype", "Archetype", "player_archetype"]
SPORTSBOOK_PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "sportsbook_projection",
]

POSITION_CEILING_BASELINES = {
    "QB": 330.0,
    "RB": 255.0,
    "WR": 255.0,
    "TE": 190.0,
}

CHAMPIONSHIP_EQUITY_WEIGHTS = {
    "breakout_probability": 0.25,
    "adp_outperformance_score": 0.25,
    "ceiling_score": 0.25,
    "age_curve_score": 0.10,
    "sportsbook_advantage_score": 0.15,
}


def _get_value(player_row, columns, key, default=None):
    col = columns.get(key)
    if col and col in player_row:
        return player_row.get(col)
    return default


def _resolve_columns(df):
    return {
        "player": safe_col(df, PLAYER_COLS),
        "position": safe_col(df, POSITION_COLS),
        "adp": safe_col(df, ADP_COLS),
        "adp_rank": safe_col(df, ADP_RANK_COLS),
        "projection": safe_col(df, PROJECTION_COLS),
        "projection_rank": safe_col(df, PROJECTION_RANK_COLS),
        "age": safe_col(df, AGE_COLS),
        "ceiling": safe_col(df, CEILING_COLS),
        "floor": safe_col(df, FLOOR_COLS),
        "boom": safe_col(df, BOOM_COLS),
        "bust": safe_col(df, BUST_COLS),
        "stability": safe_col(df, STABILITY_COLS),
        "archetype": safe_col(df, ARCHETYPE_COLS),
        "sportsbook_projection": safe_col(df, SPORTSBOOK_PROJECTION_COLS),
    }


def calculate_age_curve_score(player_row, columns=None):
    columns = columns or {}
    position = str(_get_value(player_row, columns, "position", "")).upper()
    age = _safe_float(_get_value(player_row, columns, "age"))

    if age is None:
        return 50.0

    if position == "RB":
        if age <= 21:
            return 55.0
        if age <= 25:
            return 62.0
        if age <= 27:
            return 55.0
        if age <= 29:
            return 47.0
        return 40.0

    if position == "WR":
        if age <= 22:
            return 56.0
        if age <= 27:
            return 62.0
        if age <= 30:
            return 54.0
        return 44.0

    if position == "TE":
        if age <= 23:
            return 55.0
        if age <= 29:
            return 60.0
        if age <= 32:
            return 53.0
        return 44.0

    if position == "QB":
        if age <= 24:
            return 55.0
        if age <= 32:
            return 59.0
        if age <= 36:
            return 53.0
        return 45.0

    return 50.0


def calculate_adp_outperformance_score(player_row, columns=None):
    columns = columns or {}
    projection_rank = _safe_float(_get_value(player_row, columns, "projection_rank"))
    adp_rank = _safe_float(_get_value(player_row, columns, "adp_rank"))

    if projection_rank is None or adp_rank is None or projection_rank <= 0 or adp_rank <= 0:
        return 50.0

    market_discount = adp_rank - projection_rank
    return _clip_score(50.0 + (market_discount * 1.6))


def calculate_ceiling_score(player_row, columns=None):
    columns = columns or {}
    position = str(_get_value(player_row, columns, "position", "")).upper()
    projection = _safe_float(_get_value(player_row, columns, "projection"), 0.0)
    ceiling = _safe_float(_get_value(player_row, columns, "ceiling"))
    floor = _safe_float(_get_value(player_row, columns, "floor"))
    boom_score = _safe_float(_get_value(player_row, columns, "boom"), 50.0)

    if ceiling is None:
        if projection <= 0:
            return _clip_score(boom_score)
        ceiling = projection * 1.18

    baseline = POSITION_CEILING_BASELINES.get(position, 240.0)
    ceiling_component = 50.0 + ((ceiling / baseline) - 0.75) * 80.0

    spread_bonus = 0.0
    if floor is not None and ceiling > floor:
        spread_bonus = min(12.0, (ceiling - floor) / max(baseline, 1.0) * 45.0)

    boom_component = (boom_score - 50.0) * 0.25
    return _clip_score(ceiling_component + spread_bonus + boom_component)


def calculate_breakout_probability(player_row, columns=None):
    columns = columns or {}
    position = str(_get_value(player_row, columns, "position", "")).upper()
    adp_rank = _safe_float(_get_value(player_row, columns, "adp_rank"))
    age_score = calculate_age_curve_score(player_row, columns)
    boom_score = _safe_float(_get_value(player_row, columns, "boom"), 50.0)
    stability_score = _safe_float(_get_value(player_row, columns, "stability"), 50.0)
    archetype = str(_get_value(player_row, columns, "archetype", "")).upper()

    score = 42.0
    score += (age_score - 50.0) * 0.45
    score += (boom_score - 50.0) * 0.25
    score += max(0.0, 60.0 - stability_score) * 0.08

    if archetype in ["UPSIDE", "BOOM"]:
        score += 10.0
    elif archetype in ["RISKY"]:
        score += 4.0
    elif archetype in ["SAFE", "STEADY"]:
        score -= 3.0

    if adp_rank is not None:
        if 25 <= adp_rank <= 120:
            score += 9.0
        elif adp_rank > 120:
            score += 6.0
        elif adp_rank <= 12:
            score -= 6.0

    if position in ["RB", "WR"]:
        score += 3.0
    elif position == "TE":
        score += 1.0

    return _clip_score(score)


def calculate_sportsbook_advantage_score(player_row, columns=None):
    columns = columns or {}
    sportsbook_projection = _safe_float(
        _get_value(player_row, columns, "sportsbook_projection")
    )
    projection = _safe_float(_get_value(player_row, columns, "projection"))

    if sportsbook_projection is None or sportsbook_projection <= 0:
        return 50.0

    if projection is None or projection <= 0:
        return _clip_score(50.0 + min(25.0, sportsbook_projection / 12.0))

    edge_pct = (sportsbook_projection - projection) / max(abs(projection), 1.0)
    return _clip_score(50.0 + (edge_pct * 160.0))


def calculate_championship_equity_score(player_row, columns=None, weights=None):
    columns = columns or {}
    weights = weights or CHAMPIONSHIP_EQUITY_WEIGHTS

    components = {
        "breakout_probability": calculate_breakout_probability(player_row, columns),
        "adp_outperformance_score": calculate_adp_outperformance_score(player_row, columns),
        "ceiling_score": calculate_ceiling_score(player_row, columns),
        "age_curve_score": calculate_age_curve_score(player_row, columns),
        "sportsbook_advantage_score": calculate_sportsbook_advantage_score(player_row, columns),
    }

    total_weight = sum(max(_safe_float(weight, 0.0), 0.0) for weight in weights.values())
    if total_weight <= 0:
        return 50.0

    score = sum(
        components[name] * max(_safe_float(weights.get(name), 0.0), 0.0)
        for name in components
    ) / total_weight
    return _clip_score(score)


def build_championship_equity_df(players_df=None):
    df = players_df.copy() if players_df is not None else load_players_df().copy()
    if df.empty:
        return pd.DataFrame()

    columns = _resolve_columns(df)
    player_col = columns["player"]
    position_col = columns["position"]

    if player_col is None or position_col is None:
        return pd.DataFrame()

    working_df = df.copy()
    working_df["position"] = working_df[position_col].astype(str).str.upper()
    working_df = working_df[working_df["position"].isin(SCORABLE_POSITIONS)].copy()
    if working_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in working_df.iterrows():
        projection = _safe_float(_get_value(row, columns, "projection"))
        adp = _safe_float(_get_value(row, columns, "adp"))

        output_row = {
            "player_name": row.get(player_col),
            "position": row.get("position"),
            "adp": adp,
            "projection_points": projection,
            "breakout_probability": calculate_breakout_probability(row, columns),
            "adp_outperformance_score": calculate_adp_outperformance_score(row, columns),
            "ceiling_score": calculate_ceiling_score(row, columns),
            "age_curve_score": calculate_age_curve_score(row, columns),
            "sportsbook_advantage_score": calculate_sportsbook_advantage_score(row, columns),
        }
        output_row["championship_equity_score"] = calculate_championship_equity_score(
            row,
            columns,
        )
        rows.append(output_row)

    equity_df = pd.DataFrame(rows)
    if equity_df.empty:
        return equity_df

    return equity_df.sort_values(
        ["championship_equity_score", "ceiling_score", "adp_outperformance_score"],
        ascending=False,
    ).reset_index(drop=True)


def get_championship_equity_debug_info(players_df=None):
    df = players_df.copy() if players_df is not None else load_players_df().copy()
    if df.empty:
        return {
            "row_count": 0,
            "coverage": {},
            "missing_inputs": ["player_data"],
            "score_distributions": {},
            "top_championship_equity_players": [],
        }

    columns = _resolve_columns(df)
    equity_df = build_championship_equity_df(df)

    coverage = {}
    missing_inputs = []
    for key, col in columns.items():
        if col is None:
            coverage[key] = 0.0
            missing_inputs.append(key)
            continue

        coverage[key] = round(float(df[col].notna().mean() * 100.0), 2)

    score_cols = [
        "breakout_probability",
        "adp_outperformance_score",
        "ceiling_score",
        "age_curve_score",
        "sportsbook_advantage_score",
        "championship_equity_score",
    ]
    score_distributions = {
        col: equity_df[col].describe().round(2).to_dict()
        for col in score_cols
        if col in equity_df.columns
    }

    return {
        "row_count": int(len(df)),
        "equity_row_count": int(len(equity_df)),
        "coverage": coverage,
        "missing_inputs": missing_inputs,
        "score_distributions": score_distributions,
        "top_championship_equity_players": equity_df.head(20).to_dict("records")
        if not equity_df.empty
        else [],
    }
