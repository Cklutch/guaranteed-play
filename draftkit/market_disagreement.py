import pandas as pd

from draftkit.common import clip_score as _clip_score
from draftkit.common import safe_float as _safe_float
from draftkit.data_access import load_players_df, safe_col


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
TEAM_COLS = ["team", "Team", "team_abbr"]
PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "projection_points",
    "Projection",
    "projected_points",
    "FPTS",
]
SPORTSBOOK_PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "sportsbook_projection",
]
PROJECTION_RANK_COLS = [
    "sportsbook_projection_rank",
    "market_projection_rank",
    "projection_rank",
    "Projection Rank",
]
FANTASYPROS_RANK_COLS = [
    "fantasypros_rank",
    "fantasypros_ecr",
    "expert_consensus_rank",
    "consensus_rank",
    "rank",
    "Rank",
    "RK",
]
ADP_RANK_COLS = ["adp_rank", "ADP Rank", "consensus_adp_rank"]
ADP_COLS = ["adp", "ADP", "consensus_adp"]


def _get_value(player_row, columns, key, default=None):
    col = columns.get(key)
    if col and col in player_row:
        return player_row.get(col)
    return default


def _resolve_columns(df):
    fantasypros_rank_col = safe_col(df, FANTASYPROS_RANK_COLS)
    adp_rank_col = safe_col(df, ADP_RANK_COLS)

    return {
        "player": safe_col(df, PLAYER_COLS),
        "position": safe_col(df, POSITION_COLS),
        "team": safe_col(df, TEAM_COLS),
        "projection": safe_col(df, PROJECTION_COLS),
        "sportsbook_projection": safe_col(df, SPORTSBOOK_PROJECTION_COLS),
        "projection_rank": safe_col(df, PROJECTION_RANK_COLS),
        "fantasypros_rank": fantasypros_rank_col or adp_rank_col,
        "adp_rank": adp_rank_col,
        "adp": safe_col(df, ADP_COLS),
    }


def calculate_projection_gap(player_row, columns=None):
    """
    Positive gap means projection/market rank is better than the public rank.
    Example: projection rank 8 vs FantasyPros rank 22 -> +14.
    """
    columns = columns or {}
    projection_rank = _safe_float(_get_value(player_row, columns, "projection_rank"))
    fantasypros_rank = _safe_float(_get_value(player_row, columns, "fantasypros_rank"))

    if projection_rank is None or fantasypros_rank is None:
        return 0.0

    return round(fantasypros_rank - projection_rank, 2)


def calculate_adp_gap(player_row, columns=None):
    """
    Positive gap means projection/market rank is better than ADP.
    Example: projection rank 8 vs ADP rank 25 -> +17.
    """
    columns = columns or {}
    projection_rank = _safe_float(_get_value(player_row, columns, "projection_rank"))
    adp_rank = _safe_float(_get_value(player_row, columns, "adp_rank"))

    if projection_rank is None or adp_rank is None:
        return 0.0

    return round(adp_rank - projection_rank, 2)


def calculate_market_disagreement(player_row, columns=None):
    """
    Return a signed disagreement score.

    Positive values indicate the projection market is higher on the player than
    public rank/ADP. Negative values indicate the public market is higher than
    projection inputs.
    """
    projection_gap = calculate_projection_gap(player_row, columns)
    adp_gap = calculate_adp_gap(player_row, columns)

    return round((projection_gap * 0.55) + (adp_gap * 0.45), 2)


def calculate_market_confidence(player_row, columns=None):
    """
    Estimate how trustworthy the disagreement signal is based on source coverage.
    """
    columns = columns or {}
    confidence = 20.0

    if _safe_float(_get_value(player_row, columns, "projection_rank")) is not None:
        confidence += 25.0
    if _safe_float(_get_value(player_row, columns, "fantasypros_rank")) is not None:
        confidence += 20.0
    if _safe_float(_get_value(player_row, columns, "adp_rank")) is not None:
        confidence += 20.0
    if _safe_float(_get_value(player_row, columns, "sportsbook_projection")) is not None:
        confidence += 25.0
    elif _safe_float(_get_value(player_row, columns, "projection")) is not None:
        confidence += 10.0

    projection_gap = abs(calculate_projection_gap(player_row, columns))
    adp_gap = abs(calculate_adp_gap(player_row, columns))
    if projection_gap >= 8 and adp_gap >= 8:
        confidence += 10.0
    elif projection_gap >= 8 or adp_gap >= 8:
        confidence += 5.0

    return _clip_score(confidence)


def _classify_market_signal(disagreement, confidence):
    if confidence < 45:
        return "Insufficient Data"
    if disagreement >= 15:
        return "Potential Hidden Value"
    if disagreement >= 8:
        return "Positive Disagreement"
    if disagreement <= -15:
        return "Potential Overvalued"
    if disagreement <= -8:
        return "Negative Disagreement"
    return "Market Aligned"


def build_market_disagreement_df(players_df=None):
    df = players_df.copy() if players_df is not None else load_players_df().copy()
    if df.empty:
        return pd.DataFrame()

    columns = _resolve_columns(df)
    player_col = columns.get("player")
    position_col = columns.get("position")

    if player_col is None or position_col is None:
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        projection_gap = calculate_projection_gap(row, columns)
        adp_gap = calculate_adp_gap(row, columns)
        disagreement = calculate_market_disagreement(row, columns)
        confidence = calculate_market_confidence(row, columns)

        rows.append({
            "player_name": row.get(player_col),
            "position": row.get(position_col),
            "team": row.get(columns["team"]) if columns.get("team") else "",
            "projection_points": _safe_float(_get_value(row, columns, "projection")),
            "sportsbook_projection": _safe_float(
                _get_value(row, columns, "sportsbook_projection")
            ),
            "projection_rank": _safe_float(_get_value(row, columns, "projection_rank")),
            "fantasypros_rank": _safe_float(_get_value(row, columns, "fantasypros_rank")),
            "adp": _safe_float(_get_value(row, columns, "adp")),
            "adp_rank": _safe_float(_get_value(row, columns, "adp_rank")),
            "projection_gap": projection_gap,
            "adp_gap": adp_gap,
            "market_disagreement": disagreement,
            "market_confidence": confidence,
            "market_signal": _classify_market_signal(disagreement, confidence),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["abs_market_disagreement"] = out["market_disagreement"].abs()
    return out.sort_values(
        ["market_confidence", "abs_market_disagreement", "market_disagreement"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def identify_hidden_value_players(players_df=None, min_disagreement=15.0, min_confidence=55.0):
    disagreement_df = build_market_disagreement_df(players_df)
    if disagreement_df.empty:
        return disagreement_df

    hidden_value_df = disagreement_df[
        (disagreement_df["market_disagreement"] >= min_disagreement)
        & (disagreement_df["market_confidence"] >= min_confidence)
    ].copy()

    return hidden_value_df.sort_values(
        ["market_disagreement", "market_confidence"],
        ascending=False,
    ).reset_index(drop=True)


def identify_overvalued_players(players_df=None, min_disagreement=15.0, min_confidence=55.0):
    disagreement_df = build_market_disagreement_df(players_df)
    if disagreement_df.empty:
        return disagreement_df

    overvalued_df = disagreement_df[
        (disagreement_df["market_disagreement"] <= -abs(min_disagreement))
        & (disagreement_df["market_confidence"] >= min_confidence)
    ].copy()

    return overvalued_df.sort_values(
        ["market_disagreement", "market_confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)


def get_market_disagreement_debug_info(players_df=None):
    df = players_df.copy() if players_df is not None else load_players_df().copy()
    if df.empty:
        return {
            "row_count": 0,
            "coverage": {},
            "missing_inputs": ["player_data"],
            "signal_counts": {},
            "top_hidden_value_players": [],
            "top_overvalued_players": [],
        }

    columns = _resolve_columns(df)
    disagreement_df = build_market_disagreement_df(df)

    coverage = {}
    missing_inputs = []
    for key, col in columns.items():
        if col is None:
            coverage[key] = 0.0
            missing_inputs.append(key)
            continue
        coverage[key] = round(float(df[col].notna().mean() * 100.0), 2)

    score_cols = [
        "projection_gap",
        "adp_gap",
        "market_disagreement",
        "market_confidence",
    ]
    distributions = {
        col: disagreement_df[col].describe().round(2).to_dict()
        for col in score_cols
        if col in disagreement_df.columns
    }

    return {
        "row_count": int(len(df)),
        "disagreement_row_count": int(len(disagreement_df)),
        "coverage": coverage,
        "missing_inputs": missing_inputs,
        "score_distributions": distributions,
        "signal_counts": disagreement_df["market_signal"].value_counts().to_dict()
        if "market_signal" in disagreement_df.columns
        else {},
        "top_hidden_value_players": identify_hidden_value_players(df).head(20).to_dict("records"),
        "top_overvalued_players": identify_overvalued_players(df).head(20).to_dict("records"),
    }
