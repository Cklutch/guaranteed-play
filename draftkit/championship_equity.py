from pathlib import Path

import pandas as pd

from draftkit.archetypes import risk_profile
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
BOOM_COLS = ["boom_score", "Boom Score"]
STABILITY_COLS = ["stability_score", "Stability Score"]
ARCHETYPE_COLS = ["archetype", "Archetype", "player_archetype"]
SPORTSBOOK_PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "sportsbook_projection",
]

# Real, populated data sources merged into working_df before per-row scoring
# (2026-08-27, replacing the ceiling/floor/boom/sportsbook_projection
# columns above, which are 0% populated in load_players_df() -- see
# CHANGELOG.md for the full diagnosis). Both manual-pull-only, same as
# Home.py's own use of them -- merge is skipped silently if missing, same
# defensive pattern as Home.py's sportsbook_vs_adp_comparison.csv read.
RISK_VARIABLES_PATH = "data/processed/risk_variables.csv"
SPORTSBOOK_COMPARISON_PATH = "data/processed/sportsbook_vs_adp_comparison.csv"

# How many equal-sized ADP/value tiers to normalize the sportsbook-vs-
# projection edge within (2026-08-27; see calculate_sportsbook_advantage_
# score's docstring). Diagnosed real, systematic edge disagreement between
# projection_points and sportsbook_half_ppr_points, biggest for round 1-2
# picks (mean -31.0) and shrinking toward deep sleepers (round 11+: -7.5) --
# a pool-wide percentile rank necessarily puts every star at the bottom on
# that data. 6 keeps each tier's rank statistically meaningful against the
# ~284-player matched sample (~47/tier) while still separating "typical for
# a 1st-rounder" from "typical for a late sleeper."
SPORTSBOOK_EDGE_TIER_COUNT = 6

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
        "boom": safe_col(df, BOOM_COLS),
        "stability": safe_col(df, STABILITY_COLS),
        "archetype": safe_col(df, ARCHETYPE_COLS),
        "sportsbook_projection": safe_col(df, SPORTSBOOK_PROJECTION_COLS),
    }


def calculate_age_curve_score(player_row, columns=None):
    columns = columns or {}
    position = str(_get_value(player_row, columns, "position", "")).upper()
    # Explicit default=None: safe_float()'s own default is 0.0, not None, so
    # a missing age would otherwise silently become "age 0" and fall through
    # to the youngest bucket below instead of the neutral 50.0 this check
    # intends -- see the same bug (and full writeup) at calculate_ceiling_score.
    age = _safe_float(_get_value(player_row, columns, "age"), None)

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
    # default=None (see calculate_ceiling_score's writeup) -- the <=0 guard
    # below happened to mask this one in practice, but keep it correct and
    # explicit rather than relying on that overlap.
    projection_rank = _safe_float(_get_value(player_row, columns, "projection_rank"), None)
    adp_rank = _safe_float(_get_value(player_row, columns, "adp_rank"), None)

    if projection_rank is None or adp_rank is None or projection_rank <= 0 or adp_rank <= 0:
        return 50.0

    market_discount = adp_rank - projection_rank
    return _clip_score(50.0 + (market_discount * 1.6))


def calculate_ceiling_score(player_row, columns=None):
    """Upside score built from the SAME real signal Home.py's live "Ceiling"
    view already uses (2026-08-27 rewrite -- see CHANGELOG.md): role_usage_td
    _score (1-5, TD-dependence) and volatility_diagnostic (~0-1.5) from
    risk_variables.csv, plus a lift for event/big-play archetypes. The
    original version read ceiling_projection/floor_projection/boom_score
    columns that are 0% populated in load_players_df() -- this replaces
    that dead path entirely rather than patching it, since even its
    fallback (raw projection * 1.18 against a flat per-position baseline)
    reintroduced the cross-position raw-points scale bug CLAUDE.md documents
    was deliberately fixed in the real Base Value engine.

    player_row must already carry role_usage_td_score/volatility_diagnostic/
    archetype_primary/_ceiling_relevance as real columns (merged/computed by
    build_championship_equity_df before the per-row loop) -- NOT resolved
    via the `columns` candidate-name dict the rest of this module uses,
    since these are fixed, known names from a merge we control, not
    free-form input data.

    _ceiling_relevance (0-1, a player's projection-points percentile within
    the pool) is NOT optional polish -- dropping it was tried first and
    produced a real, wrong result: role_usage_td_score/volatility_diagnostic
    measure RELATIVE unpredictability (variance as a fraction of a player's
    OWN baseline), so a practice-squad player with near-zero expected
    points reads as "high upside" the moment he scores one garbage-time TD,
    while a true bellcow (steady, high-volume, high-floor -- Gibbs, CMC,
    Chase) reads LOW on this signal precisely because he's not boom-bust.
    Undamped, this inverted the whole ranking: deep-bench names at the top,
    elite players at the bottom. Home.py's own use of this exact signal
    (the live Ceiling view) already solves this by weighting the bonus by
    percentile-of-final_score -- "a flat bonus [at the bottom of the board]
    would catapult irrelevant deep-bench TD vultures over startable
    players." Same fix, applied here.
    """
    td_score = _safe_float(player_row.get("role_usage_td_score"), None)
    volatility = _safe_float(player_row.get("volatility_diagnostic"), None)

    parts = []
    if td_score is not None:
        parts.append(max(0.0, min(1.0, (td_score - 1.0) / 4.0)))
    if volatility is not None:
        parts.append(max(0.0, min(1.0, volatility / 1.0)))

    if not parts:
        return 50.0

    upside = sum(parts) / len(parts)
    archetype_primary = player_row.get("archetype_primary")
    if risk_profile(archetype_primary) == "event":
        upside = min(1.0, upside + 0.15)

    relevance = _safe_float(player_row.get("_ceiling_relevance"), 0.0)
    return _clip_score(50.0 + (upside - 0.5) * 100.0 * relevance)


def calculate_breakout_probability(player_row, columns=None):
    columns = columns or {}
    position = str(_get_value(player_row, columns, "position", "")).upper()
    # default=None (see calculate_ceiling_score's writeup): without it, a
    # missing adp_rank (83% of the pool -- adp_rank coverage is ~17%)
    # silently became 0.0, which fell into "elif adp_rank <= 12" below and
    # wrongly applied the elite-top-12-ADP -6 penalty to most of the player
    # pool instead of skipping the adjustment entirely as intended.
    adp_rank = _safe_float(_get_value(player_row, columns, "adp_rank"), None)
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


def calculate_sportsbook_advantage_score(player_row):
    """Where the sportsbook and the internal projection disagree, scored by
    PERCENTILE RANK of the raw (sportsbook - projection) edge WITHIN the
    player's own ADP/value tier -- not the edge as a % of the player's own
    projection, and not ranked across the whole pool either. Two real bugs
    found and fixed here in sequence while wiring in real
    sportsbook_vs_adp_comparison.csv data (2026-08-27):

    1. The original formula divided the edge by the player's OWN
       projection, which blows up for anyone with a small baseline. Real
       case: Nicholas Singleton, projection 17.5, sportsbook 59.7 -- a real
       ~42-point edge, but as a fraction of his tiny 17.5-point baseline
       that's +241%, clipping to the 100 ceiling alongside every other
       bench player with a modest absolute edge and a small baseline,
       while Gibbs (a much bigger real edge in absolute points) sat
       mid-pack. Same class of bug CLAUDE.md documents Base Value's own
       market component was built to avoid -- fixed with a percentile rank
       instead of a raw ratio.
    2. Percentile rank ACROSS THE WHOLE POOL then revealed a second,
       different problem: projection_points runs systematically higher
       than sportsbook_half_ppr_points, and the gap is real and
       ADP-correlated -- biggest for round 1-2 picks (mean -31.0, 0%
       positive in the matched sample), shrinking toward deep sleepers
       (round 11+: mean -7.5, 37% positive). Ranked pool-wide, that
       necessarily put every star at the bottom and every bench player at
       the top -- an accurate rank on a genuinely tier-skewed input, not a
       coding error, but not the intended "does the market like this
       player more than my model" signal either. Fixed by ranking within
       SPORTSBOOK_EDGE_TIER_COUNT ADP-quantile tiers instead, so the
       question becomes "more than typical for someone drafted around
       here" -- see build_championship_equity_df() for the qcut.

    _sportsbook_edge_percentile is precomputed vectorized in
    build_championship_equity_df() for both reasons above -- this function
    stays a simple per-row lookup.
    """
    percentile = _safe_float(player_row.get("_sportsbook_edge_percentile"), None)
    if percentile is None:
        return 50.0
    return _clip_score(50.0 + (percentile - 0.5) * 100.0)


def calculate_championship_equity_score(player_row, columns=None, weights=None):
    columns = columns or {}
    weights = weights or CHAMPIONSHIP_EQUITY_WEIGHTS

    components = {
        "breakout_probability": calculate_breakout_probability(player_row, columns),
        "adp_outperformance_score": calculate_adp_outperformance_score(player_row, columns),
        "ceiling_score": calculate_ceiling_score(player_row, columns),
        "age_curve_score": calculate_age_curve_score(player_row, columns),
        "sportsbook_advantage_score": calculate_sportsbook_advantage_score(player_row),
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

    # Real, populated signals merged in for calculate_ceiling_score() and
    # calculate_sportsbook_advantage_score() (2026-08-27) -- both manual-
    # pull-only data sources, same defensive "skip silently if missing"
    # pattern Home.py itself uses for these exact two files.
    risk_path = Path(RISK_VARIABLES_PATH)
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        keep = [c for c in ("player_name", "role_usage_td_score", "volatility_diagnostic")
                if c in risk_df.columns]
        if "player_name" in keep and player_col:
            working_df = working_df.merge(
                risk_df[keep].drop_duplicates("player_name"),
                left_on=player_col, right_on="player_name",
                how="left", suffixes=("", "_risk"),
            )

    sportsbook_path = Path(SPORTSBOOK_COMPARISON_PATH)
    if sportsbook_path.exists():
        sportsbook_df = pd.read_csv(sportsbook_path)
        if "player_name" in sportsbook_df.columns and "sportsbook_half_ppr_points" in sportsbook_df.columns and player_col:
            # adp_rank pulled from this SAME file (not working_df's own,
            # possibly sparser resolution) so the value tier used to
            # normalize the edge below is consistent with the edge itself --
            # both come from the one file that actually has both numbers
            # for the same 284-ish matched players.
            sb_cols = ["player_name", "sportsbook_half_ppr_points"]
            if "adp_rank" in sportsbook_df.columns:
                sb_cols.append("adp_rank")
            keep = sportsbook_df[sb_cols].drop_duplicates("player_name")
            keep = keep.rename(columns={
                "player_name": "_sportsbook_player_name",
                "sportsbook_half_ppr_points": "sportsbook_projection",
                "adp_rank": "_sportsbook_adp_rank",
            })
            working_df = working_df.merge(
                keep, left_on=player_col, right_on="_sportsbook_player_name", how="left",
            ).drop(columns=["_sportsbook_player_name"])

    # Re-resolve columns against the merged frame: sportsbook_projection is
    # now a real column when the sportsbook merge above found a match.
    columns = _resolve_columns(working_df)

    # Relevance weight for calculate_ceiling_score() -- see that function's
    # docstring for why this damping is required, not optional. Percentile
    # of projection within this pool; players with no real projection
    # (~87% of load_players_df(), see CHANGELOG.md) rank last and get
    # relevance 0.0, correctly holding them at the neutral 50 regardless of
    # their raw upside signal, same as Home.py's own damping intent.
    _proj = (
        pd.to_numeric(working_df[columns["projection"]], errors="coerce")
        if columns.get("projection") else pd.Series(pd.NA, index=working_df.index)
    )
    working_df["_ceiling_relevance"] = _proj.rank(pct=True).fillna(0.0)

    # Percentile rank of (sportsbook - projection) for
    # calculate_sportsbook_advantage_score() -- see that function's
    # docstring for why percentile rank, not a raw ratio, AND why it's
    # ranked WITHIN value tier rather than across the whole pool (2026-08-27
    # follow-up fix). Only ranks players with both a real projection and
    # real sportsbook data; everyone else gets NaN -> the function's own
    # neutral-50 fallback.
    if columns.get("projection") and "sportsbook_projection" in working_df.columns:
        _sb = pd.to_numeric(working_df["sportsbook_projection"], errors="coerce")
        _has_both = _proj.notna() & _sb.notna() & (_proj > 0) & (_sb > 0)
        _edge = (_sb - _proj).where(_has_both)

        if "_sportsbook_adp_rank" in working_df.columns:
            _tier_rank = pd.to_numeric(working_df["_sportsbook_adp_rank"], errors="coerce").where(_has_both)
            # qcut, not fixed round-number bins: with only ~284 players
            # carrying both a real projection and real sportsbook data,
            # fixed round-based buckets came out badly lopsided when this
            # was diagnosed (round 1-2 n=24 vs round 11+ n=166) -- qcut
            # keeps every tier's percentile rank statistically meaningful.
            # duplicates="drop" defends against too few unique adp_rank
            # values to cut SPORTSBOOK_EDGE_TIER_COUNT ways.
            _tier = pd.qcut(_tier_rank, q=SPORTSBOOK_EDGE_TIER_COUNT, duplicates="drop")
            working_df["_sportsbook_edge_percentile"] = _edge.groupby(_tier, observed=True).rank(pct=True)
        else:
            # sportsbook_vs_adp_comparison.csv had no adp_rank column to tier
            # by -- fall back to a pool-wide rank rather than silently
            # dropping the signal entirely.
            working_df["_sportsbook_edge_percentile"] = _edge.rank(pct=True)
    else:
        working_df["_sportsbook_edge_percentile"] = pd.NA

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
            "sportsbook_advantage_score": calculate_sportsbook_advantage_score(row),
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

    # The two signals merged in from separate files (risk_variables.csv,
    # sportsbook_vs_adp_comparison.csv -- see build_championship_equity_df)
    # aren't on the raw, unmerged `df` this loop reads, so they'd otherwise
    # misleadingly report as 0% here even when the merge found real data.
    # Real, non-neutral output on the equity_df is the honest coverage
    # signal for these two: a score of exactly 50.0 means "no real signal
    # reached this player," by construction of both functions.
    if not equity_df.empty:
        coverage["ceiling_score_real_signal"] = round(
            float((equity_df["ceiling_score"] != 50.0).mean() * 100.0), 2
        )
        coverage["sportsbook_advantage_score_real_signal"] = round(
            float((equity_df["sportsbook_advantage_score"] != 50.0).mean() * 100.0), 2
        )

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
