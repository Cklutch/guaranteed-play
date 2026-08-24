import pandas as pd

from draftkit.common import clip_score as _clip_score
from draftkit.common import safe_float as _safe_float
from draftkit.data_access import load_players_df, safe_col
from draftkit.market_disagreement import build_market_disagreement_df
from draftkit.sportsbook_provider import get_sportsbook_provider_debug_info


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
AGE_COLS = ["age", "Age"]
ADP_COLS = ["adp", "ADP", "consensus_adp"]
ADP_RANK_COLS = ["adp_rank", "ADP Rank", "consensus_adp_rank", "rank", "Rank"]
PROJECTION_COLS = [
    "projection_points",
    "fantasy_points_projection",
    "market_projection",
    "Projection",
    "projected_points",
    "FPTS",
]
PROJECTION_RANK_COLS = [
    "projection_rank",
    "sportsbook_projection_rank",
    "market_projection_rank",
    "Projection Rank",
]
SPORTSBOOK_COLS = [
    "sportsbook_projection",
    "market_projection",
    "fantasy_points_projection",
]

SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
EXTREME_RANK_GAP = 75
LARGE_RANK_GAP = 35
PROJECTION_OUTLIER_MULTIPLIER = 1.45


def _resolve_columns(df):
    return {
        "player": safe_col(df, PLAYER_COLS),
        "position": safe_col(df, POSITION_COLS),
        "age": safe_col(df, AGE_COLS),
        "adp": safe_col(df, ADP_COLS),
        "adp_rank": safe_col(df, ADP_RANK_COLS),
        "projection": safe_col(df, PROJECTION_COLS),
        "projection_rank": safe_col(df, PROJECTION_RANK_COLS),
        "sportsbook_projection": safe_col(df, SPORTSBOOK_COLS),
    }


def _get(row, columns, key, default=None):
    column = columns.get(key)
    if column and column in row:
        return row.get(column)
    return default


def _rank_gap(row, columns):
    projection_rank = _safe_float(_get(row, columns, "projection_rank"))
    adp_rank = _safe_float(_get(row, columns, "adp_rank"))

    if projection_rank is None or adp_rank is None:
        return None

    return adp_rank - projection_rank


def _projection_context(players_df, columns):
    projection_col = columns.get("projection")
    position_col = columns.get("position")

    if projection_col is None or position_col is None or players_df.empty:
        return {}

    working = players_df.copy()
    working[projection_col] = pd.to_numeric(working[projection_col], errors="coerce")
    working[position_col] = working[position_col].astype(str).str.upper()
    working = working[
        working[position_col].isin(SCORABLE_POSITIONS)
        & working[projection_col].notna()
    ].copy()

    context = {}
    for position, group in working.groupby(position_col):
        context[position] = {
            "median": _safe_float(group[projection_col].median(), 0.0),
            "q95": _safe_float(group[projection_col].quantile(0.95), 0.0),
            "max": _safe_float(group[projection_col].max(), 0.0),
        }
    return context


def calculate_adp_trust(player_row, columns=None):
    """Real, disclosed split (plan_deanchor_scoring_from_adp.pdf): only
    genuine ADP-DATA-QUALITY issues (missing, internally inconsistent, out
    of range) deduct from trust_score. Model-vs-ADP RANK-GAP flags
    (extreme_adp_projection_gap, large_adp_projection_gap,
    elite_projection_late_adp_conflict) are real, disclosed divergence
    signals, not evidence the data is untrustworthy -- returned separately
    as `divergence_flags`, never scored. Confirmed circular before this
    fix: a real model-vs-ADP disagreement directly lowered this score,
    which then fed apply_signal_trust_adjustments()'s dampening -- pulling
    the very disagreement the model exists to surface back toward ADP."""
    columns = columns or {}
    adp = _safe_float(_get(player_row, columns, "adp"))
    adp_rank = _safe_float(_get(player_row, columns, "adp_rank"))
    projection_rank = _safe_float(_get(player_row, columns, "projection_rank"))

    score = 100.0
    flags = []
    divergence_flags = []

    if adp is None and adp_rank is None:
        score -= 60.0
        flags.append("missing_adp")

    if adp is not None and adp_rank is not None and abs(adp - adp_rank) > 10:
        score -= 20.0
        flags.append("adp_rank_inconsistency")

    if adp_rank is not None and (adp_rank <= 0 or adp_rank > 350):
        score -= 30.0
        flags.append("adp_rank_out_of_range")

    if projection_rank is not None and adp_rank is not None:
        gap = adp_rank - projection_rank
        if abs(gap) >= EXTREME_RANK_GAP:
            divergence_flags.append("extreme_adp_projection_gap")
        elif abs(gap) >= LARGE_RANK_GAP:
            divergence_flags.append("large_adp_projection_gap")

    if adp_rank is not None and adp_rank >= 180 and projection_rank is not None and projection_rank <= 24:
        divergence_flags.append("elite_projection_late_adp_conflict")

    return {
        "trust_score": _clip_score(score),
        "flags": flags,
        "divergence_flags": divergence_flags,
    }


def calculate_projection_trust(player_row, columns=None, projection_context=None):
    columns = columns or {}
    projection_context = projection_context or {}

    projection = _safe_float(_get(player_row, columns, "projection"))
    projection_rank = _safe_float(_get(player_row, columns, "projection_rank"))
    position = str(_get(player_row, columns, "position", "")).upper()

    score = 100.0
    flags = []

    if projection is None:
        score -= 65.0
        flags.append("missing_projection")

    if projection_rank is None:
        score -= 25.0
        flags.append("missing_projection_rank")

    if projection is not None and projection <= 0:
        score -= 35.0
        flags.append("invalid_projection")

    context = projection_context.get(position, {})
    q95 = _safe_float(context.get("q95"), 0.0)
    median = _safe_float(context.get("median"), 0.0)
    if projection is not None and q95 and projection > q95 * PROJECTION_OUTLIER_MULTIPLIER:
        score -= 30.0
        flags.append("projection_outlier")
    elif projection is not None and median and projection > median * 2.0:
        score -= 20.0
        flags.append("projection_above_position_norm")

    return {
        "trust_score": _clip_score(score),
        "flags": flags,
    }


def calculate_sportsbook_trust(provider_debug=None):
    provider_debug = provider_debug or get_sportsbook_provider_debug_info()
    health_report = provider_debug.get("health_report", [])
    fallback_chain = provider_debug.get("fallback_chain", {})
    selected_source = fallback_chain.get("selected_source")

    active_health = None
    for report in health_report:
        if report.get("active"):
            active_health = report
            break

    if active_health is None and health_report:
        active_health = health_report[0]

    row_count = int(active_health.get("row_count", 0)) if active_health else 0
    coverage = _safe_float(active_health.get("coverage"), 0.0) if active_health else 0.0
    status = active_health.get("status") if active_health else "missing"

    score = 100.0
    flags = []

    if status != "healthy":
        score -= 50.0
        flags.append("provider_unhealthy")

    if row_count <= 0:
        score -= 50.0
        flags.append("no_sportsbook_rows")
    elif row_count < 25:
        score -= 30.0
        flags.append("low_sportsbook_row_count")

    if coverage < 25:
        score -= 30.0
        flags.append("low_sportsbook_coverage")
    elif coverage < 60:
        score -= 15.0
        flags.append("partial_sportsbook_coverage")

    if selected_source != "active_provider":
        score -= 20.0
        flags.append("using_projection_fallback")

    trust_score = _clip_score(score)
    if trust_score >= 75:
        label = "HIGH"
    elif trust_score >= 45:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "trust_score": trust_score,
        "trust_label": label,
        "flags": flags,
        "provider_name": active_health.get("provider_name") if active_health else None,
        "row_count": row_count,
        "coverage": coverage,
        "status": status,
    }


def calculate_market_disagreement_trust(player_row, columns=None, market_row=None, adp_trust=None, sportsbook_trust=None):
    """Same real/divergence split as calculate_adp_trust() -- see that
    function's docstring. `low_adp_trust` now checks the GENUINE (post-
    split) adp_trust score, so it only fires for real ADP data gaps, not
    for a real model-vs-ADP disagreement. `extreme_market_disagreement`
    (a direct model-vs-market rank-gap check, same shape as
    extreme_adp_projection_gap) moves to divergence_flags, never scored."""
    columns = columns or {}
    adp_trust = adp_trust or calculate_adp_trust(player_row, columns)
    sportsbook_trust = sportsbook_trust or calculate_sportsbook_trust()

    score = 100.0
    flags = []
    divergence_flags = []

    if sportsbook_trust.get("trust_score", 0.0) < 45:
        score -= 35.0
        flags.append("low_sportsbook_trust")

    if adp_trust.get("trust_score", 0.0) < 50:
        score -= 35.0
        flags.append("low_adp_trust")

    market_disagreement = None
    market_confidence = None
    if market_row is not None:
        market_disagreement = _safe_float(market_row.get("market_disagreement"))
        market_confidence = _safe_float(market_row.get("market_confidence"))

    if market_confidence is None or market_confidence < 55:
        score -= 25.0
        flags.append("low_market_confidence")

    if market_disagreement is not None and abs(market_disagreement) >= EXTREME_RANK_GAP:
        divergence_flags.append("extreme_market_disagreement")

    if _safe_float(_get(player_row, columns, "sportsbook_projection")) is None:
        score -= 25.0
        flags.append("missing_sportsbook_inputs")

    return {
        "trust_score": _clip_score(score),
        "flags": flags,
        "divergence_flags": divergence_flags,
    }


def calculate_signal_trust(adp_trust, projection_trust, market_trust, sportsbook_trust):
    scores = [
        adp_trust.get("trust_score", 50.0),
        projection_trust.get("trust_score", 50.0),
        market_trust.get("trust_score", 50.0),
        sportsbook_trust.get("trust_score", 50.0),
    ]
    weights = [0.30, 0.30, 0.25, 0.15]
    weighted = sum(score * weight for score, weight in zip(scores, weights))
    return _clip_score(weighted)


def detect_data_anomalies(player_row, columns=None, market_row=None, duplicate_names=None, projection_context=None, sportsbook_trust=None):
    """Returns (genuine_flags, divergence_flags) -- see calculate_adp_trust()'s
    docstring for the real/divergence split this mirrors.
    projection_rank_conflict/adp_rank_conflict are the SAME rank_gap>=75
    check as adp_trust's extreme_adp_projection_gap (a real model-vs-ADP
    disagreement, not a data anomaly) -- moved to divergence_flags. This
    was the entire real contents of apply_signal_trust_adjustments()'s old
    `severe_flags` set (all 4 members were divergence-only), which is why
    that dampening tier is removed rather than fixed in place."""
    columns = columns or {}
    duplicate_names = duplicate_names or set()
    projection_context = projection_context or {}
    sportsbook_trust = sportsbook_trust or calculate_sportsbook_trust()

    flags = []
    divergence_flags = []
    player_name = str(_get(player_row, columns, "player", "")).strip()
    age = _safe_float(_get(player_row, columns, "age"))
    rank_gap = _rank_gap(player_row, columns)

    adp_trust = calculate_adp_trust(player_row, columns)
    projection_trust = calculate_projection_trust(player_row, columns, projection_context)
    market_trust = calculate_market_disagreement_trust(
        player_row,
        columns,
        market_row=market_row,
        adp_trust=adp_trust,
        sportsbook_trust=sportsbook_trust,
    )

    flags.extend(adp_trust["flags"])
    flags.extend(projection_trust["flags"])
    flags.extend(market_trust["flags"])
    divergence_flags.extend(adp_trust["divergence_flags"])
    divergence_flags.extend(market_trust["divergence_flags"])

    if player_name.lower() in duplicate_names:
        flags.append("duplicate_player_name")

    if age is not None and (age < 18 or age > 45):
        flags.append("invalid_age")

    if rank_gap is not None and abs(rank_gap) >= EXTREME_RANK_GAP:
        divergence_flags.append("projection_rank_conflict")
        divergence_flags.append("adp_rank_conflict")

    if sportsbook_trust.get("trust_score", 0.0) < 45:
        flags.append("missing_or_low_trust_sportsbook_inputs")

    return sorted(set(flags)), sorted(set(divergence_flags))


def _market_lookup(players_df):
    try:
        market_df = build_market_disagreement_df(players_df)
    except Exception:
        return {}

    if market_df.empty or "player_name" not in market_df.columns:
        return {}

    return {
        str(row["player_name"]).strip().lower(): row.to_dict()
        for _, row in market_df.iterrows()
    }


def build_signal_trust_report(players_df=None):
    df = players_df.copy() if players_df is not None else load_players_df().copy()
    if df.empty:
        return pd.DataFrame()

    columns = _resolve_columns(df)
    player_col = columns.get("player")
    if player_col is None:
        return pd.DataFrame()

    duplicate_names = set(
        df[player_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .loc[lambda values: values.duplicated(keep=False)]
        .tolist()
    )
    projection_context = _projection_context(df, columns)
    market_lookup = _market_lookup(df)
    sportsbook_trust = calculate_sportsbook_trust()

    rows = []
    for _, row in df.iterrows():
        player_name = str(row.get(player_col, "")).strip()
        market_row = market_lookup.get(player_name.lower())
        adp_trust = calculate_adp_trust(row, columns)
        projection_trust = calculate_projection_trust(row, columns, projection_context)
        market_trust = calculate_market_disagreement_trust(
            row,
            columns,
            market_row=market_row,
            adp_trust=adp_trust,
            sportsbook_trust=sportsbook_trust,
        )
        anomalies, divergence = detect_data_anomalies(
            row,
            columns,
            market_row=market_row,
            duplicate_names=duplicate_names,
            projection_context=projection_context,
            sportsbook_trust=sportsbook_trust,
        )
        # Real, disclosed model-vs-ADP/market divergence -- surfaced for
        # transparency, deliberately excluded from anomaly_flags/
        # signal_trust_score (see calculate_adp_trust()'s docstring).
        divergence = sorted(set(divergence) | set(adp_trust["divergence_flags"]) | set(market_trust["divergence_flags"]))

        rows.append({
            "player_name": player_name,
            "position": row.get(columns["position"]) if columns.get("position") else "",
            "signal_trust_score": calculate_signal_trust(
                adp_trust,
                projection_trust,
                market_trust,
                sportsbook_trust,
            ),
            "adp_trust_score": adp_trust["trust_score"],
            "projection_trust_score": projection_trust["trust_score"],
            "market_trust_score": market_trust["trust_score"],
            "sportsbook_trust_score": sportsbook_trust["trust_score"],
            "sportsbook_trust_label": sportsbook_trust["trust_label"],
            "projection_rank": _safe_float(_get(row, columns, "projection_rank")),
            "adp_rank": _safe_float(_get(row, columns, "adp_rank")),
            "rank_gap": _rank_gap(row, columns),
            "market_disagreement": _safe_float(market_row.get("market_disagreement"))
            if market_row
            else None,
            "anomaly_flags": anomalies,
            "anomaly_count": len(anomalies),
            "market_divergence_flags": divergence,
        })

    return pd.DataFrame(rows).sort_values(
        ["signal_trust_score", "anomaly_count"],
        ascending=[True, False],
    ).reset_index(drop=True)


def get_signal_trust_debug_info(players_df=None):
    df = players_df.copy() if players_df is not None else load_players_df().copy()
    report_df = build_signal_trust_report(df)

    if df.empty or report_df.empty:
        return {
            "row_count": int(len(df)),
            "signal_coverage": {},
            "sportsbook_trust": calculate_sportsbook_trust(),
            "most_common_anomalies": {},
            "lowest_trust_players": [],
            "tyreek_hill": [],
        }

    columns = _resolve_columns(df)
    signal_coverage = {}
    for key, column in columns.items():
        signal_coverage[key] = (
            round(float(df[column].notna().mean() * 100.0), 2)
            if column and column in df.columns
            else 0.0
        )

    anomaly_counts = {}
    for flags in report_df["anomaly_flags"]:
        for flag in flags:
            anomaly_counts[flag] = anomaly_counts.get(flag, 0) + 1

    tyreek_df = report_df[
        report_df["player_name"].astype(str).str.lower() == "tyreek hill"
    ]

    return {
        "row_count": int(len(df)),
        "signal_coverage": signal_coverage,
        "sportsbook_trust": calculate_sportsbook_trust(),
        "most_common_anomalies": dict(
            sorted(anomaly_counts.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
        "score_distributions": {
            column: report_df[column].describe().round(2).to_dict()
            for column in [
                "signal_trust_score",
                "adp_trust_score",
                "projection_trust_score",
                "market_trust_score",
                "sportsbook_trust_score",
            ]
            if column in report_df.columns
        },
        "lowest_trust_players": report_df.head(25).to_dict("records"),
        "tyreek_hill": tyreek_df.to_dict("records"),
    }
