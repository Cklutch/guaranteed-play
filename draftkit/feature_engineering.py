import pandas as pd

from draftkit.data_access import load_players_df, safe_col


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
POSITION_COLS = ["position", "pos", "Pos", "Position"]
PROJECTION_COLS = ["projection_points", "Projection", "projected_points", "FPTS", "proj_points"]
ADP_COLS = ["adp", "ADP", "rank", "Rank"]
DURABILITY_COLS = ["durability_grade", "Durability", "durability", "durability_score"]
INJURY_RISK_COLS = ["injury_risk", "Injury Risk", "injury_score", "risk_score"]
VOLATILITY_COLS = ["volatility_score", "Volatility", "weekly_volatility", "std_dev"]
BOOM_COLS = ["boom_score", "boom_rate", "ceiling_score", "ceiling_projection"]
BUST_COLS = ["bust_score", "bust_rate", "floor_score", "floor_projection"]
GAMES_PLAYED_COLS = ["games_played", "Games Played", "gp"]
MISSED_GAMES_COLS = ["missed_games", "games_missed", "Missed Games"]
AGE_COLS = ["age", "Age"]
STATUS_COLS = ["injury_status", "status", "Status"]

SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
SUPPORTED_ARCHETYPES = ["BOOM", "STEADY", "RISKY", "UPSIDE", "SAFE"]


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value, default=50.0):
    score = _safe_float(value, default)
    return round(max(0.0, min(100.0, score)), 2)


def _get_row_value(player_row, columns, key, default=None):
    column = columns.get(key) if columns else None
    if column is None:
        return default

    value = player_row.get(column, default)
    if pd.isna(value):
        return default

    return value


def _normalize_durability(value, default=70.0):
    if isinstance(value, str):
        grade = value.strip().upper()
        grade_map = {
            "A+": 98.0,
            "A": 94.0,
            "A-": 90.0,
            "B+": 86.0,
            "B": 82.0,
            "B-": 78.0,
            "C+": 74.0,
            "C": 70.0,
            "C-": 66.0,
            "D": 58.0,
            "F": 45.0,
        }
        if grade in grade_map:
            return grade_map[grade]

    return _clamp_score(value, default)


def _percentile_score(value, low, high, default=50.0):
    value = _safe_float(value, None)
    low = _safe_float(low, None)
    high = _safe_float(high, None)

    if value is None or low is None or high is None or high <= low:
        return default

    return _clamp_score(((value - low) / (high - low)) * 100.0, default)


def _get_feature_columns(df):
    return {
        "player_col": safe_col(df, PLAYER_COLS),
        "position_col": safe_col(df, POSITION_COLS),
        "projection_col": safe_col(df, PROJECTION_COLS),
        "adp_col": safe_col(df, ADP_COLS),
        "durability_col": safe_col(df, DURABILITY_COLS),
        "injury_risk_col": safe_col(df, INJURY_RISK_COLS),
        "volatility_col": safe_col(df, VOLATILITY_COLS),
        "boom_col": safe_col(df, BOOM_COLS),
        "bust_col": safe_col(df, BUST_COLS),
        "games_played_col": safe_col(df, GAMES_PLAYED_COLS),
        "missed_games_col": safe_col(df, MISSED_GAMES_COLS),
        "age_col": safe_col(df, AGE_COLS),
        "status_col": safe_col(df, STATUS_COLS),
    }


def _build_feature_context(df, columns):
    projection_col = columns.get("projection_col")
    adp_col = columns.get("adp_col")

    context = {
        "projection_low": 0.0,
        "projection_median": 0.0,
        "projection_high": 0.0,
        "adp_median": 100.0,
        "adp_late": 100.0,
    }

    if df.empty:
        return context

    if projection_col is not None:
        projections = pd.to_numeric(df[projection_col], errors="coerce").dropna()
        if not projections.empty:
            context["projection_low"] = float(projections.quantile(0.10))
            context["projection_median"] = float(projections.median())
            context["projection_high"] = float(projections.quantile(0.90))

    if adp_col is not None:
        adps = pd.to_numeric(df[adp_col], errors="coerce").dropna()
        if not adps.empty:
            context["adp_median"] = float(adps.median())
            context["adp_late"] = float(adps.quantile(0.75))

    return context


def calculate_durability_grade(player_row, columns=None, context=None):
    """
    Estimate durability on a 0-100 scale. Higher means safer availability.
    """
    direct_value = _get_row_value(player_row, columns, "durability_col")
    if direct_value is not None:
        return _normalize_durability(direct_value, 70.0)

    games_played = _get_row_value(player_row, columns, "games_played_col")
    if games_played is not None:
        return _clamp_score((_safe_float(games_played, 12.0) / 17.0) * 100.0, 70.0)

    missed_games = _get_row_value(player_row, columns, "missed_games_col")
    if missed_games is not None:
        return _clamp_score(92.0 - (_safe_float(missed_games, 0.0) * 5.0), 70.0)

    return 70.0


def calculate_injury_risk(player_row, columns=None, context=None):
    """
    Estimate injury risk on a 0-100 scale. Higher means more risk.
    """
    direct_value = _get_row_value(player_row, columns, "injury_risk_col")
    if direct_value is not None:
        return _clamp_score(direct_value, 50.0)

    missed_games = _get_row_value(player_row, columns, "missed_games_col")
    if missed_games is not None:
        return _clamp_score(25.0 + (_safe_float(missed_games, 0.0) * 6.0), 50.0)

    status = str(_get_row_value(player_row, columns, "status_col", "")).strip().lower()
    if status:
        if any(term in status for term in ["out", "ir", "pup", "doubtful"]):
            return 85.0
        if any(term in status for term in ["questionable", "limited"]):
            return 65.0
        if any(term in status for term in ["probable", "active", "healthy"]):
            return 30.0

    age = _get_row_value(player_row, columns, "age_col")
    if age is not None:
        age_value = _safe_float(age, 27.0)
        if age_value >= 32:
            return 62.0
        if age_value >= 29:
            return 55.0
        if age_value <= 23:
            return 42.0

    return 50.0


def calculate_volatility_score(player_row, columns=None, context=None):
    """
    Estimate week-to-week volatility on a 0-100 scale.
    """
    context = context or {}
    direct_value = _get_row_value(player_row, columns, "volatility_col")
    if direct_value is not None:
        return _clamp_score(direct_value, 50.0)

    projection = _safe_float(
        _get_row_value(player_row, columns, "projection_col"),
        context.get("projection_median", 0.0),
    )
    adp = _safe_float(
        _get_row_value(player_row, columns, "adp_col"),
        context.get("adp_median", 100.0),
    )

    projection_percentile = _percentile_score(
        projection,
        context.get("projection_low", 0.0),
        context.get("projection_high", 0.0),
        50.0,
    )

    volatility = 45.0
    if projection_percentile >= 75 and adp > context.get("adp_median", 100.0):
        volatility += 18.0
    elif projection_percentile >= 65:
        volatility += 8.0
    elif projection_percentile <= 30:
        volatility += 10.0

    if adp >= context.get("adp_late", 100.0):
        volatility += 8.0

    return _clamp_score(volatility, 50.0)


def calculate_boom_score(player_row, columns=None, context=None):
    """
    Estimate ceiling upside on a 0-100 scale.
    """
    context = context or {}
    direct_value = _get_row_value(player_row, columns, "boom_col")
    if direct_value is not None:
        return _clamp_score(direct_value, 50.0)

    projection = _safe_float(
        _get_row_value(player_row, columns, "projection_col"),
        context.get("projection_median", 0.0),
    )
    projection_percentile = _percentile_score(
        projection,
        context.get("projection_low", 0.0),
        context.get("projection_high", 0.0),
        50.0,
    )
    volatility = calculate_volatility_score(player_row, columns, context)

    return _clamp_score((projection_percentile * 0.70) + (volatility * 0.30), 50.0)


def calculate_bust_score(player_row, columns=None, context=None):
    """
    Estimate downside risk on a 0-100 scale.
    """
    context = context or {}
    direct_value = _get_row_value(player_row, columns, "bust_col")
    if direct_value is not None:
        return _clamp_score(direct_value, 50.0)

    injury_risk = calculate_injury_risk(player_row, columns, context)
    volatility = calculate_volatility_score(player_row, columns, context)
    durability = calculate_durability_grade(player_row, columns, context)

    return _clamp_score(
        (injury_risk * 0.40)
        + (volatility * 0.35)
        + ((100.0 - durability) * 0.25),
        50.0,
    )


def calculate_stability_score(player_row, columns=None, context=None):
    """
    Estimate dependability on a 0-100 scale.
    """
    durability = calculate_durability_grade(player_row, columns, context)
    injury_risk = calculate_injury_risk(player_row, columns, context)
    volatility = calculate_volatility_score(player_row, columns, context)

    return _clamp_score(
        (durability * 0.45)
        + ((100.0 - injury_risk) * 0.35)
        + ((100.0 - volatility) * 0.20),
        50.0,
    )


def assign_player_archetype(player_row, columns=None, context=None):
    """
    Assign an explainable player archetype from engineered feature scores.
    """
    boom_score = calculate_boom_score(player_row, columns, context)
    bust_score = calculate_bust_score(player_row, columns, context)
    stability_score = calculate_stability_score(player_row, columns, context)
    injury_risk = calculate_injury_risk(player_row, columns, context)

    if injury_risk >= 70 or bust_score >= 68:
        return "RISKY"
    if stability_score >= 72 and bust_score <= 42:
        return "SAFE"
    if boom_score >= 72 and stability_score >= 55:
        return "UPSIDE"
    if boom_score >= 65 and bust_score >= 50:
        return "BOOM"
    if stability_score >= 60:
        return "STEADY"

    return "RISKY" if bust_score > boom_score else "BOOM"


def build_player_features_df(players_df=None):
    """
    Build player-level analytics for recommendation and validation workflows.
    """
    df = load_players_df().copy() if players_df is None else players_df.copy()
    if df.empty:
        return pd.DataFrame()

    columns = _get_feature_columns(df)
    player_col = columns.get("player_col")
    position_col = columns.get("position_col")
    projection_col = columns.get("projection_col")

    if player_col is None or position_col is None or projection_col is None:
        return pd.DataFrame()

    df = df[df[position_col].astype(str).str.upper().isin(SCORABLE_POSITIONS)].copy()
    if df.empty:
        return pd.DataFrame()

    df[projection_col] = pd.to_numeric(df[projection_col], errors="coerce")
    df = df.dropna(subset=[projection_col]).copy()
    if df.empty:
        return pd.DataFrame()

    context = _build_feature_context(df, columns)
    rows = []

    for _, row in df.iterrows():
        injury_risk = calculate_injury_risk(row, columns, context)
        durability_grade = calculate_durability_grade(row, columns, context)
        boom_score = calculate_boom_score(row, columns, context)
        bust_score = calculate_bust_score(row, columns, context)
        stability_score = calculate_stability_score(row, columns, context)

        rows.append({
            "player_name": str(row[player_col]),
            "position": str(row[position_col]).upper(),
            "projection_points": round(_safe_float(row[projection_col], 0.0), 2),
            "injury_risk": injury_risk,
            "durability_grade": durability_grade,
            "boom_score": boom_score,
            "bust_score": bust_score,
            "stability_score": stability_score,
            "archetype": assign_player_archetype(row, columns, context),
        })

    return pd.DataFrame(rows)


def _distribution_summary(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }

    return {
        "count": int(values.count()),
        "min": round(float(values.min()), 2),
        "median": round(float(values.median()), 2),
        "mean": round(float(values.mean()), 2),
        "max": round(float(values.max()), 2),
    }


def get_feature_engineering_debug_info():
    """
    Return diagnostics for validating engineered player features.
    """
    raw_df = load_players_df().copy()
    columns = _get_feature_columns(raw_df) if not raw_df.empty else {}
    features_df = build_player_features_df(raw_df)

    required_columns = ["player_col", "position_col", "projection_col"]
    optional_columns = [
        "adp_col",
        "durability_col",
        "injury_risk_col",
        "volatility_col",
        "boom_col",
        "bust_col",
        "games_played_col",
        "missed_games_col",
        "age_col",
        "status_col",
    ]

    missing_required = [
        key for key in required_columns if not columns.get(key)
    ]
    missing_optional = [
        key for key in optional_columns if not columns.get(key)
    ]

    feature_distributions = {}
    if not features_df.empty:
        for column in [
            "injury_risk",
            "durability_grade",
            "boom_score",
            "bust_score",
            "stability_score",
        ]:
            feature_distributions[column] = _distribution_summary(features_df[column])

    return {
        "raw_shape": raw_df.shape,
        "features_shape": features_df.shape,
        "detected_columns": columns,
        "missing_required_columns": missing_required,
        "missing_optional_columns": missing_optional,
        "feature_distributions": feature_distributions,
        "archetype_counts": features_df["archetype"].value_counts().to_dict()
        if not features_df.empty
        else {},
        "position_counts": features_df["position"].value_counts().to_dict()
        if not features_df.empty
        else {},
        "sample_features": features_df.head(10).to_dict("records")
        if not features_df.empty
        else [],
    }
