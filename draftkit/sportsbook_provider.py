import pandas as pd

from draftkit.data_access import load_players_df, safe_col
from draftkit.projection_engine import build_player_projection_df
from draftkit.providers.provider_base import (
    CANONICAL_PROVIDER_COLUMNS,
    MockSportsbookProvider,
    SportsbookProvider,
    empty_provider_df,
    normalize_provider_output,
    utc_timestamp,
)
from draftkit.providers.provider_registry import (
    get_active_provider,
    get_provider,
    get_provider_health_report,
    list_providers,
    register_provider,
    set_active_provider,
)


PLAYER_COLS = ["player_name", "Player", "player", "name", "full_name"]
PROJECTION_COLS = [
    "fantasy_points_projection",
    "market_projection",
    "projection_points",
    "Projection",
    "projected_points",
    "FPTS",
]

INTERNAL_PROJECTION_COLUMN_MAP = {
    "fantasy_points_projection": "sportsbook_projection",
    "passing_yards_projection": "passing_yards_projection",
    "passing_td_projection": "passing_tds_projection",
    "rushing_yards_projection": "rushing_yards_projection",
    "rushing_td_projection": "rushing_tds_projection",
    "receptions_projection": "receptions_projection",
    "receiving_yards_projection": "receiving_yards_projection",
    "receiving_td_projection": "receiving_tds_projection",
}


def _normalize_internal_projection_df(projections_df, provider_name):
    if projections_df is None or projections_df.empty:
        return empty_provider_df(provider_name)

    out = pd.DataFrame()
    out["player_name"] = projections_df["player_name"]
    for source_col, target_col in INTERNAL_PROJECTION_COLUMN_MAP.items():
        if source_col in projections_df.columns:
            out[target_col] = projections_df[source_col]

    out["provider_name"] = provider_name
    out["last_updated"] = utc_timestamp()
    return normalize_provider_output(out, provider_name)


def _build_internal_projection_fallback():
    try:
        projections_df = build_player_projection_df()
    except Exception:
        return empty_provider_df("internal_projection_model")

    return _normalize_internal_projection_df(
        projections_df,
        provider_name="internal_projection_model",
    )


def _build_fantasypros_projection_fallback():
    try:
        players_df = load_players_df()
    except Exception:
        return empty_provider_df("fantasypros_projection_fallback")

    if players_df.empty:
        return empty_provider_df("fantasypros_projection_fallback")

    player_col = safe_col(players_df, PLAYER_COLS)
    projection_col = safe_col(players_df, PROJECTION_COLS)
    if player_col is None or projection_col is None:
        return empty_provider_df("fantasypros_projection_fallback")

    out = pd.DataFrame({
        "player_name": players_df[player_col],
        "sportsbook_projection": pd.to_numeric(
            players_df[projection_col],
            errors="coerce",
        ),
        "provider_name": "fantasypros_projection_fallback",
        "last_updated": utc_timestamp(),
    })
    out = out.dropna(subset=["sportsbook_projection"]).copy()

    return normalize_provider_output(out, "fantasypros_projection_fallback")


def fetch_active_provider_projection_data():
    provider = get_active_provider()
    if provider is None:
        return empty_provider_df("active_provider")

    try:
        projections_df = provider.fetch_projection_data()
        return provider.normalize_data(projections_df)
    except Exception:
        return empty_provider_df(provider.get_provider_name())


def fetch_sportsbook_projection_data():
    """
    Return canonical sportsbook-style projections using the configured fallback chain.

    Priority:
    1. Active sportsbook provider
    2. Internal projection model
    3. FantasyPros projection fallback
    """
    active_provider_df = fetch_active_provider_projection_data()
    if not active_provider_df.empty and active_provider_df["sportsbook_projection"].notna().any():
        active_provider_df.attrs["source"] = "active_provider"
        return active_provider_df

    internal_df = _build_internal_projection_fallback()
    if not internal_df.empty and internal_df["sportsbook_projection"].notna().any():
        internal_df.attrs["source"] = "internal_projection_model"
        return internal_df

    fantasypros_df = _build_fantasypros_projection_fallback()
    fantasypros_df.attrs["source"] = "fantasypros_projection_fallback"
    return fantasypros_df


def get_fallback_chain_debug_info():
    active_provider = get_active_provider()
    active_df = fetch_active_provider_projection_data()
    internal_df = _build_internal_projection_fallback()
    fantasypros_df = _build_fantasypros_projection_fallback()
    selected_df = fetch_sportsbook_projection_data()

    return {
        "active_provider": active_provider.get_provider_name() if active_provider else None,
        "selected_source": selected_df.attrs.get("source"),
        "sources": [
            {
                "source": "active_provider",
                "provider_name": active_provider.get_provider_name() if active_provider else None,
                "row_count": int(len(active_df)),
                "coverage": _projection_coverage(active_df),
            },
            {
                "source": "internal_projection_model",
                "provider_name": "internal_projection_model",
                "row_count": int(len(internal_df)),
                "coverage": _projection_coverage(internal_df),
            },
            {
                "source": "fantasypros_projection_fallback",
                "provider_name": "fantasypros_projection_fallback",
                "row_count": int(len(fantasypros_df)),
                "coverage": _projection_coverage(fantasypros_df),
            },
        ],
    }


def _projection_coverage(projections_df):
    if projections_df is None or projections_df.empty:
        return 0.0

    if "sportsbook_projection" not in projections_df.columns:
        return 0.0

    return round(float(projections_df["sportsbook_projection"].notna().mean() * 100.0), 2)


def validate_provider_projection_data(projections_df=None):
    projections_df = (
        fetch_sportsbook_projection_data()
        if projections_df is None
        else projections_df.copy()
    )

    missing_columns = [
        column for column in CANONICAL_PROVIDER_COLUMNS
        if column not in projections_df.columns
    ]
    return {
        "is_valid": not missing_columns and not projections_df.empty,
        "row_count": int(len(projections_df)),
        "missing_columns": missing_columns,
        "projection_coverage": _projection_coverage(projections_df),
        "provider_names": sorted(
            projections_df["provider_name"].dropna().unique().tolist()
        )
        if "provider_name" in projections_df.columns
        else [],
    }


def get_sportsbook_provider_debug_info():
    projections_df = fetch_sportsbook_projection_data()
    return {
        "registered_providers": list_providers(),
        "active_provider": (
            get_active_provider().get_provider_name()
            if get_active_provider()
            else None
        ),
        "health_report": get_provider_health_report(),
        "fallback_chain": get_fallback_chain_debug_info(),
        "selected_projection_validation": validate_provider_projection_data(projections_df),
        "sample_projection_rows": projections_df.head(10).to_dict("records")
        if not projections_df.empty
        else [],
    }


__all__ = [
    "SportsbookProvider",
    "MockSportsbookProvider",
    "register_provider",
    "get_provider",
    "list_providers",
    "get_active_provider",
    "set_active_provider",
    "get_provider_health_report",
    "fetch_active_provider_projection_data",
    "fetch_sportsbook_projection_data",
    "validate_provider_projection_data",
    "get_fallback_chain_debug_info",
    "get_sportsbook_provider_debug_info",
]
