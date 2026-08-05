import re
import unicodedata

import pandas as pd

from draftkit.sportsbook_provider import fetch_sportsbook_projection_data


MASTER_PLAYERS_PATH = "data/processed/master_players.csv"

PLAYER_NAME_ALIASES = {
    "gabe davis": "gabriel davis",
    "hollywood brown": "marquise brown",
    "dj moore": "dj moore",
    "d j moore": "dj moore",
    "juju smith schuster": "juju smithschuster",
    "kenneth walker": "kenneth walker",
    "kenneth walker iii": "kenneth walker",
}

SPORTSBOOK_COLUMNS = [
    "player_name",
    "sportsbook_projection",
    "sportsbook_projection_rank",
    "passing_yards_projection",
    "passing_tds_projection",
    "rushing_yards_projection",
    "rushing_tds_projection",
    "receptions_projection",
    "receiving_yards_projection",
    "receiving_tds_projection",
    "provider_name",
    "last_updated",
]

ENRICHMENT_COLUMNS = [
    "sportsbook_projection",
    "sportsbook_projection_rank",
    "sportsbook_provider",
    "sportsbook_last_updated",
    "projection_gap",
    "projection_gap_pct",
]

EXISTING_PROJECTION_COLS = [
    "projection_points",
    "fantasy_points_projection",
    "market_projection",
    "Projection",
    "projected_points",
    "FPTS",
]


def normalize_player_name(name):
    if name is None or pd.isna(name):
        return ""

    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[`'’]", "", normalized)
    normalized = re.sub(r"\b([a-z])\.", r"\1", normalized)
    normalized = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return PLAYER_NAME_ALIASES.get(normalized, normalized)


def _load_master_players_df(master_path=MASTER_PLAYERS_PATH):
    try:
        return pd.read_csv(master_path)
    except Exception:
        return pd.DataFrame()


def _safe_float(value, default=pd.NA):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_col(df, columns):
    for column in columns:
        if column in df.columns:
            return column
    return None


def _prepare_master_df(master_df=None):
    df = _load_master_players_df() if master_df is None else master_df.copy()
    if df.empty or "player_name" not in df.columns:
        return pd.DataFrame()

    df["normalized_player_name"] = df["player_name"].apply(normalize_player_name)
    return df


def load_sportsbook_projections(projections_df=None):
    projections = (
        fetch_sportsbook_projection_data()
        if projections_df is None
        else projections_df.copy()
    )

    if projections.empty:
        return pd.DataFrame(columns=SPORTSBOOK_COLUMNS + ["normalized_player_name"])

    for column in SPORTSBOOK_COLUMNS:
        if column not in projections.columns:
            projections[column] = pd.NA

    projections["normalized_player_name"] = projections["player_name"].apply(
        normalize_player_name
    )
    projections["sportsbook_projection"] = pd.to_numeric(
        projections["sportsbook_projection"],
        errors="coerce",
    )
    projections["sportsbook_projection_rank"] = calculate_sportsbook_projection_rank(
        projections,
    )

    return projections[SPORTSBOOK_COLUMNS + ["normalized_player_name"]].copy()


def calculate_sportsbook_projection_rank(projections_df):
    if projections_df is None or projections_df.empty:
        return pd.Series(dtype="float64")

    if (
        "sportsbook_projection_rank" in projections_df.columns
        and projections_df["sportsbook_projection_rank"].notna().any()
    ):
        return pd.to_numeric(
            projections_df["sportsbook_projection_rank"],
            errors="coerce",
        )

    if "sportsbook_projection" not in projections_df.columns:
        return pd.Series([pd.NA] * len(projections_df), index=projections_df.index)

    projections = pd.to_numeric(
        projections_df["sportsbook_projection"],
        errors="coerce",
    )
    if projections.notna().sum() == 0:
        return pd.Series([pd.NA] * len(projections_df), index=projections_df.index)

    return projections.rank(method="first", ascending=False)


def calculate_projection_disagreement(sportsbook_projection, existing_projection):
    sportsbook_projection = _safe_float(sportsbook_projection)
    existing_projection = _safe_float(existing_projection)

    if pd.isna(sportsbook_projection) or pd.isna(existing_projection):
        return {
            "projection_gap": pd.NA,
            "projection_gap_pct": pd.NA,
        }

    projection_gap = sportsbook_projection - existing_projection
    if existing_projection == 0:
        projection_gap_pct = pd.NA
    else:
        projection_gap_pct = (projection_gap / abs(existing_projection)) * 100.0

    return {
        "projection_gap": round(float(projection_gap), 3),
        "projection_gap_pct": round(float(projection_gap_pct), 3)
        if not pd.isna(projection_gap_pct)
        else pd.NA,
    }


def merge_sportsbook_projections(master_df=None, projections_df=None):
    master = _prepare_master_df(master_df)
    projections = load_sportsbook_projections(projections_df)

    if master.empty:
        return pd.DataFrame()

    if projections.empty:
        enriched = master.copy()
        for column in ENRICHMENT_COLUMNS:
            enriched[column] = pd.NA
        enriched.attrs["master_rows"] = int(len(master))
        enriched.attrs["sportsbook_rows"] = 0
        enriched.attrs["matched_players"] = 0
        enriched.attrs["unmatched_players"] = []
        return enriched

    projection_subset = projections.drop_duplicates(
        "normalized_player_name",
        keep="first",
    ).copy()
    projection_subset = projection_subset.rename(
        columns={
            "provider_name": "sportsbook_provider",
            "last_updated": "sportsbook_last_updated",
            "player_name": "sportsbook_player_name",
        }
    )

    enriched = master.merge(
        projection_subset,
        on="normalized_player_name",
        how="left",
        suffixes=("", "_sportsbook"),
    )

    existing_projection_col = _safe_col(enriched, EXISTING_PROJECTION_COLS)
    existing_projection = (
        pd.to_numeric(enriched[existing_projection_col], errors="coerce")
        if existing_projection_col
        else pd.Series([pd.NA] * len(enriched), index=enriched.index)
    )

    disagreement = enriched.apply(
        lambda row: calculate_projection_disagreement(
            row.get("sportsbook_projection"),
            row.get(existing_projection_col) if existing_projection_col else pd.NA,
        ),
        axis=1,
    )
    enriched["projection_gap"] = disagreement.apply(lambda item: item["projection_gap"])
    enriched["projection_gap_pct"] = disagreement.apply(
        lambda item: item["projection_gap_pct"]
    )

    if "sportsbook_provider" not in enriched.columns:
        enriched["sportsbook_provider"] = pd.NA
    if "sportsbook_last_updated" not in enriched.columns:
        enriched["sportsbook_last_updated"] = pd.NA

    # Keep a sportsbook-compatible projection field for legacy consumers.
    enriched["market_projection"] = enriched["sportsbook_projection"]
    enriched["projection_source"] = enriched["sportsbook_provider"].where(
        enriched["sportsbook_projection"].notna(),
        pd.NA,
    )
    if "fantasy_points_projection" not in enriched.columns:
        enriched["fantasy_points_projection"] = enriched["sportsbook_projection"]

    matched_mask = enriched["sportsbook_projection"].notna()
    matched_players = int(matched_mask.sum())
    master_keys = set(master["normalized_player_name"].dropna())
    unmatched_projection_names = projections[
        ~projections["normalized_player_name"].isin(master_keys)
    ]["player_name"].dropna().astype(str).tolist()

    enriched.attrs["master_rows"] = int(len(master))
    enriched.attrs["sportsbook_rows"] = int(len(projections))
    enriched.attrs["matched_players"] = matched_players
    enriched.attrs["unmatched_players"] = unmatched_projection_names
    enriched.attrs["existing_projection_column"] = existing_projection_col
    enriched.attrs["provider_name"] = (
        str(projections["provider_name"].dropna().iloc[0])
        if "provider_name" in projections.columns and projections["provider_name"].notna().any()
        else None
    )

    return enriched


def build_enriched_player_dataset(master_df=None, projections_df=None):
    enriched = merge_sportsbook_projections(
        master_df=master_df,
        projections_df=projections_df,
    )
    if enriched.empty:
        return enriched

    drop_cols = [
        "normalized_player_name",
        "sportsbook_player_name",
    ]
    output = enriched.drop(columns=[col for col in drop_cols if col in enriched.columns])
    return output.copy()


def calculate_projection_confidence(player_projection_row):
    projection_columns = [
        "passing_yards_projection",
        "passing_tds_projection",
        "rushing_yards_projection",
        "rushing_tds_projection",
        "receptions_projection",
        "receiving_yards_projection",
        "receiving_tds_projection",
    ]
    available_count = sum(
        1
        for column in projection_columns
        if column in player_projection_row and not pd.isna(player_projection_row[column])
    )
    if available_count <= 0:
        return 0.0

    return round(min(100.0, (available_count / len(projection_columns)) * 100.0), 2)


def calculate_projection_variance(player_projection_row):
    projection_columns = [
        "passing_yards_projection",
        "passing_tds_projection",
        "rushing_yards_projection",
        "rushing_tds_projection",
        "receptions_projection",
        "receiving_yards_projection",
        "receiving_tds_projection",
    ]
    values = []
    for column in projection_columns:
        value = _safe_float(player_projection_row.get(column))
        if not pd.isna(value):
            values.append(value)

    if len(values) < 2:
        return pd.NA

    return round(float(pd.Series(values).var()), 3)


def get_projection_enrichment_debug_info(master_df=None, projections_df=None):
    master = _prepare_master_df(master_df)
    projections = load_sportsbook_projections(projections_df)
    enriched = merge_sportsbook_projections(
        master_df=master,
        projections_df=projections,
    )

    if enriched.empty:
        return {
            "master_rows": int(len(master)),
            "sportsbook_rows": int(len(projections)),
            "matched_players": 0,
            "unmatched_players": projections["player_name"].dropna().tolist()
            if "player_name" in projections.columns
            else [],
            "projection_coverage": 0.0,
            "provider_name": None,
            "example_enriched_rows": [],
        }

    matched_players = int(enriched["sportsbook_projection"].notna().sum())
    projection_coverage = (
        round((matched_players / len(enriched)) * 100.0, 2)
        if len(enriched)
        else 0.0
    )
    provider_name = enriched.attrs.get("provider_name")

    if provider_name is None and "sportsbook_provider" in enriched.columns:
        provider_values = enriched["sportsbook_provider"].dropna()
        provider_name = str(provider_values.iloc[0]) if not provider_values.empty else None

    return {
        "master_rows": int(enriched.attrs.get("master_rows", len(master))),
        "sportsbook_rows": int(enriched.attrs.get("sportsbook_rows", len(projections))),
        "matched_players": matched_players,
        "unmatched_players": enriched.attrs.get("unmatched_players", []),
        "projection_coverage": projection_coverage,
        "provider_name": provider_name,
        "existing_projection_column": enriched.attrs.get("existing_projection_column"),
        "example_enriched_rows": build_enriched_player_dataset(
            master_df=master,
            projections_df=projections,
        ).head(10).to_dict("records")
        if not enriched.empty
        else [],
    }


# Backward-compatible wrappers from the earlier market projection layer.
def merge_market_projections(master_df=None, projections_df=None):
    return merge_sportsbook_projections(master_df=master_df, projections_df=projections_df)


def build_projection_enriched_dataset(master_df=None, projections_df=None):
    return build_enriched_player_dataset(master_df=master_df, projections_df=projections_df)
