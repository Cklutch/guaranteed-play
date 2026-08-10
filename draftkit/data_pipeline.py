from pathlib import Path
import re
import unicodedata

import pandas as pd

from draftkit.data_sources.fantasypros_source import load_fantasypros_projections
from draftkit.data_sources.sleeper_source import load_sleeper_players
from draftkit.data_sources.underdog_source import load_underdog_adp
from draftkit.feature_engineering import build_player_features_df


CANONICAL_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "team",
    "bye_week",
    "age",
    "injury_status",
    "projection_points",
    "projection_rank",
    "position_rank",
    "tier",
    "ceiling_projection",
    "floor_projection",
    "adp",
    "adp_rank",
    "expert_rank",
    "projection_source",
    "archetype_primary",
    "archetype_confidence",
    "injury_risk",
    "durability_grade",
    "boom_score",
    "bust_score",
    "stability_score",
    "archetype",
]
SCORABLE_POSITIONS = ["QB", "RB", "WR", "TE"]
MASTER_OUTPUT_PATH = Path("data/processed/master_players.csv")
# Minimum acceptable share of ranked players (real ADP/tier/expert rank present)
# that must also have a real projection. Below this, validate_player_data()
# marks the dataset invalid -- this is the guard against a repeat of the
# 27-of-3931 projection-coverage collapse found in the technical audit.
PROJECTION_COVERAGE_MIN_PCT = 90.0


def normalize_player_name(name):
    """
    Normalize player names into a stable matching key.
    """
    if name is None or pd.isna(name):
        return ""

    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower().strip()
    normalized = re.sub(r"['’]", "", normalized)
    normalized = re.sub(r"\b([a-z])\.", r"\1", normalized)
    normalized = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def _canonical_empty_df():
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _ensure_canonical_columns(df):
    if df.empty:
        return _canonical_empty_df()

    out = df.copy()
    for column in CANONICAL_COLUMNS:
        if column not in out.columns:
            out[column] = None

    return out[CANONICAL_COLUMNS]


def _prepare_source(df, source_name):
    out = _ensure_canonical_columns(df)
    if out.empty:
        out["player_key"] = []
        out["source"] = []
        return out

    if "player_key" in df.columns:
        out["player_key"] = df["player_key"]
    else:
        out["player_key"] = out["player_name"].apply(normalize_player_name)
    out = out[out["player_key"].astype(str) != ""].copy()
    out["source"] = source_name
    return out.reset_index(drop=True)


def _dedupe_source(df):
    if df.empty:
        return df

    if "player_id" in df.columns and df["player_id"].notna().any():
        id_df = df[df["player_id"].notna()].drop_duplicates("player_id", keep="first")
        no_id_df = df[df["player_id"].isna()].drop_duplicates("player_key", keep="first")
        return pd.concat([id_df, no_id_df], ignore_index=True)

    return df.drop_duplicates("player_key", keep="first").reset_index(drop=True)


def _source_duplicates(df):
    if df.empty:
        return {
            "duplicate_player_ids": [],
            "duplicate_player_names": [],
            "duplicate_normalized_names": [],
        }

    duplicate_player_ids = []
    if "player_id" in df.columns:
        duplicate_player_ids = (
            df.loc[df["player_id"].notna() & df["player_id"].duplicated(keep=False), "player_id"]
            .astype(str)
            .unique()
            .tolist()
        )

    duplicate_player_names = (
        df.loc[df["player_name"].duplicated(keep=False), "player_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "player_name" in df.columns
        else []
    )

    duplicate_normalized_names = (
        df.loc[df["player_key"].duplicated(keep=False), "player_key"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "player_key" in df.columns
        else []
    )

    return {
        "duplicate_player_ids": duplicate_player_ids,
        "duplicate_player_names": duplicate_player_names,
        "duplicate_normalized_names": duplicate_normalized_names,
    }


def _combine_values(primary, secondary):
    if primary is not None and not pd.isna(primary) and str(primary).strip() != "":
        return primary
    return secondary


def _merge_source(base_df, source_df, source_name):
    if base_df.empty:
        return source_df.copy(), []
    if source_df.empty:
        return base_df.copy(), []

    merged_rows = []
    matched_source_indexes = set()
    source_by_id = {}
    source_by_key = {}

    for source_index, source_row in source_df.iterrows():
        player_id = source_row.get("player_id")
        player_key = source_row.get("player_key")
        if player_id is not None and not pd.isna(player_id) and str(player_id).strip() != "":
            source_by_id[str(player_id)] = (source_index, source_row)
        if player_key:
            source_by_key[str(player_key)] = (source_index, source_row)

    for _, base_row in base_df.iterrows():
        match = None
        base_id = base_row.get("player_id")
        base_key = base_row.get("player_key")

        if base_id is not None and not pd.isna(base_id) and str(base_id) in source_by_id:
            match = source_by_id[str(base_id)]
        elif base_key in source_by_key:
            match = source_by_key[base_key]

        combined = base_row.to_dict()
        if match is not None:
            source_index, source_row = match
            matched_source_indexes.add(source_index)
            for column in CANONICAL_COLUMNS:
                combined[column] = _combine_values(combined.get(column), source_row.get(column))
            combined["source"] = f"{combined.get('source', '')}+{source_name}".strip("+")

        merged_rows.append(combined)

    unmatched_rows = []
    for source_index, source_row in source_df.iterrows():
        if source_index not in matched_source_indexes:
            unmatched_rows.append(source_row.to_dict())
            merged_rows.append(source_row.to_dict())

    return pd.DataFrame(merged_rows), unmatched_rows


def _dedupe_master_df(master_df):
    if master_df.empty or "player_key" not in master_df.columns:
        return master_df

    out = master_df.copy()
    out["_has_adp"] = out["adp"].notna().astype(int) if "adp" in out.columns else 0
    out["_has_projection"] = (
        out["projection_points"].notna().astype(int)
        if "projection_points" in out.columns
        else 0
    )
    out["_has_team"] = (
        out["team"].fillna("").astype(str).str.strip().ne("").astype(int)
        if "team" in out.columns
        else 0
    )
    out["_has_player_id"] = (
        out["player_id"].notna().astype(int)
        if "player_id" in out.columns
        else 0
    )
    out["_has_age"] = out["age"].notna().astype(int) if "age" in out.columns else 0

    out = out.sort_values(
        [
            "player_key",
            "_has_adp",
            "_has_projection",
            "_has_team",
            "_has_player_id",
            "_has_age",
        ],
        ascending=[True, False, False, False, False, False],
    )
    out = out.drop_duplicates("player_key", keep="first").copy()

    return out.drop(
        columns=[
            "_has_adp",
            "_has_projection",
            "_has_team",
            "_has_player_id",
            "_has_age",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def _attach_engineered_features(master_df):
    if master_df.empty:
        return master_df

    features_df = build_player_features_df()
    if features_df.empty:
        return master_df

    feature_lookup = {
        normalize_player_name(row["player_name"]): row.to_dict()
        for _, row in features_df.iterrows()
    }

    out = master_df.copy()
    for column in [
        "injury_risk",
        "durability_grade",
        "boom_score",
        "bust_score",
        "stability_score",
        "archetype",
    ]:
        if column in out.columns:
            out[column] = out[column].astype(object)

    for idx, row in out.iterrows():
        features = feature_lookup.get(row.get("player_key"))
        if not features:
            continue

        for column in [
            "injury_risk",
            "durability_grade",
            "boom_score",
            "bust_score",
            "stability_score",
            "archetype",
        ]:
            out.at[idx, column] = _combine_values(row.get(column), features.get(column))

    return out


ARCHETYPE_SOURCE_PATH = Path("research/validation_v1/data/current_season_archetypes.csv")


def _attach_current_archetypes(master_df):
    """
    Attach draft-time player archetypes (bellcow RB, deep threat WR, etc.).

    These are for roster construction and explanation only -- they are
    deliberately NOT part of scoring. Validation showed archetype features
    do not beat ADP as model inputs, so wiring them into the score would be
    adding unproven signal. What they're for is telling the drafter what
    kind of player, and what kind of risk, they're taking on.

    Regenerate with:
        research/validation_v1/build_current_season_archetypes.py
    """
    out = master_df.copy()
    out["archetype_primary"] = None
    out["archetype_confidence"] = None

    if not ARCHETYPE_SOURCE_PATH.exists():
        return out

    try:
        archetypes = pd.read_csv(ARCHETYPE_SOURCE_PATH)
    except Exception:
        return out

    if archetypes.empty or "player_name" not in archetypes.columns:
        return out

    archetypes = archetypes[["player_name", "archetype_primary", "archetype_confidence"]].copy()
    archetypes["player_key"] = archetypes["player_name"].apply(normalize_player_name)
    archetypes = archetypes.drop(columns=["player_name"]).drop_duplicates("player_key", keep="first")

    out = out.drop(columns=["archetype_primary", "archetype_confidence"])
    return out.merge(archetypes, on="player_key", how="left")


def merge_player_sources(sleeper_df=None, fantasypros_df=None, underdog_df=None):
    """
    Merge Sleeper, FantasyPros, and Underdog data into one canonical dataframe.
    """
    sleeper_df = load_sleeper_players() if sleeper_df is None else sleeper_df
    fantasypros_df = load_fantasypros_projections() if fantasypros_df is None else fantasypros_df
    underdog_df = load_underdog_adp() if underdog_df is None else underdog_df

    sources = {
        "sleeper": _dedupe_source(_prepare_source(sleeper_df, "sleeper")),
        "fantasypros": _dedupe_source(_prepare_source(fantasypros_df, "fantasypros")),
        "underdog": _dedupe_source(_prepare_source(underdog_df, "underdog")),
    }

    base_df = sources["sleeper"].copy()
    if base_df.empty:
        base_df = sources["fantasypros"].copy()
    if base_df.empty:
        base_df = sources["underdog"].copy()

    unmatched = {
        "fantasypros": [],
        "underdog": [],
    }

    if not sources["fantasypros"].empty and not base_df.equals(sources["fantasypros"]):
        base_df, unmatched["fantasypros"] = _merge_source(
            base_df,
            sources["fantasypros"],
            "fantasypros",
        )

    if not sources["underdog"].empty and not base_df.equals(sources["underdog"]):
        base_df, unmatched["underdog"] = _merge_source(
            base_df,
            sources["underdog"],
            "underdog",
        )

    if base_df.empty:
        out = _canonical_empty_df()
    else:
        base_df = _attach_engineered_features(base_df)
        out = _ensure_canonical_columns(base_df)
        out["player_key"] = out["player_name"].apply(normalize_player_name)
        out = out[out["position"].astype(str).str.upper().isin(SCORABLE_POSITIONS)].copy()
        out = _dedupe_master_df(out)

        numeric_cols = [
            "bye_week",
            "age",
            "projection_points",
            "projection_rank",
            "position_rank",
            "tier",
            "ceiling_projection",
            "floor_projection",
            "adp",
            "adp_rank",
            "expert_rank",
            "injury_risk",
            "durability_grade",
            "boom_score",
            "bust_score",
            "stability_score",
        ]
        for column in numeric_cols:
            out[column] = pd.to_numeric(out[column], errors="coerce")

        out["projection_source"] = out["projection_points"].notna().map(
            {True: "real", False: None}
        )
        out = _attach_current_archetypes(out)

        if out["projection_rank"].isna().all() and out["projection_points"].notna().any():
            out["projection_rank"] = out["projection_points"].rank(
                ascending=False,
                method="min",
            )

        if out["adp_rank"].isna().all() and out["adp"].notna().any():
            out["adp_rank"] = out["adp"].rank(
                ascending=True,
                method="min",
            )

        out = out.sort_values(
            ["projection_rank", "adp_rank", "projection_points"],
            ascending=[True, True, False],
            na_position="last",
        ).reset_index(drop=True)
        out = out[CANONICAL_COLUMNS + ["player_key"]]

    out.attrs["source_shapes"] = {
        name: source_df.shape for name, source_df in sources.items()
    }
    out.attrs["source_paths"] = {
        "sleeper": sleeper_df.attrs.get("source_path") or sleeper_df.attrs.get("source_url"),
        "fantasypros": fantasypros_df.attrs.get("source_path") or fantasypros_df.attrs.get("source_file"),
        "underdog": underdog_df.attrs.get("source_path"),
    }
    out.attrs["source_duplicates"] = {
        name: _source_duplicates(source_df)
        for name, source_df in sources.items()
    }
    out.attrs["unmatched_players"] = {
        name: [
            row.get("player_name")
            for row in rows
            if row.get("player_name")
        ]
        for name, rows in unmatched.items()
    }

    return out


def validate_player_data(master_df=None):
    """
    Validate the canonical player dataset and return diagnostics.
    """
    master_df = merge_player_sources() if master_df is None else master_df.copy()

    if master_df.empty:
        return {
            "is_valid": False,
            "row_count": 0,
            "missing_columns": CANONICAL_COLUMNS,
            "duplicate_player_ids": [],
            "duplicate_normalized_names": [],
            "missing_data_counts": {},
            "unmatched_players": master_df.attrs.get("unmatched_players", {}),
            "messages": ["Master dataset is empty."],
        }

    missing_columns = [column for column in CANONICAL_COLUMNS if column not in master_df.columns]
    player_key = (
        master_df["player_key"]
        if "player_key" in master_df.columns
        else master_df["player_name"].apply(normalize_player_name)
    )

    duplicate_player_ids = []
    if "player_id" in master_df.columns:
        duplicate_player_ids = (
            master_df.loc[
                master_df["player_id"].notna() & master_df["player_id"].duplicated(keep=False),
                "player_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

    duplicate_normalized_names = (
        player_key[player_key.duplicated(keep=False)]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    missing_data_counts = {
        column: int(master_df[column].isna().sum())
        for column in CANONICAL_COLUMNS
        if column in master_df.columns
    }

    # A player is "ranked" if any external source placed them on a board at
    # all (real ADP or a FantasyPros tier/expert rank). Coverage is measured
    # against this group, not the full Sleeper-metadata pool of thousands of
    # bench/practice-squad players who were never going to have projections.
    ranked_mask = pd.Series(False, index=master_df.index)
    for column in ["adp", "tier", "expert_rank"]:
        if column in master_df.columns:
            ranked_mask = ranked_mask | master_df[column].notna()
    ranked_count = int(ranked_mask.sum())
    ranked_projection_coverage = (
        round(float(master_df.loc[ranked_mask, "projection_points"].notna().mean()) * 100.0, 2)
        if ranked_count and "projection_points" in master_df.columns
        else 0.0
    )
    coverage_ok = ranked_count == 0 or ranked_projection_coverage >= PROJECTION_COVERAGE_MIN_PCT

    messages = []
    if missing_columns:
        messages.append("One or more canonical columns are missing.")
    if duplicate_player_ids:
        messages.append("Duplicate player IDs detected.")
    if duplicate_normalized_names:
        messages.append("Duplicate normalized player names detected.")
    if missing_data_counts.get("projection_points", 0) > 0:
        messages.append("Some players are missing real projection points.")
    if missing_data_counts.get("adp", 0) > 0:
        messages.append("Some players are missing real ADP.")
    if missing_data_counts.get("position", 0) > 0:
        messages.append("Some players are missing position.")
    if not coverage_ok:
        messages.append(
            "CRITICAL: only {0}% of the {1} ranked players (real ADP/tier/expert "
            "rank present) have a real projection. The recommendation engine "
            "will be effectively blind for most of the ranked player pool "
            "until a real projections export is added to data/raw/. Threshold "
            "is {2}%.".format(ranked_projection_coverage, ranked_count, PROJECTION_COVERAGE_MIN_PCT)
        )

    return {
        "is_valid": (
            not missing_columns
            and not duplicate_player_ids
            and not duplicate_normalized_names
            and coverage_ok
        ),
        "row_count": int(len(master_df)),
        "missing_columns": missing_columns,
        "duplicate_player_ids": duplicate_player_ids,
        "duplicate_normalized_names": duplicate_normalized_names,
        "missing_data_counts": missing_data_counts,
        "source_duplicates": master_df.attrs.get("source_duplicates", {}),
        "unmatched_players": master_df.attrs.get("unmatched_players", {}),
        "ranked_player_count": ranked_count,
        "ranked_projection_coverage": ranked_projection_coverage,
        "projection_coverage_ok": coverage_ok,
        "projection_coverage_min_pct": PROJECTION_COVERAGE_MIN_PCT,
        "projection_coverage": round(
            float(master_df["projection_points"].notna().mean()) * 100.0,
            2,
        )
        if "projection_points" in master_df.columns and len(master_df)
        else 0.0,
        "tier_coverage": round(
            float(master_df["tier"].notna().mean()) * 100.0,
            2,
        )
        if "tier" in master_df.columns and len(master_df)
        else 0.0,
        "messages": messages,
    }


def build_master_dataset(
    output_path=MASTER_OUTPUT_PATH,
    sleeper_df=None,
    fantasypros_df=None,
    underdog_df=None,
):
    """
    Build and write the canonical production player dataset.
    """
    master_df = merge_player_sources(
        sleeper_df=sleeper_df,
        fantasypros_df=fantasypros_df,
        underdog_df=underdog_df,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_df = master_df[CANONICAL_COLUMNS].copy() if not master_df.empty else _canonical_empty_df()
    write_df.to_csv(output_path, index=False)
    write_df.attrs.update(master_df.attrs)
    write_df.attrs["output_path"] = str(output_path)

    return write_df


def get_pipeline_debug_info():
    """
    Return diagnostics for the multi-source ingestion framework.
    """
    sleeper_df = load_sleeper_players()
    fantasypros_df = load_fantasypros_projections()
    underdog_df = load_underdog_adp()
    master_df = merge_player_sources(sleeper_df, fantasypros_df, underdog_df)
    validation = validate_player_data(master_df)

    return {
        "source_shapes": master_df.attrs.get("source_shapes", {}),
        "source_paths": master_df.attrs.get("source_paths", {}),
        "source_duplicates": master_df.attrs.get("source_duplicates", {}),
        "unmatched_players": master_df.attrs.get("unmatched_players", {}),
        "master_shape": master_df.shape,
        "output_path": str(MASTER_OUTPUT_PATH),
        "validation": validation,
        "sample_rows": master_df[CANONICAL_COLUMNS].head(10).to_dict("records")
        if not master_df.empty
        else [],
    }


# Backward-compatible wrappers from the prototype pipeline.
def build_master_player_table():
    return merge_player_sources()[CANONICAL_COLUMNS].reset_index(drop=True)


def validate_master_table(master_df=None):
    return validate_player_data(master_df)


def match_players(projection_df=None, adp_df=None, injury_df=None):
    return merge_player_sources(
        sleeper_df=injury_df,
        fantasypros_df=projection_df,
        underdog_df=adp_df,
    )


def load_projection_data():
    return load_fantasypros_projections()


def load_adp_data():
    return load_underdog_adp()


def load_injury_data():
    return load_sleeper_players()
