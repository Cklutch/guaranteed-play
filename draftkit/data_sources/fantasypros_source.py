from pathlib import Path
import re
import unicodedata

import pandas as pd

from draftkit.data_access import safe_col


FANTASYPROS_OUTPUT_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "team",
    "bye_week",
    "projection_points",
    "projection_rank",
    "position_rank",
    "tier",
    "ceiling_projection",
    "floor_projection",
    "adp",
    "adp_rank",
    "player_key",
]

FANTASYPROS_SOURCE_PATTERNS = [
    "*fantasypros*projection*.csv",
    "*fantasypros*ranking*.csv",
    "*fantasy_pros*projection*.csv",
    "*fantasy_pros*ranking*.csv",
]
DEFAULT_SOURCE_DIR = Path("data/raw")


def normalize_fantasypros_name(name):
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


def _empty_fantasypros_df(source_file=None, error=None):
    df = pd.DataFrame(columns=FANTASYPROS_OUTPUT_COLUMNS)
    df.attrs["source_file"] = str(source_file) if source_file else None
    df.attrs["error"] = error
    df.attrs["validation"] = validate_fantasypros_projections(df)
    df.attrs["missing_fields"] = [
        "player_name",
        "position",
        "team",
        "bye_week",
        "projection_points",
        "projection_rank",
        "position_rank",
        "tier",
        "ceiling_projection",
        "floor_projection",
        "adp",
        "adp_rank",
    ]
    return df


def _find_source_file(source_path=None, source_dir=DEFAULT_SOURCE_DIR):
    if source_path:
        path = Path(source_path)
        return path if path.exists() else None

    source_dir = Path(source_dir)
    if not source_dir.exists():
        return None

    matches = []
    for pattern in FANTASYPROS_SOURCE_PATTERNS:
        matches.extend(source_dir.glob(pattern))

    matches = sorted(set(matches), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def map_fantasypros_columns(df):
    """
    Detect common FantasyPros export columns.
    """
    return {
        "player_id": safe_col(df, ["player_id", "fantasypros_id", "id", "ID"]),
        "player_name": safe_col(df, ["player", "player_name", "name", "Player", "PLAYER", "PLAYER NAME"]),
        "position": safe_col(df, ["position", "pos", "Position", "POS"]),
        "team": safe_col(df, ["team", "Team", "TEAM"]),
        "bye_week": safe_col(df, ["bye_week", "bye", "Bye", "BYE", "Bye Week", "BYE WEEK"]),
        "projection_points": safe_col(
            df,
            ["projection_points", "fpts", "fantasy_points", "FPTS", "Fantasy Points", "Proj", "Projection"],
        ),
        "projection_rank": safe_col(df, ["projection_rank", "proj_rank", "Projection Rank"]),
        "position_rank": safe_col(df, ["position_rank", "pos_rank", "Pos Rank", "Position Rank"]),
        "tier": safe_col(df, ["tier", "Tier", "TIERS"]),
        "ceiling_projection": safe_col(df, ["ceiling_projection", "ceiling", "Ceiling"]),
        "floor_projection": safe_col(df, ["floor_projection", "floor", "Floor"]),
        "adp": safe_col(df, ["adp", "ADP", "avg_adp", "Average ADP"]),
        "adp_rank": safe_col(df, ["adp_rank", "ADP Rank", "rank", "Rank", "RANK", "RK"]),
    }


def _parse_position(value):
    text = str(value or "").strip().upper()
    match = re.match(r"([A-Z]+)", text)
    return match.group(1) if match else text


def _parse_position_rank(value):
    text = str(value or "").strip().upper()
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else pd.NA


def normalize_fantasypros_players(df):
    """
    Add normalized player keys for downstream source matching.
    """
    out = df.copy()
    if "player_name" not in out.columns:
        out["player_key"] = []
        return out

    out["player_key"] = out["player_name"].apply(normalize_fantasypros_name)
    return out


def _coerce_numeric_columns(df):
    out = df.copy()
    for column in [
        "bye_week",
        "projection_points",
        "projection_rank",
        "position_rank",
        "tier",
        "ceiling_projection",
        "floor_projection",
        "adp",
        "adp_rank",
    ]:
        if column in out.columns:
            out[column] = (
                out[column]
                .astype(str)
                .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
            )
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def validate_fantasypros_projections(projections_df):
    """
    Validate FantasyPros projection output without throwing exceptions.
    """
    try:
        if projections_df.empty:
            return {
                "row_count": 0,
                "duplicate_player_names": [],
                "missing_projection_count": 0,
                "missing_rank_count": 0,
                "missing_position_count": 0,
                "is_valid": False,
                "messages": ["FantasyPros projections are empty."],
            }

        duplicate_player_names = (
            projections_df.loc[
                projections_df["player_key"].astype(str).ne("")
                & projections_df["player_key"].duplicated(keep=False),
                "player_name",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if "player_key" in projections_df.columns
            else []
        )

        missing_projection_count = int(projections_df["projection_points"].isna().sum())
        missing_rank_count = int(projections_df["projection_rank"].isna().sum())
        missing_position_count = int(
            projections_df["position"].isna().sum()
            + projections_df["position"].astype(str).str.strip().eq("").sum()
        )

        messages = []
        if duplicate_player_names:
            messages.append("Duplicate normalized FantasyPros player names detected.")
        if missing_projection_count:
            messages.append("Some FantasyPros rows are missing projections.")
        if missing_rank_count:
            messages.append("Some FantasyPros rows are missing projection ranks.")
        if missing_position_count:
            messages.append("Some FantasyPros rows are missing positions.")

        return {
            "row_count": int(len(projections_df)),
            "duplicate_player_names": duplicate_player_names,
            "missing_projection_count": missing_projection_count,
            "missing_rank_count": missing_rank_count,
            "missing_position_count": missing_position_count,
            "is_valid": len(projections_df) > 0 and not duplicate_player_names,
            "messages": messages,
        }
    except Exception as exc:
        return {
            "row_count": 0,
            "duplicate_player_names": [],
            "missing_projection_count": 0,
            "missing_rank_count": 0,
            "missing_position_count": 0,
            "is_valid": False,
            "messages": [f"FantasyPros validation failed: {exc}"],
        }


def load_fantasypros_projections(source_path=None, source_dir=DEFAULT_SOURCE_DIR):
    """
    Load a FantasyPros projections CSV into a normalized projection dataframe.
    """
    source_file = _find_source_file(source_path=source_path, source_dir=source_dir)
    if source_file is None:
        return _empty_fantasypros_df()

    try:
        raw_df = pd.read_csv(source_file)
    except Exception as exc:
        return _empty_fantasypros_df(source_file=source_file, error=str(exc))

    if raw_df.empty:
        return _empty_fantasypros_df(source_file=source_file)

    column_map = map_fantasypros_columns(raw_df)
    rows = []
    for _, row in raw_df.iterrows():
        player_name_col = column_map.get("player_name")
        player_name = row.get(player_name_col) if player_name_col else None
        if not player_name or pd.isna(player_name):
            continue

        rows.append({
            "player_id": row.get(column_map["player_id"])
            if column_map.get("player_id")
            else pd.NA,
            "player_name": str(player_name).strip(),
            "position": _parse_position(row.get(column_map["position"], ""))
            if column_map.get("position")
            else "",
            "team": str(row.get(column_map["team"], ""))
            if column_map.get("team")
            else "",
            "bye_week": row.get(column_map["bye_week"])
            if column_map.get("bye_week")
            else pd.NA,
            "projection_points": row.get(column_map["projection_points"])
            if column_map.get("projection_points")
            else pd.NA,
            "projection_rank": row.get(column_map["projection_rank"])
            if column_map.get("projection_rank")
            else pd.NA,
            "position_rank": (
                row.get(column_map["position_rank"])
                if column_map.get("position_rank")
                else _parse_position_rank(row.get(column_map["position"], ""))
            ),
            "tier": row.get(column_map["tier"]) if column_map.get("tier") else pd.NA,
            "ceiling_projection": row.get(column_map["ceiling_projection"])
            if column_map.get("ceiling_projection")
            else pd.NA,
            "floor_projection": row.get(column_map["floor_projection"])
            if column_map.get("floor_projection")
            else pd.NA,
            "adp": (
                row.get(column_map["adp"])
                if column_map.get("adp")
                else row.get(column_map["adp_rank"])
                if column_map.get("adp_rank")
                else pd.NA
            ),
            "adp_rank": row.get(column_map["adp_rank"])
            if column_map.get("adp_rank")
            else pd.NA,
        })

    out = pd.DataFrame(rows, columns=[col for col in FANTASYPROS_OUTPUT_COLUMNS if col != "player_key"])
    out = normalize_fantasypros_players(out)
    out = _coerce_numeric_columns(out)

    if out["projection_rank"].isna().all() and out["projection_points"].notna().any():
        out = out.sort_values("projection_points", ascending=False).reset_index(drop=True)
        out["projection_rank"] = out.index + 1

    validation = validate_fantasypros_projections(out)
    missing_fields = []
    for canonical_col, source_col in column_map.items():
        if source_col is not None:
            continue
        if canonical_col == "position_rank" and column_map.get("position"):
            continue
        if canonical_col == "adp" and column_map.get("adp_rank"):
            continue
        missing_fields.append(canonical_col)

    out.attrs["source_file"] = str(source_file)
    out.attrs["column_map"] = column_map
    out.attrs["missing_fields"] = missing_fields
    out.attrs["validation"] = validation
    out.attrs["error"] = None

    return out[FANTASYPROS_OUTPUT_COLUMNS]


def get_fantasypros_source_debug_info(source_path=None, source_dir=DEFAULT_SOURCE_DIR):
    projections_df = load_fantasypros_projections(
        source_path=source_path,
        source_dir=source_dir,
    )
    validation = projections_df.attrs.get(
        "validation",
        validate_fantasypros_projections(projections_df),
    )

    rows_loaded = int(len(projections_df))
    projection_coverage = (
        round(float(projections_df["projection_points"].notna().mean()) * 100.0, 2)
        if rows_loaded and "projection_points" in projections_df.columns
        else 0.0
    )
    tier_coverage = (
        round(float(projections_df["tier"].notna().mean()) * 100.0, 2)
        if rows_loaded and "tier" in projections_df.columns
        else 0.0
    )

    return {
        "source_file": projections_df.attrs.get("source_file"),
        "rows_loaded": rows_loaded,
        "projection_coverage": projection_coverage,
        "tier_coverage": tier_coverage,
        "missing_fields": projections_df.attrs.get("missing_fields", []),
        "duplicate_players": validation.get("duplicate_player_names", []),
        "validation": validation,
        "error": projections_df.attrs.get("error"),
        "sample_rows": projections_df.head(10).to_dict("records")
        if rows_loaded
        else [],
    }
