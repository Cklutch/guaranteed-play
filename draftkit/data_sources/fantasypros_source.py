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
    "expert_rank",
    "player_key",
]

# Two distinct export types: a rankings/ECR export (RK, TIERS, POS, no FPTS) and a
# points-projection export (has an actual FPTS/points column, no RK/tier). They are
# loaded and merged separately -- do not assume a single file has both.
FANTASYPROS_RANKING_PATTERNS = [
    "*fantasypros*ranking*.csv",
    "*fantasy_pros*ranking*.csv",
]
FANTASYPROS_PROJECTION_PATTERNS = [
    "*fantasypros*projection*.csv",
    "*fantasy_pros*projection*.csv",
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
    df.attrs["ranking_source_file"] = None
    df.attrs["projection_source_file"] = None
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
        "expert_rank",
    ]
    return df


def _find_source_file(patterns, source_path=None, source_dir=DEFAULT_SOURCE_DIR):
    if source_path:
        path = Path(source_path)
        return path if path.exists() else None

    source_dir = Path(source_dir)
    if not source_dir.exists():
        return None

    matches = []
    for pattern in patterns:
        matches.extend(source_dir.glob(pattern))

    matches = sorted(set(matches), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _find_all_source_files(patterns, source_dir=DEFAULT_SOURCE_DIR):
    """
    Return every file matching any of the given patterns, not just the
    newest. FantasyPros' real projections export is split into one file per
    position (QB/RB/WR/TE) -- loading only the single most-recently-modified
    match silently dropped the other three positions.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        return []

    matches = []
    for pattern in patterns:
        matches.extend(source_dir.glob(pattern))

    return sorted(set(matches))


def map_fantasypros_columns(df):
    """
    Detect common FantasyPros export columns.

    NOTE: "adp" / "adp_rank" only ever map to columns that are genuinely
    labeled as ADP. FantasyPros' consensus "RK" (rank) column is a different
    quantity -- expert rank, not market average draft position -- and is
    mapped separately to "expert_rank". Do not fold RK into adp_rank; a
    prior version of this loader did that, which silently mislabeled expert
    rank as ADP throughout the app.
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
        "adp_rank": safe_col(df, ["adp_rank", "ADP Rank"]),
        "expert_rank": safe_col(df, ["expert_rank", "ECR", "expert_consensus_rank", "RK", "Rank", "RANK", "rank"]),
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
        "expert_rank",
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


_FILENAME_POSITION_PATTERN = re.compile(r"(?i)(?:^|[_\-])(QB|RB|WR|TE|DST|DEF|K)(?:[_\-.]|$)")


def _infer_position_from_filename(source_file):
    """
    FantasyPros' real projections export is split into one file per position
    (e.g. FantasyPros_Fantasy_Football_Projections_QB.csv), and those files
    have no "position" column at all -- every row is implicitly that
    position. Fall back to reading it off the filename in that case.
    """
    if source_file is None:
        return ""
    match = _FILENAME_POSITION_PATTERN.search(Path(source_file).stem)
    return match.group(1).upper() if match else ""


def _load_single_fantasypros_file(source_file):
    """
    Parse one FantasyPros export file into the canonical row shape.
    Any canonical field not present in this particular export is left NA --
    the caller is responsible for merging multiple exports together.
    """
    try:
        raw_df = pd.read_csv(source_file)
    except Exception as exc:
        return _empty_fantasypros_df(source_file=source_file, error=str(exc))

    if raw_df.empty:
        return _empty_fantasypros_df(source_file=source_file)

    column_map = map_fantasypros_columns(raw_df)
    filename_position = _infer_position_from_filename(source_file)
    rows = []
    for _, row in raw_df.iterrows():
        player_name_col = column_map.get("player_name")
        player_name = row.get(player_name_col) if player_name_col else None
        if not player_name or pd.isna(player_name) or not str(player_name).strip():
            # Real FantasyPros exports include a blank spacer row (e.g.
            # `" ","",""`) between the header and the data -- skip it rather
            # than let it become a phantom empty-name player.
            continue

        rows.append({
            "player_id": row.get(column_map["player_id"])
            if column_map.get("player_id")
            else pd.NA,
            "player_name": str(player_name).strip(),
            "position": _parse_position(row.get(column_map["position"], ""))
            if column_map.get("position")
            else filename_position,
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
            # "adp" is only ever populated from a genuinely-labeled ADP column.
            # There is intentionally no fallback to expert_rank/RK here.
            "adp": row.get(column_map["adp"]) if column_map.get("adp") else pd.NA,
            "adp_rank": row.get(column_map["adp_rank"]) if column_map.get("adp_rank") else pd.NA,
            "expert_rank": row.get(column_map["expert_rank"]) if column_map.get("expert_rank") else pd.NA,
        })

    out = pd.DataFrame(rows, columns=[col for col in FANTASYPROS_OUTPUT_COLUMNS if col != "player_key"])
    out = normalize_fantasypros_players(out)
    out = _coerce_numeric_columns(out)

    if out["projection_rank"].isna().all() and out["projection_points"].notna().any():
        out = out.sort_values("projection_points", ascending=False).reset_index(drop=True)
        out["projection_rank"] = out.index + 1

    out.attrs["source_file"] = str(source_file)
    out.attrs["column_map"] = column_map
    out.attrs["error"] = None

    return out[FANTASYPROS_OUTPUT_COLUMNS]


def _merge_fantasypros_frames(ranking_df, projection_df):
    """
    Merge a rankings-pattern export (tier/position_rank/expert_rank) with a
    projections-pattern export (projection_points) on normalized player key.
    Either side may be empty; whichever side has a value for a given
    canonical column wins, preferring the non-null value.
    """
    if ranking_df.empty and projection_df.empty:
        return _empty_fantasypros_df()
    if ranking_df.empty:
        return projection_df
    if projection_df.empty:
        return ranking_df

    merge_cols = [col for col in FANTASYPROS_OUTPUT_COLUMNS if col != "player_key"]

    def _dedupe_by_key(df):
        # A player can legitimately appear in more than one per-position
        # projection file (e.g. a hybrid FB/TE listed in both the RB and TE
        # exports). Keep the row with the higher projection so
        # .set_index("player_key") stays unique -- a non-unique index makes
        # .loc[key] return a DataFrame instead of a Series below, which
        # corrupts every downstream value for that player.
        out = df.copy()
        out["_proj_sort"] = pd.to_numeric(out["projection_points"], errors="coerce").fillna(-1.0)
        out = out.sort_values("_proj_sort", ascending=False)
        out = out.drop_duplicates("player_key", keep="first")
        return out.drop(columns=["_proj_sort"])

    left = _dedupe_by_key(ranking_df).set_index("player_key")
    right = _dedupe_by_key(projection_df).set_index("player_key")

    combined_index = left.index.union(right.index)
    rows = []
    for key in combined_index:
        left_row = left.loc[key] if key in left.index else None
        right_row = right.loc[key] if key in right.index else None
        merged = {"player_key": key}
        for column in merge_cols:
            left_value = left_row.get(column) if left_row is not None else None
            right_value = right_row.get(column) if right_row is not None else None
            if left_value is not None and not (isinstance(left_value, float) and pd.isna(left_value)) and str(left_value).strip() != "":
                merged[column] = left_value
            else:
                merged[column] = right_value
        rows.append(merged)

    out = pd.DataFrame(rows)
    return out[FANTASYPROS_OUTPUT_COLUMNS]


def load_fantasypros_projections(source_path=None, source_dir=DEFAULT_SOURCE_DIR):
    """
    Load FantasyPros export(s) into a normalized projection dataframe.

    Rankings exports (RK/TIERS/POS, no FPTS) and points-projection exports
    (FPTS, no RK/TIERS) are separate FantasyPros downloads. Both are looked
    for independently and merged by normalized player name -- loading only
    the single most-recently-modified matching file (the previous behavior)
    silently discarded whichever export type wasn't picked.
    """
    if source_path:
        single_file = _find_source_file([], source_path=source_path)
        result = _load_single_fantasypros_file(single_file) if single_file else _empty_fantasypros_df()
        result.attrs["ranking_source_file"] = str(single_file) if single_file else None
        result.attrs["projection_source_file"] = None
        result.attrs["validation"] = validate_fantasypros_projections(result)
        result.attrs["missing_fields"] = _missing_fields(result)
        return result

    ranking_files = _find_all_source_files(FANTASYPROS_RANKING_PATTERNS, source_dir=source_dir)
    projection_files = _find_all_source_files(FANTASYPROS_PROJECTION_PATTERNS, source_dir=source_dir)

    if not ranking_files and not projection_files:
        return _empty_fantasypros_df()

    # FantasyPros' real projections export is one file per position
    # (..._QB.csv, ..._RB.csv, ..._WR.csv, ..._TE.csv) -- concatenate every
    # matching file rather than picking only the newest.
    ranking_df = (
        pd.concat([_load_single_fantasypros_file(f) for f in ranking_files], ignore_index=True)
        if ranking_files
        else _empty_fantasypros_df()
    )
    projection_df = (
        pd.concat([_load_single_fantasypros_file(f) for f in projection_files], ignore_index=True)
        if projection_files
        else _empty_fantasypros_df()
    )

    out = _merge_fantasypros_frames(ranking_df, projection_df)

    if out["projection_rank"].isna().all() and out["projection_points"].notna().any():
        out = out.sort_values("projection_points", ascending=False).reset_index(drop=True)
        out["projection_rank"] = out.index + 1

    out.attrs["ranking_source_file"] = [str(f) for f in ranking_files]
    out.attrs["projection_source_file"] = [str(f) for f in projection_files]
    out.attrs["source_file"] = (
        out.attrs["projection_source_file"] or out.attrs["ranking_source_file"]
    )
    out.attrs["validation"] = validate_fantasypros_projections(out)
    out.attrs["missing_fields"] = _missing_fields(out)
    out.attrs["error"] = None

    return out


def _missing_fields(df):
    if df.empty:
        return [col for col in FANTASYPROS_OUTPUT_COLUMNS if col not in ("player_key",)]
    return [
        column
        for column in FANTASYPROS_OUTPUT_COLUMNS
        if column != "player_key" and (column not in df.columns or df[column].isna().all())
    ]


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
    adp_coverage = (
        round(float(projections_df["adp"].notna().mean()) * 100.0, 2)
        if rows_loaded and "adp" in projections_df.columns
        else 0.0
    )

    return {
        "ranking_source_file": projections_df.attrs.get("ranking_source_file"),
        "projection_source_file": projections_df.attrs.get("projection_source_file"),
        "rows_loaded": rows_loaded,
        "projection_coverage": projection_coverage,
        "tier_coverage": tier_coverage,
        "adp_coverage": adp_coverage,
        "missing_fields": projections_df.attrs.get("missing_fields", []),
        "duplicate_players": validation.get("duplicate_player_names", []),
        "validation": validation,
        "error": projections_df.attrs.get("error"),
        "sample_rows": projections_df.head(10).to_dict("records")
        if rows_loaded
        else [],
    }
