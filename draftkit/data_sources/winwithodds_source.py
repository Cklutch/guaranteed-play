"""Loader for WinWithOdds season-long fantasy projections (winwithodds.com).

WinWithOdds aggregates sportsbook player prop lines across books into full
season-long projected stat lines. The export is wide-format (one row per
player, one column per stat category) rather than the long
[player_name, market_type, line_value, sportsbook] shape used by
draftkit/data_sources/{draftkings,fanduel,pinnacle}_source.py, so it gets its
own loader instead of being forced into that shape.

Their own `Projections` column is NOT used here -- it does not cleanly
reconcile against standard scoring formulas (their site appears to score
exports with a custom first-down-bonus setting this app doesn't use). Instead
this module recomputes fantasy points directly from the raw per-category
stats using draftkit.projection_engine's half-PPR formula, so the result is
directly comparable to master_players.csv's projection_points column.
"""

from pathlib import Path

import pandas as pd

from draftkit.projection_engine import HALF_PPR_SCORING, calculate_fantasy_points_projection


SPORTSBOOK = "WinWithOdds"
DEFAULT_SOURCE_PATH = Path("data/raw/winwithodds_season_projections.csv")
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

OUTPUT_COLUMNS = [
    "player_name",
    "position",
    "sportsbook_half_ppr_points",
    "categories_available",
]

COLUMN_MAP = {
    "Pass Yards": "passing_yards_projection",
    "Pass TDs": "passing_td_projection",
    "Rush Yards": "rushing_yards_projection",
    "Rush TDs": "rushing_td_projection",
    "Rec Yards": "receiving_yards_projection",
    "Rec TDs": "receiving_td_projection",
    "Receptions": "receptions_projection",
}

PROJECTION_COLUMNS = list(COLUMN_MAP.values())


def _empty_df(source_path=None, error=None):
    df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    df.attrs["source_path"] = str(source_path) if source_path else str(DEFAULT_SOURCE_PATH)
    df.attrs["error"] = error
    df.attrs["validation"] = validate_winwithodds_projections(df)
    return df


def validate_winwithodds_projections(projections_df):
    try:
        missing_columns = [column for column in OUTPUT_COLUMNS if column not in projections_df.columns]
        if projections_df.empty:
            return {
                "is_valid": False,
                "row_count": 0,
                "missing_columns": missing_columns,
                "missing_player_count": 0,
                "messages": ["No WinWithOdds projections loaded."],
            }

        missing_player_count = int(projections_df["player_name"].isna().sum())
        messages = []
        if missing_columns:
            messages.append("One or more canonical columns are missing.")
        if missing_player_count:
            messages.append("Some rows are missing player names.")

        return {
            "is_valid": not missing_columns and missing_player_count == 0,
            "row_count": int(len(projections_df)),
            "missing_columns": missing_columns,
            "missing_player_count": missing_player_count,
            "messages": messages,
        }
    except Exception as exc:
        return {
            "is_valid": False,
            "row_count": 0,
            "missing_columns": OUTPUT_COLUMNS,
            "missing_player_count": 0,
            "messages": [f"WinWithOdds validation failed: {exc}"],
        }


def load_winwithodds_projections(source_path=DEFAULT_SOURCE_PATH):
    source_path = Path(source_path)
    if not source_path.exists():
        return _empty_df(source_path=source_path)

    try:
        raw_df = pd.read_csv(source_path)
    except Exception as exc:
        return _empty_df(source_path=source_path, error=str(exc))

    if raw_df.empty or "Name" not in raw_df.columns or "Pos" not in raw_df.columns:
        out = _empty_df(source_path=source_path)
        out.attrs["error"] = "Missing Name or Pos column."
        return out

    skill_df = raw_df[raw_df["Pos"].isin(SKILL_POSITIONS)].copy()
    if skill_df.empty:
        return _empty_df(source_path=source_path)

    projection_inputs = pd.DataFrame(index=skill_df.index)
    for source_column, target_column in COLUMN_MAP.items():
        projection_inputs[target_column] = (
            pd.to_numeric(skill_df[source_column], errors="coerce")
            if source_column in skill_df.columns
            else pd.NA
        )

    out = pd.DataFrame({
        "player_name": skill_df["Name"].astype(str).str.strip(),
        "position": skill_df["Pos"].astype(str).str.strip(),
    })
    out["sportsbook_half_ppr_points"] = projection_inputs.apply(
        lambda row: calculate_fantasy_points_projection(row, scoring=HALF_PPR_SCORING),
        axis=1,
    )
    out["categories_available"] = projection_inputs.notna().sum(axis=1)

    out = out[out["player_name"].ne("")].copy()
    out.attrs["source_path"] = str(source_path)
    out.attrs["error"] = None
    out.attrs["validation"] = validate_winwithodds_projections(out)

    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def get_source_debug_info(source_path=DEFAULT_SOURCE_PATH):
    projections_df = load_winwithodds_projections(source_path=source_path)
    validation = projections_df.attrs.get("validation", validate_winwithodds_projections(projections_df))

    return {
        "sportsbook": SPORTSBOOK,
        "source_path": projections_df.attrs.get("source_path"),
        "row_count": int(len(projections_df)),
        "position_counts": projections_df["position"].value_counts().to_dict()
        if not projections_df.empty
        else {},
        "player_count": int(projections_df["player_name"].nunique()) if not projections_df.empty else 0,
        "validation": validation,
        "error": projections_df.attrs.get("error"),
        "sample_rows": projections_df.head(10).to_dict("records") if not projections_df.empty else [],
    }
