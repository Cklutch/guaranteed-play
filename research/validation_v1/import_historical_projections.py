from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR, PROJECT_ROOT, clean_name, initial_last_key

ACCEPTED_PATHS = [
    VALIDATION_DIR / "historical_projections.csv",
    VALIDATION_DIR / "historical_projection_market.csv",
    PROJECT_ROOT / "data" / "research" / "historical_projections.csv",
]

NORMALIZED_OUTPUT = VALIDATION_DIR / "historical_projections_normalized.csv"
DIAGNOSTICS_OUTPUT = VALIDATION_DIR / "projection_import_diagnostics.json"
COVERAGE_OUTPUT = VALIDATION_DIR / "projection_coverage_by_season.csv"
UNMATCHED_OUTPUT = VALIDATION_DIR / "projection_unmatched_player_examples.csv"
MARKET_NOT_MATCHED_OUTPUT = VALIDATION_DIR / "projection_market_rows_not_matched_examples.csv"

REQUIRED_COLUMNS = ["season", "player_name", "position"]
STAT_COLUMNS = [
    "projected_fantasy_points",
    "projected_receptions",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "projected_carries",
    "projected_rushing_yards",
    "projected_rushing_tds",
    "projected_total_tds",
]
NUMERIC_COLUMNS = STAT_COLUMNS + [
    "projected_targets",
    "projected_positional_rank",
    "projected_games",
]
OPTIONAL_COLUMNS = [
    "projection_source",
    "scoring_format",
    "projected_targets",
    "projected_positional_rank",
    "projected_games",
    "projected_team",
    "projection_date",
    "source_url_or_file",
    "is_preseason_projection",
    "projection_safety_status",
]
BASE_COLUMNS = list(dict.fromkeys(REQUIRED_COLUMNS + OPTIONAL_COLUMNS + NUMERIC_COLUMNS))
OUTPUT_COLUMNS = BASE_COLUMNS + [
    "player_key",
    "initial_last_key",
    "missing_projected_stats_flag",
    "suspicious_projection_flag",
]


def first_existing_path() -> Path | None:
    return next((path for path in ACCEPTED_PATHS if path.exists()), None)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(col).strip().lower(): col for col in df.columns}
    aliases = {
        "season": ["season", "year"],
        "player_name": ["player_name", "player", "name"],
        "position": ["position", "pos"],
        "projected_team": ["projected_team", "team", "tm"],
        "projected_fantasy_points": ["projected_fantasy_points", "fantasy_points", "fpts", "projected_points"],
        "projected_positional_rank": ["projected_positional_rank", "projection_rank", "positional_rank"],
        "projected_targets": ["projected_targets", "targets", "rec_tgt"],
        "projected_carries": ["projected_carries", "carries", "rushing_attempts", "projected_rush_attempts"],
        "projected_receptions": ["projected_receptions", "receptions", "rec"],
        "projected_receiving_yards": ["projected_receiving_yards", "receiving_yards", "rec_yd", "rec_yds"],
        "projected_receiving_tds": ["projected_receiving_tds", "receiving_tds", "rec_td", "rec_tds"],
        "projected_rushing_yards": ["projected_rushing_yards", "rushing_yards", "rush_yd", "rush_yds"],
        "projected_rushing_tds": ["projected_rushing_tds", "rushing_tds", "rush_td", "rush_tds"],
        "projected_total_tds": ["projected_total_tds", "total_tds"],
        "projected_games": ["projected_games", "games", "gp"],
        "projection_source": ["projection_source", "source"],
        "scoring_format": ["scoring_format"],
        "projection_date": ["projection_date", "snapshot_date"],
        "source_url_or_file": ["source_url_or_file", "source_file", "source_url"],
        "is_preseason_projection": ["is_preseason_projection", "preseason_projection"],
        "projection_safety_status": ["projection_safety_status", "safety_status"],
    }
    rename = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            if candidate in lower:
                rename[lower[candidate]] = target
                break
    return df.rename(columns=rename)


def normalize_position(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"WR", "WIDE RECEIVER"}:
        return "WR"
    if text in {"RB", "RUNNING BACK"}:
        return "RB"
    return text


def load_base_keys() -> pd.DataFrame:
    path = VALIDATION_DIR / "predraft_validation_dataset.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "position", "player_key", "initial_last_key"])
    base = pd.read_csv(path, usecols=lambda c: c in {"season", "position", "player_name"})
    base = base[base["position"].isin(["WR", "RB"])].copy()
    base["season"] = pd.to_numeric(base["season"], errors="coerce").astype("Int64")
    base["player_key"] = base["player_name"].apply(clean_name)
    base["initial_last_key"] = base["player_name"].apply(initial_last_key)
    return base[["season", "position", "player_key", "initial_last_key"]].drop_duplicates()


def import_projection_file(path: Path | None) -> tuple[pd.DataFrame, dict[str, object]]:
    if path is None:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty, {"projection_file_used": "not_found", "projection_source": "not_found", "raw_rows": 0, "normalized_rows": 0}

    raw = pd.read_csv(path)
    df = normalize_columns(raw)
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns or df[col].isna().all()]
    stat_cols_present = [col for col in STAT_COLUMNS if col in df.columns and df[col].notna().any()]

    df = df[BASE_COLUMNS].copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["position"] = df["position"].apply(normalize_position)
    df = df[df["position"].isin(["WR", "RB"])].copy()
    df["player_name"] = df["player_name"].astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "projected_total_tds" in df.columns:
        missing_total = df["projected_total_tds"].isna()
        df.loc[missing_total, "projected_total_tds"] = (
            df.loc[missing_total, "projected_receiving_tds"].fillna(0)
            + df.loc[missing_total, "projected_rushing_tds"].fillna(0)
        ).replace(0, np.nan)
    for col in ["projection_source", "scoring_format", "source_url_or_file", "projection_safety_status"]:
        df[col] = df[col].fillna("").astype(str)
    df["projection_source"] = df["projection_source"].replace("", "unknown")
    df["player_key"] = df["player_name"].apply(clean_name)
    df["initial_last_key"] = df["player_name"].apply(initial_last_key)
    df["missing_projected_stats_flag"] = df[STAT_COLUMNS].isna().all(axis=1)

    suspicious = pd.Series(False, index=df.index)
    for col in STAT_COLUMNS:
        suspicious |= df[col].lt(0).fillna(False)
    suspicious |= df["projected_receptions"].gt(180).fillna(False)
    suspicious |= df["projected_receiving_yards"].gt(2500).fillna(False)
    suspicious |= df["projected_carries"].gt(500).fillna(False)
    suspicious |= df["projected_rushing_yards"].gt(2500).fillna(False)
    suspicious |= df["projected_total_tds"].gt(35).fillna(False)
    df["suspicious_projection_flag"] = suspicious

    before = len(df)
    df = df.sort_values(["season", "position", "player_name"]).drop_duplicates(["season", "position", "player_key"], keep="first")
    duplicates_dropped = before - len(df)

    base_keys = load_base_keys()
    exact = df.merge(base_keys[["season", "position", "player_key"]].drop_duplicates(), on=["season", "position", "player_key"], how="left", indicator=True)
    unmatched = exact[exact["_merge"].eq("left_only")].copy()
    unmatched.to_csv(MARKET_NOT_MATCHED_OUTPUT, index=False)
    base_unmatched = base_keys.merge(df[["season", "position", "player_key"]].drop_duplicates(), on=["season", "position", "player_key"], how="left", indicator=True)
    base_unmatched[base_unmatched["_merge"].eq("left_only")].head(200).to_csv(UNMATCHED_OUTPUT, index=False)

    coverage = df.groupby(["season", "position"], dropna=False).size().unstack(fill_value=0).reset_index()
    for pos in ["WR", "RB"]:
        if pos not in coverage.columns:
            coverage[pos] = 0
    coverage = coverage.rename(columns={"WR": "wr_projection_rows", "RB": "rb_projection_rows"})
    coverage.to_csv(COVERAGE_OUTPUT, index=False)

    meta = {
        "projection_file_used": str(path),
        "projection_source": str(df["projection_source"].dropna().iloc[0]) if df["projection_source"].notna().any() else path.name,
        "raw_rows": int(len(raw)),
        "normalized_rows": int(len(df)),
        "missing_required_columns_or_empty": missing_required,
        "projected_stat_columns_present": stat_cols_present,
        "duplicate_rows_dropped": int(duplicates_dropped),
        "missing_projected_stats_rows": int(df["missing_projected_stats_flag"].sum()),
        "suspicious_projection_rows": int(df["suspicious_projection_flag"].sum()),
        "seasons_covered": sorted([int(x) for x in df["season"].dropna().unique().tolist()]),
        "wr_projection_rows": int(df["position"].eq("WR").sum()),
        "rb_projection_rows": int(df["position"].eq("RB").sum()),
        "unmatched_projection_rows": int(len(unmatched)),
    }
    return df[OUTPUT_COLUMNS], meta


def main() -> None:
    path = first_existing_path()
    df, meta = import_projection_file(path)
    df.to_csv(NORMALIZED_OUTPUT, index=False)
    DIAGNOSTICS_OUTPUT.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"projection file used: {meta['projection_file_used']}")
    print(f"projection source: {meta['projection_source']}")
    print(f"seasons covered: {meta.get('seasons_covered', [])}")
    print(f"total projection rows: {len(df)}")
    print(f"WR projection rows: {meta.get('wr_projection_rows', 0)}")
    print(f"RB projection rows: {meta.get('rb_projection_rows', 0)}")
    print(f"unmatched projection rows: {meta.get('unmatched_projection_rows', 0)}")


if __name__ == "__main__":
    main()

