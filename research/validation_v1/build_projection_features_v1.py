from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR, clean_name, initial_last_key
from import_historical_projections import NORMALIZED_OUTPUT, main as import_main

BASE_DATASET_CANDIDATES = [
    VALIDATION_DIR / "predraft_validation_dataset_expanded.csv",
    VALIDATION_DIR / "predraft_validation_dataset.csv",
]
OUTPUT_DATASET = VALIDATION_DIR / "predraft_validation_dataset_projected.csv"
FEATURES_OUTPUT = VALIDATION_DIR / "projection_features_v1.csv"
COVERAGE_OUTPUT = VALIDATION_DIR / "projection_feature_coverage_report.csv"
DIAGNOSTICS_OUTPUT = VALIDATION_DIR / "projection_feature_merge_diagnostics.json"

PROJECTION_COLUMNS = [
    "projection_available_flag",
    "projected_fantasy_points",
    "projected_positional_rank",
    "projection_rank_minus_positional_adp",
    "projection_value_over_adp",
    "projection_points_per_adp",
    "projected_targets",
    "projected_carries",
    "projected_receptions",
    "projected_receiving_yards",
    "projected_receiving_tds",
    "projected_rushing_yards",
    "projected_rushing_tds",
    "projected_total_tds",
    "projected_games",
    "projected_team",
    "projection_source",
    "scoring_format",
    "projection_date",
    "projection_safety_status",
    "projected_volume_score",
    "projected_touch_score",
    "projected_receiving_role_score",
]


def base_dataset_path() -> Path:
    return next(path for path in BASE_DATASET_CANDIDATES if path.exists())


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["position"] = out["position"].astype(str).str.upper().str.strip()
    out["player_key"] = out["player_name"].apply(clean_name)
    out["initial_last_key"] = out["player_name"].apply(initial_last_key)
    return out


def unique_initial_projection_keys(proj: pd.DataFrame) -> pd.DataFrame:
    counts = proj.groupby(["season", "position", "initial_last_key"], dropna=False)["player_key"].nunique().reset_index(name="key_count")
    unique_keys = counts[counts["key_count"].eq(1)][["season", "position", "initial_last_key"]]
    return proj.merge(unique_keys, on=["season", "position", "initial_last_key"], how="inner")


def build_features() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not NORMALIZED_OUTPUT.exists():
        import_main()
    base_path = base_dataset_path()
    base = pd.read_csv(base_path)
    proj = pd.read_csv(NORMALIZED_OUTPUT)
    base = add_match_keys(base)
    proj = add_match_keys(proj)
    numeric = [
        "projected_fantasy_points",
        "projected_positional_rank",
        "projected_targets",
        "projected_carries",
        "projected_receptions",
        "projected_receiving_yards",
        "projected_receiving_tds",
        "projected_rushing_yards",
        "projected_rushing_tds",
        "projected_total_tds",
        "projected_games",
    ]
    proj = safe_numeric(proj, numeric)
    if "projected_total_tds" in proj.columns:
        missing_total = proj["projected_total_tds"].isna()
        proj.loc[missing_total, "projected_total_tds"] = (
            proj.loc[missing_total, "projected_receiving_tds"].fillna(0)
            + proj.loc[missing_total, "projected_rushing_tds"].fillna(0)
        ).replace(0, np.nan)

    keep_cols = ["season", "position", "player_key", "initial_last_key"] + [
        col for col in PROJECTION_COLUMNS if col in proj.columns and col != "projection_available_flag"
    ]
    proj = proj[keep_cols].drop_duplicates(["season", "position", "player_key"], keep="first")
    exact = base.merge(
        proj.drop(columns=["initial_last_key"]),
        on=["season", "position", "player_key"],
        how="left",
        suffixes=("", "_projection"),
        indicator="_projection_exact_merge",
    )
    exact["projection_match_type"] = np.where(exact["_projection_exact_merge"].eq("both"), "exact_name", "")
    for object_col in ["projected_team", "projection_source", "scoring_format", "projection_date", "projection_safety_status"]:
        if object_col in exact.columns:
            exact[object_col] = exact[object_col].astype("object")
    unmatched_mask = exact["projection_match_type"].eq("")
    fallback_proj = unique_initial_projection_keys(proj)
    fallback_cols = [col for col in fallback_proj.columns if col not in {"player_key"}]
    fallback = exact.loc[unmatched_mask, ["season", "position", "initial_last_key"]].merge(
        fallback_proj[fallback_cols],
        on=["season", "position", "initial_last_key"],
        how="left",
        suffixes=("", "_fallback"),
    )
    for col in PROJECTION_COLUMNS:
        if col == "projection_available_flag":
            continue
        fb_col = col if col in fallback.columns else f"{col}_fallback"
        if col in exact.columns and fb_col in fallback.columns:
            exact.loc[unmatched_mask, col] = fallback[fb_col].to_numpy()
    exact.loc[unmatched_mask & fallback.get("projection_source", pd.Series(index=fallback.index)).notna().to_numpy(), "projection_match_type"] = "unique_initial_last"
    exact = exact.drop(columns=["_projection_exact_merge"], errors="ignore")
    for col in PROJECTION_COLUMNS:
        if col not in exact.columns:
            exact[col] = np.nan
    exact["projection_available_flag"] = exact[[c for c in [
        "projected_fantasy_points",
        "projected_receptions",
        "projected_receiving_yards",
        "projected_receiving_tds",
        "projected_carries",
        "projected_rushing_yards",
        "projected_rushing_tds",
        "projected_total_tds",
    ] if c in exact.columns]].notna().any(axis=1).astype(int)

    exact["projection_rank_minus_positional_adp"] = pd.to_numeric(exact["projected_positional_rank"], errors="coerce") - pd.to_numeric(exact.get("positional_adp"), errors="coerce")
    volume_points = (
        exact["projected_receptions"].fillna(0) * 1.0
        + exact["projected_receiving_yards"].fillna(0) * 0.1
        + exact["projected_receiving_tds"].fillna(0) * 6
        + exact["projected_rushing_yards"].fillna(0) * 0.1
        + exact["projected_rushing_tds"].fillna(0) * 6
    )
    exact["projection_value_over_adp"] = volume_points - (200 - pd.to_numeric(exact.get("overall_adp"), errors="coerce")).clip(lower=0).fillna(0) / 2
    exact["projection_points_per_adp"] = np.where(pd.to_numeric(exact.get("overall_adp"), errors="coerce") > 0, volume_points / pd.to_numeric(exact.get("overall_adp"), errors="coerce"), np.nan)
    exact["projected_volume_score"] = (
        exact["projected_receptions"].fillna(0) * 1.0
        + exact["projected_receiving_yards"].fillna(0) * 0.1
        + exact["projected_receiving_tds"].fillna(0) * 6
    )
    exact["projected_touch_score"] = (
        exact["projected_carries"].fillna(0) * 0.35
        + exact["projected_rushing_yards"].fillna(0) * 0.1
        + exact["projected_rushing_tds"].fillna(0) * 6
        + exact["projected_receptions"].fillna(0) * 1.0
        + exact["projected_receiving_yards"].fillna(0) * 0.1
    )
    exact["projected_receiving_role_score"] = exact["projected_receptions"].fillna(0) + exact["projected_receiving_yards"].fillna(0) / 10

    feature_cols = ["season", "player_name", "position", "team", "overall_adp", "positional_adp", "projection_match_type"] + PROJECTION_COLUMNS
    features = exact[[col for col in feature_cols if col in exact.columns]].copy()
    coverage = exact.groupby(["season", "position"], dropna=False).agg(
        rows=("player_name", "size"),
        rows_with_projection=("projection_available_flag", "sum"),
    ).reset_index()
    coverage["projection_coverage_rate"] = coverage["rows_with_projection"] / coverage["rows"]
    coverage.to_csv(COVERAGE_OUTPUT, index=False)

    meta = {
        "base_dataset_used": str(base_path),
        "normalized_projection_file": str(NORMALIZED_OUTPUT),
        "base_rows": int(len(base)),
        "projection_rows": int(len(proj)),
        "matched_projection_rows": int(exact["projection_available_flag"].sum()),
        "unmatched_projection_rows": int(max(0, len(proj) - exact["projection_available_flag"].sum())),
        "wr_validation_dataset_rows_with_projections": int(exact[exact["position"].eq("WR")]["projection_available_flag"].sum()),
        "rb_validation_dataset_rows_with_projections": int(exact[exact["position"].eq("RB")]["projection_available_flag"].sum()),
        "seasons_with_projection_features": sorted([int(x) for x in exact.loc[exact["projection_available_flag"].eq(1), "season"].dropna().unique().tolist()]),
    }
    return exact, features, meta


def main() -> None:
    dataset, features, meta = build_features()
    dataset.to_csv(OUTPUT_DATASET, index=False)
    features.to_csv(FEATURES_OUTPUT, index=False)
    DIAGNOSTICS_OUTPUT.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"matched projection rows: {meta['matched_projection_rows']}")
    print(f"unmatched projection rows: {meta['unmatched_projection_rows']}")
    print(f"WR validation dataset rows with projections: {meta['wr_validation_dataset_rows_with_projections']}")
    print(f"RB validation dataset rows with projections: {meta['rb_validation_dataset_rows_with_projections']}")
    print(f"output dataset: {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()


