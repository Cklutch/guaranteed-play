from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR, clean_name, initial_last_key
from import_sportsbook_props import NORMALIZED_OUTPUT, normalize_props, first_existing_path, write_diagnostics

BASE_DATASET = VALIDATION_DIR / "predraft_validation_dataset_expanded.csv"
OUTPUT_DATASET = VALIDATION_DIR / "predraft_validation_dataset_sportsbook.csv"
FEATURE_OUTPUT = VALIDATION_DIR / "sportsbook_features_v1.csv"

SPORTSBOOK_FEATURES = [
    "sportsbook_projected_receptions",
    "sportsbook_projected_receiving_yards",
    "sportsbook_projected_receiving_tds",
    "sportsbook_projected_anytime_td_probability",
    "sportsbook_projected_rushing_yards",
    "sportsbook_projected_rushing_attempts",
    "sportsbook_projected_rushing_tds",
    "sportsbook_wr_volume_score",
    "sportsbook_rb_touch_score",
    "sportsbook_rb_receiving_role_score",
    "sportsbook_team_win_total",
    "sportsbook_team_implied_points",
    "sportsbook_game_total",
    "sportsbook_offensive_environment_score",
    "sportsbook_value_over_adp",
    "sportsbook_rank_minus_positional_adp",
    "sportsbook_projection_available_flag",
    "sportsbook_volume_available_flag",
]


def load_props() -> tuple[pd.DataFrame, dict[str, object]]:
    if NORMALIZED_OUTPUT.exists():
        df = pd.read_csv(NORMALIZED_OUTPUT)
        meta_path = VALIDATION_DIR / "sportsbook_props_import_diagnostics.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return df, meta
    df, meta = normalize_props(first_existing_path())
    df.to_csv(NORMALIZED_OUTPUT, index=False)
    write_diagnostics(df, meta)
    return df, meta


def best_probability(group: pd.DataFrame) -> float:
    for col in ["no_vig_over_probability", "over_implied_probability", "raw_implied_probability"]:
        values = pd.to_numeric(group.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    return np.nan


def aggregate_features(props: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "player_name", "position", "player_key", "initial_last_key"] + SPORTSBOOK_FEATURES
    if props.empty:
        return pd.DataFrame(columns=cols)
    df = props[~props.get("unsupported_market_flag", False).astype(bool)].copy()
    df = df[df["position"].isin(["WR", "RB", "TEAM"])].copy()
    player_rows = []
    for (season, position, player_key), group in df[df["position"].isin(["WR", "RB"])].groupby(["season", "position", "player_key"], dropna=False):
        row = {
            "season": season,
            "position": position,
            "player_key": player_key,
            "initial_last_key": group["initial_last_key"].dropna().iloc[0] if group["initial_last_key"].notna().any() else "",
            "player_name": group["player_name"].dropna().iloc[0] if group["player_name"].notna().any() else "",
        }
        for feature in SPORTSBOOK_FEATURES:
            row[feature] = np.nan
        for market, mgroup in group.groupby("normalized_market", dropna=False):
            line = pd.to_numeric(mgroup["line"], errors="coerce").mean()
            prob = best_probability(mgroup)
            if market in {"receptions", "season_receptions"}:
                row["sportsbook_projected_receptions"] = line
            elif market in {"receiving_yards", "season_receiving_yards"}:
                row["sportsbook_projected_receiving_yards"] = line
            elif market in {"receiving_tds", "season_receiving_tds"}:
                row["sportsbook_projected_receiving_tds"] = line
            elif market == "anytime_td":
                row["sportsbook_projected_anytime_td_probability"] = prob
            elif market == "rushing_yards":
                row["sportsbook_projected_rushing_yards"] = line
            elif market == "rushing_attempts":
                row["sportsbook_projected_rushing_attempts"] = line
            elif market in {"rushing_tds", "season_rushing_tds"}:
                row["sportsbook_projected_rushing_tds"] = line
        rec = row["sportsbook_projected_receptions"]
        rec_yards = row["sportsbook_projected_receiving_yards"]
        rush_yards = row["sportsbook_projected_rushing_yards"]
        rush_attempts = row["sportsbook_projected_rushing_attempts"]
        if position == "WR":
            row["sportsbook_wr_volume_score"] = np.nanmean([rec, rec_yards / 10 if pd.notna(rec_yards) else np.nan])
        if position == "RB":
            row["sportsbook_rb_touch_score"] = np.nanmean([rush_attempts, rush_yards / 6 if pd.notna(rush_yards) else np.nan, rec])
            row["sportsbook_rb_receiving_role_score"] = np.nanmean([rec, row["sportsbook_projected_receiving_yards"] / 8 if pd.notna(row["sportsbook_projected_receiving_yards"]) else np.nan])
        volume_cols = ["sportsbook_projected_receptions", "sportsbook_projected_receiving_yards", "sportsbook_projected_rushing_yards", "sportsbook_projected_rushing_attempts"]
        projection_cols = volume_cols + ["sportsbook_projected_receiving_tds", "sportsbook_projected_rushing_tds", "sportsbook_projected_anytime_td_probability"]
        row["sportsbook_projection_available_flag"] = float(any(pd.notna(row[c]) for c in projection_cols))
        row["sportsbook_volume_available_flag"] = float(any(pd.notna(row[c]) for c in volume_cols))
        player_rows.append(row)
    features = pd.DataFrame(player_rows, columns=cols)
    if features.empty:
        return pd.DataFrame(columns=cols)
    score_col = np.where(features["position"].eq("WR"), features["sportsbook_wr_volume_score"], features["sportsbook_rb_touch_score"])
    features["_sportsbook_score"] = pd.to_numeric(pd.Series(score_col), errors="coerce")
    features["sportsbook_rank"] = features.groupby(["season", "position"])["_sportsbook_score"].rank(ascending=False, method="first")
    return features.drop(columns=["_sportsbook_score"])


def merge_features(dataset: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    for col in SPORTSBOOK_FEATURES + ["sportsbook_rank"]:
        if col not in out.columns:
            out[col] = np.nan
    if features.empty:
        return out
    exact = features.drop(columns=["player_name", "initial_last_key"], errors="ignore")
    out = out.merge(exact, on=["season", "position", "player_key"], how="left", suffixes=("", "_sb"))
    for col in SPORTSBOOK_FEATURES + ["sportsbook_rank"]:
        sb = f"{col}_sb"
        if sb in out.columns:
            out[col] = out[sb].combine_first(out[col])
            out = out.drop(columns=[sb])
    missing = out["sportsbook_projection_available_flag"].isna()
    fallback = features[features["initial_last_key"].astype(str).ne("")].copy()
    if not fallback.empty and missing.any():
        unique = fallback.groupby(["season", "position", "initial_last_key"])["player_key"].transform("nunique")
        fallback = fallback[unique.eq(1)].drop_duplicates(["season", "position", "initial_last_key"], keep="first")
        fb_cols = ["season", "position", "initial_last_key"] + SPORTSBOOK_FEATURES + ["sportsbook_rank"]
        matched = out.loc[missing, ["season", "position", "initial_last_key"]].merge(fallback[fb_cols], on=["season", "position", "initial_last_key"], how="left")
        for col in SPORTSBOOK_FEATURES + ["sportsbook_rank"]:
            out.loc[missing, col] = matched[col].to_numpy()
    out["sportsbook_projection_available_flag"] = out["sportsbook_projection_available_flag"].fillna(0.0)
    out["sportsbook_volume_available_flag"] = out["sportsbook_volume_available_flag"].fillna(0.0)
    out["sportsbook_rank_minus_positional_adp"] = out["sportsbook_rank"] - pd.to_numeric(out.get("positional_adp", np.nan), errors="coerce")
    score = np.where(out["position"].eq("WR"), out["sportsbook_wr_volume_score"], out["sportsbook_rb_touch_score"])
    out["sportsbook_value_over_adp"] = pd.Series(score, index=out.index).rank(pct=True) - (-pd.to_numeric(out.get("overall_adp", np.nan), errors="coerce")).rank(pct=True)
    env = out[["sportsbook_team_win_total", "sportsbook_team_implied_points", "sportsbook_game_total"]].apply(pd.to_numeric, errors="coerce")
    out["sportsbook_offensive_environment_score"] = env.rank(pct=True).mean(axis=1)
    return out


def update_unified_export(dataset: pd.DataFrame) -> None:
    path = VALIDATION_DIR / "unified_player_signal_export.csv"
    if path.exists():
        export = pd.read_csv(path)
    else:
        export = pd.DataFrame(columns=["season", "player_name", "position"])
    keys = ["season", "player_name", "position"]
    feature_cols = [
        "sportsbook_projection_available_flag", "sportsbook_volume_available_flag", "sportsbook_projected_receptions",
        "sportsbook_projected_receiving_yards", "sportsbook_projected_rushing_yards", "sportsbook_projected_rushing_attempts",
        "sportsbook_projected_anytime_td_probability", "sportsbook_value_over_adp",
    ]
    rename = {"sportsbook_projected_anytime_td_probability": "sportsbook_anytime_td_probability"}
    small = dataset[keys + feature_cols].copy().rename(columns=rename).drop_duplicates(keys, keep="first")
    for col in small.columns:
        if col not in keys and col in export.columns:
            export = export.drop(columns=[col])
    export = export.merge(small, on=keys, how="left") if not export.empty else small
    for col in ["target_probability", "ADP_baseline_probability", "edge_over_ADP", "primary_signal", "risk_notes", "feature_explanation"]:
        if col not in export.columns:
            export[col] = np.nan
    has_sb = export.get("sportsbook_projection_available_flag", pd.Series(0, index=export.index)).fillna(0).astype(float).gt(0)
    export.loc[has_sb, "risk_notes"] = "Sportsbook-derived features present for research only; not betting advice and not app-ready without validation lift"
    export.loc[has_sb, "feature_explanation"] = "Sportsbook prop lines converted to fantasy expectation features for historical validation only."
    export.to_csv(path, index=False)


def write_report(dataset: pd.DataFrame, props: pd.DataFrame, meta: dict[str, object]) -> None:
    wr = dataset[dataset["position"].eq("WR")]
    rb = dataset[dataset["position"].eq("RB")]
    wr_cov = int(wr["sportsbook_projection_available_flag"].fillna(0).sum()) if "sportsbook_projection_available_flag" in wr else 0
    rb_cov = int(rb["sportsbook_projection_available_flag"].fillna(0).sum()) if "sportsbook_projection_available_flag" in rb else 0
    markets = sorted(props["normalized_market"].dropna().unique().tolist()) if not props.empty else []
    seasons = sorted(props["season"].dropna().astype(int).unique().tolist()) if not props.empty else []
    lines = [
        "# Sportsbook-Implied Projection V1 Report",
        "",
        "Date: 2026-07-07",
        "",
        "Scope: research-only fantasy football validation. This is not a betting system, does not recommend wagers, does not scrape sportsbooks, and does not connect to sportsbook accounts.",
        "",
        "## Data Status",
        "",
        f"Sportsbook data file used: `{meta.get('sportsbook_data_file_used', 'not_found')}`",
        "",
        f"Sportsbook source: `{meta.get('sportsbook_source', 'not_found')}`",
        "",
        f"Seasons covered: {seasons if seasons else 'none'}",
        "",
        f"Markets covered: {markets if markets else 'none'}",
        "",
        f"WR sportsbook coverage: {wr_cov} / {len(wr)}",
        "",
        f"RB sportsbook coverage: {rb_cov} / {len(rb)}",
        "",
        "## Required CSV Format",
        "",
        "No historical sportsbook prop data was found locally." if props.empty else "Historical sportsbook prop data was found and imported.",
        "",
        "Use `research/validation_v1/historical_sportsbook_props_TEMPLATE.csv` and the format documented in `sportsbook_props_IMPORT_SPEC.md`.",
        "",
        "Minimum columns: `season`, `player_name`, `position`, `market`, `line`, `sportsbook`, `odds`, `odds_format`, `snapshot_date`.",
        "",
        "## Validation Verdict",
        "",
        "Did sportsbook features improve WR lift over ADP? Not tested; no sportsbook prop rows are available." if props.empty else "Did sportsbook features improve WR lift over ADP? See refreshed validation outputs.",
        "",
        "Did sportsbook features improve RB lift over ADP? Not tested; no sportsbook prop rows are available." if props.empty else "Did sportsbook features improve RB lift over ADP? See refreshed validation outputs.",
        "",
        "Did any bucket become Tie-Breaker Only? No." if props.empty else "Did any bucket become Tie-Breaker Only? See bucket report.",
        "",
        "Did any bucket become Strong Draft Signal? No." if props.empty else "Did any bucket become Strong Draft Signal? See bucket report.",
        "",
        "Did anything become App-Ready? No.",
        "",
        "Which sportsbook-derived features helped most? Not answerable without historical props coverage.",
        "",
        "Which markets are most useful? Not answerable without historical props coverage. Season-long volume markets are expected to be more draft-relevant than single-game anytime touchdown markets, but that must be validated before use.",
        "",
        "## Next Research Build",
        "",
        "Import a legal, documented or manually provided historical preseason prop CSV with season-long WR/RB volume markets. Then rerun `import_sportsbook_props.py`, `build_sportsbook_features_v1.py`, WR/RB validators, and draft-window analysis.",
        "",
        "Recommended next Codex prompt:",
        "",
        "```text",
        "Continue research-only validation in research/validation_v1. I have added historical sportsbook prop CSV data. Import it, validate preseason safety and market coverage, build sportsbook-implied fantasy features, rerun WR/RB validators and bucket analysis, and classify signals without making betting recommendations.",
        "```",
    ]
    (VALIDATION_DIR / "sportsbook_implied_projection_v1_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sportsbook-implied fantasy features for research validation only.")
    parser.parse_args()
    props, meta = load_props()
    features = aggregate_features(props)
    features.to_csv(FEATURE_OUTPUT, index=False)
    dataset = pd.read_csv(BASE_DATASET if BASE_DATASET.exists() else VALIDATION_DIR / "predraft_validation_dataset.csv")
    merged = merge_features(dataset, features)
    merged.to_csv(OUTPUT_DATASET, index=False)
    update_unified_export(merged)
    write_report(merged, props, meta)
    wr = merged[merged["position"].eq("WR")]
    rb = merged[merged["position"].eq("RB")]
    print(f"sportsbook data file used: {meta.get('sportsbook_data_file_used', 'not_found')}")
    print(f"sportsbook source: {meta.get('sportsbook_source', 'not_found')}")
    print(f"seasons covered: {sorted(props['season'].dropna().astype(int).unique().tolist()) if not props.empty else []}")
    print(f"markets covered: {sorted(props['normalized_market'].dropna().unique().tolist()) if not props.empty else []}")
    print(f"total rows: {len(merged)}")
    print(f"rows with sportsbook features: {int(merged['sportsbook_projection_available_flag'].fillna(0).sum())}")
    print(f"WR rows with sportsbook features: {int(wr['sportsbook_projection_available_flag'].fillna(0).sum())} / {len(wr)}")
    print(f"RB rows with sportsbook features: {int(rb['sportsbook_projection_available_flag'].fillna(0).sum())} / {len(rb)}")
    print("anything Tie-Breaker Only: No")
    print("anything Strong Draft Signal: No")
    print("anything App-Ready: No")


if __name__ == "__main__":
    main()
