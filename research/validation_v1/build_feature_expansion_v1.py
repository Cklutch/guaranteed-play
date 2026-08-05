from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR, clean_name, initial_last_key, ensure_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = VALIDATION_DIR / "predraft_validation_dataset.csv"
EXPANDED_DATASET = VALIDATION_DIR / "predraft_validation_dataset_expanded.csv"
OUTCOME_SOURCE = PROJECT_ROOT / "case_studies" / "data" / "qb_rb_te_wr_elite_age_player_seasons_ppr.csv"
WR_OPPORTUNITY_SOURCE = PROJECT_ROOT / "case_studies" / "data" / "opportunity_wr_target_share_player_seasons_ppr.csv"
ADP_RAW_DIR = VALIDATION_DIR / "source_adp_raw"

ADDED_FEATURES = [
    "projected_fantasy_points",
    "projected_positional_rank",
    "projected_points_over_adp_expectation",
    "projection_minus_adp_implied_expectation",
    "adp_team",
    "adp_source_total_drafts",
    "adp_times_drafted",
    "adp_high",
    "adp_low",
    "adp_stdev",
    "adp_uncertainty_score",
    "team_position_adp_count",
    "same_team_better_adp_count",
    "same_team_top48_count",
    "same_team_top120_count",
    "teammate_best_adp",
    "teammate_second_best_adp",
    "adp_gap_to_next_teammate",
    "target_competition_score",
    "backfield_competition_score",
    "prior_wr_targets_expanded",
    "prior_wr_target_share_expanded",
    "prior_wr_team_targets_expanded",
    "prior_wr_games_expanded",
    "prior_wr_fantasy_ppg_expanded",
    "prior_wr_late_season_target_growth",
    "prior_team_from_opportunity",
    "adp_team_change_from_prior_adp",
    "wr_team_change_from_prior_opportunity",
    "player_first_dataset_season",
    "years_in_league_proxy",
    "rookie_or_first_year_flag",
    "age_bucket_code",
    "prior_games_missed_proxy",
]


def age_bucket(age: object) -> float:
    val = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    if pd.isna(val):
        return np.nan
    if val <= 22:
        return 1
    if val <= 24:
        return 2
    if val <= 26:
        return 3
    if val <= 28:
        return 4
    if val <= 30:
        return 5
    return 6


def load_raw_adp() -> pd.DataFrame:
    files = sorted(ADP_RAW_DIR.glob("ffcalc_ppr_*.csv"))
    files = [p for p in files if "manifest" not in p.name.lower()]
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["_source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = ensure_columns(raw, [
        "season", "player_name", "position", "overall_adp", "adp_source", "source_team",
        "source_total_drafts", "source_times_drafted", "source_high", "source_low", "source_stdev",
    ])
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce").astype("Int64")
    raw["position"] = raw["position"].astype(str).str.upper()
    raw["player_key"] = raw["player_name"].apply(clean_name)
    raw["initial_last_key"] = raw["player_name"].apply(initial_last_key)
    for col in ["overall_adp", "source_total_drafts", "source_times_drafted", "source_high", "source_low", "source_stdev"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.sort_values(["season", "position", "overall_adp"], na_position="last")
    return raw


def attach_raw_adp_features(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if raw.empty:
        return out
    raw_keep = raw[[
        "season", "position", "player_key", "initial_last_key", "source_team", "source_total_drafts",
        "source_times_drafted", "source_high", "source_low", "source_stdev",
    ]].copy()
    raw_keep = raw_keep.drop_duplicates(["season", "position", "player_key"], keep="first")
    exact = raw_keep.drop(columns=["initial_last_key"], errors="ignore")
    out = out.merge(exact, on=["season", "position", "player_key"], how="left")
    missing = out["source_team"].isna()
    fallback = raw_keep[raw_keep["initial_last_key"].astype(str).ne("")].copy()
    unique_counts = fallback.groupby(["season", "position", "initial_last_key"])["player_key"].transform("nunique")
    fallback = fallback[unique_counts.eq(1)].drop_duplicates(["season", "position", "initial_last_key"], keep="first")
    fallback_cols = ["season", "position", "initial_last_key", "source_team", "source_total_drafts", "source_times_drafted", "source_high", "source_low", "source_stdev"]
    matched = out.loc[missing, ["season", "position", "initial_last_key"]].merge(fallback[fallback_cols], on=["season", "position", "initial_last_key"], how="left")
    if not matched.empty:
        for col in ["source_team", "source_total_drafts", "source_times_drafted", "source_high", "source_low", "source_stdev"]:
            out.loc[missing, col] = matched[col].to_numpy()
    rename = {
        "source_team": "adp_team",
        "source_total_drafts": "adp_source_total_drafts",
        "source_times_drafted": "adp_times_drafted",
        "source_high": "adp_high",
        "source_low": "adp_low",
        "source_stdev": "adp_stdev",
    }
    out = out.rename(columns=rename)
    out["adp_uncertainty_score"] = pd.to_numeric(out["adp_stdev"], errors="coerce") / pd.to_numeric(out["overall_adp"], errors="coerce").replace(0, np.nan)
    return out


def attach_competition_features(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "team_position_adp_count": np.nan,
        "same_team_better_adp_count": np.nan,
        "same_team_top48_count": np.nan,
        "same_team_top120_count": np.nan,
        "teammate_best_adp": np.nan,
        "teammate_second_best_adp": np.nan,
        "adp_gap_to_next_teammate": np.nan,
        "target_competition_score": np.nan,
        "backfield_competition_score": np.nan,
    }
    for col, value in defaults.items():
        out[col] = value
    if raw.empty:
        return out
    comp = raw[raw["source_team"].notna() & raw["overall_adp"].notna()].copy()
    comp["source_team"] = comp["source_team"].astype(str)
    grouped = comp.groupby(["season", "position", "source_team"])
    rows = []
    for (season, position, team), group in grouped:
        adps = sorted(pd.to_numeric(group["overall_adp"], errors="coerce").dropna().tolist())
        for _, player in group.iterrows():
            player_adp = float(player["overall_adp"])
            others = sorted([a for a in adps if a != player_adp or adps.count(a) > 1])
            better = [a for a in adps if a < player_adp]
            next_worse = [a for a in adps if a > player_adp]
            rows.append({
                "season": season,
                "position": position,
                "player_key": player["player_key"],
                "initial_last_key": player["initial_last_key"],
                "adp_team": team,
                "team_position_adp_count_src": len(adps),
                "same_team_better_adp_count_src": len(better),
                "same_team_top48_count_src": sum(a <= 48 for a in adps),
                "same_team_top120_count_src": sum(a <= 120 for a in adps),
                "teammate_best_adp_src": min(others) if others else np.nan,
                "teammate_second_best_adp_src": sorted(others)[1] if len(others) > 1 else np.nan,
                "adp_gap_to_next_teammate_src": min(next_worse) - player_adp if next_worse else np.nan,
            })
    features = pd.DataFrame(rows)
    exact = features.drop(columns=["initial_last_key", "adp_team"], errors="ignore").drop_duplicates(["season", "position", "player_key"], keep="first")
    out = out.merge(exact, on=["season", "position", "player_key"], how="left")
    missing = out["team_position_adp_count_src"].isna()
    fallback = features[features["initial_last_key"].astype(str).ne("")].copy()
    unique_counts = fallback.groupby(["season", "position", "initial_last_key"])["player_key"].transform("nunique")
    fallback = fallback[unique_counts.eq(1)].drop_duplicates(["season", "position", "initial_last_key"], keep="first")
    fb_cols = [c for c in fallback.columns if c not in {"player_key", "adp_team"}]
    matched = out.loc[missing, ["season", "position", "initial_last_key"]].merge(fallback[fb_cols], on=["season", "position", "initial_last_key"], how="left")
    if not matched.empty:
        for col in [c for c in features.columns if c.endswith("_src")]:
            out.loc[missing, col] = matched[col].to_numpy()
    src_map = {
        "team_position_adp_count_src": "team_position_adp_count",
        "same_team_better_adp_count_src": "same_team_better_adp_count",
        "same_team_top48_count_src": "same_team_top48_count",
        "same_team_top120_count_src": "same_team_top120_count",
        "teammate_best_adp_src": "teammate_best_adp",
        "teammate_second_best_adp_src": "teammate_second_best_adp",
        "adp_gap_to_next_teammate_src": "adp_gap_to_next_teammate",
    }
    for src, dest in src_map.items():
        out[dest] = out[src].combine_first(out[dest]) if src in out.columns else out[dest]
    out = out.drop(columns=[c for c in out.columns if c.endswith("_src")], errors="ignore")
    has_comp = out["team_position_adp_count"].notna()
    competition = out["same_team_better_adp_count"].fillna(0) + out["same_team_top120_count"].fillna(0) * 0.25
    out.loc[out["position"].eq("WR") & has_comp, "target_competition_score"] = competition[out["position"].eq("WR") & has_comp]
    out.loc[out["position"].eq("RB") & has_comp, "backfield_competition_score"] = competition[out["position"].eq("RB") & has_comp]
    return out


def attach_wr_prior_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded", "prior_wr_games_expanded", "prior_wr_fantasy_ppg_expanded", "prior_team_from_opportunity"]:
        out[col] = np.nan
    if not WR_OPPORTUNITY_SOURCE.exists():
        return out
    opp = pd.read_csv(WR_OPPORTUNITY_SOURCE)
    opp = ensure_columns(opp, ["season", "player_name", "team", "targets", "team_targets", "target_share", "games", "fantasy_ppg"])
    opp["season"] = pd.to_numeric(opp["season"], errors="coerce") + 1
    opp["position"] = "WR"
    opp["player_key"] = opp["player_name"].apply(clean_name)
    opp["initial_last_key"] = opp["player_name"].apply(initial_last_key)
    opp = opp.rename(columns={
        "targets": "prior_wr_targets_expanded",
        "target_share": "prior_wr_target_share_expanded",
        "team_targets": "prior_wr_team_targets_expanded",
        "games": "prior_wr_games_expanded",
        "fantasy_ppg": "prior_wr_fantasy_ppg_expanded",
        "team": "prior_team_from_opportunity",
    })
    keep = ["season", "position", "player_key", "initial_last_key", "prior_team_from_opportunity", "prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded", "prior_wr_games_expanded", "prior_wr_fantasy_ppg_expanded"]
    opp = opp[keep].copy()
    exact = opp.drop(columns=["initial_last_key"], errors="ignore").drop_duplicates(["season", "position", "player_key"], keep="first")
    out = out.merge(exact, on=["season", "position", "player_key"], how="left", suffixes=("", "_new"))
    for col in ["prior_team_from_opportunity", "prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded", "prior_wr_games_expanded", "prior_wr_fantasy_ppg_expanded"]:
        new_col = f"{col}_new"
        if new_col in out.columns:
            out[col] = out[new_col].combine_first(out[col])
            out = out.drop(columns=[new_col])
    missing = out["prior_wr_targets_expanded"].isna() & out["position"].eq("WR")
    fallback = opp[opp["initial_last_key"].astype(str).ne("")].copy()
    unique_counts = fallback.groupby(["season", "position", "initial_last_key"])["player_key"].transform("nunique")
    fallback = fallback[unique_counts.eq(1)].drop_duplicates(["season", "position", "initial_last_key"], keep="first")
    fb_cols = [c for c in fallback.columns if c != "player_key"]
    matched = out.loc[missing, ["season", "position", "initial_last_key"]].merge(fallback[fb_cols], on=["season", "position", "initial_last_key"], how="left")
    if not matched.empty:
        for col in ["prior_team_from_opportunity", "prior_wr_targets_expanded", "prior_wr_target_share_expanded", "prior_wr_team_targets_expanded", "prior_wr_games_expanded", "prior_wr_fantasy_ppg_expanded"]:
            out.loc[missing, col] = matched[col].to_numpy()
    out["prior_wr_late_season_target_growth"] = np.nan
    return out


def attach_player_context(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if OUTCOME_SOURCE.exists():
        hist = pd.read_csv(OUTCOME_SOURCE, usecols=lambda c: c in {"season", "player_name", "position"})
        hist = hist[hist["position"].isin(["WR", "RB"])].copy()
        hist["player_key"] = hist["player_name"].apply(clean_name)
        first = hist.groupby(["position", "player_key"], dropna=False)["season"].min().reset_index().rename(columns={"season": "player_first_dataset_season"})
        out = out.merge(first, on=["position", "player_key"], how="left")
    else:
        out["player_first_dataset_season"] = np.nan
    out["years_in_league_proxy"] = pd.to_numeric(out["season"], errors="coerce") - pd.to_numeric(out["player_first_dataset_season"], errors="coerce") + 1
    out["rookie_or_first_year_flag"] = (out["years_in_league_proxy"] == 1).astype(float)
    out.loc[out["years_in_league_proxy"].isna(), "rookie_or_first_year_flag"] = np.nan
    out["age_bucket_code"] = out["age"].apply(age_bucket)
    out["prior_games_missed_proxy"] = np.where(pd.to_numeric(out.get("prior_games", np.nan), errors="coerce").notna(), 17 - pd.to_numeric(out.get("prior_games", np.nan), errors="coerce"), np.nan)
    out["prior_games_missed_proxy"] = out["prior_games_missed_proxy"].clip(lower=0)

    out["adp_team_change_from_prior_adp"] = np.nan
    if not raw.empty:
        prev = raw[["season", "position", "player_key", "source_team"]].copy()
        prev["season"] = prev["season"] + 1
        prev = prev.rename(columns={"source_team": "prior_adp_team"}).drop_duplicates(["season", "position", "player_key"], keep="first")
        out = out.merge(prev, on=["season", "position", "player_key"], how="left")
        has_both = out["adp_team"].notna() & out["prior_adp_team"].notna()
        out.loc[has_both, "adp_team_change_from_prior_adp"] = (out.loc[has_both, "adp_team"].astype(str) != out.loc[has_both, "prior_adp_team"].astype(str)).astype(float)
    out["wr_team_change_from_prior_opportunity"] = np.nan
    has_wr_team = out["position"].eq("WR") & out["adp_team"].notna() & out["prior_team_from_opportunity"].notna()
    out.loc[has_wr_team, "wr_team_change_from_prior_opportunity"] = (out.loc[has_wr_team, "adp_team"].astype(str) != out.loc[has_wr_team, "prior_team_from_opportunity"].astype(str)).astype(float)
    return out


def attach_projection_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["projected_fantasy_points", "projected_positional_rank", "projected_points_over_adp_expectation", "projection_minus_adp_implied_expectation"]:
        out[col] = np.nan
    return out


def build_inventory(expanded: pd.DataFrame, base_cols: set[str]) -> pd.DataFrame:
    rows = []
    definitions = {
        "projected_fantasy_points": ("preseason projection", "Historical projection file not found", "pre-draft-safe if sourced historically", "unavailable"),
        "projected_positional_rank": ("preseason projection rank", "Historical projection file not found", "pre-draft-safe if sourced historically", "unavailable"),
        "projected_points_over_adp_expectation": ("projection value vs ADP", "Requires historical projections", "pre-draft-safe if sourced historically", "unavailable"),
        "projection_minus_adp_implied_expectation": ("projection minus ADP implied expectation", "Requires historical projections", "pre-draft-safe if sourced historically", "unavailable"),
        "adp_team": ("preseason team from FFCalc ADP", "FFCalc raw ADP archive", "pre-draft-safe", "usable"),
        "adp_source_total_drafts": ("number of drafts in ADP sample", "FFCalc raw ADP archive", "pre-draft-safe", "usable"),
        "adp_times_drafted": ("times player was drafted in ADP sample", "FFCalc raw ADP archive", "pre-draft-safe", "usable"),
        "adp_high": ("highest pick in ADP sample", "FFCalc raw ADP archive", "pre-draft-safe", "usable"),
        "adp_low": ("lowest pick in ADP sample", "FFCalc raw ADP archive", "pre-draft-safe", "usable"),
        "adp_stdev": ("ADP standard deviation", "FFCalc raw ADP archive", "pre-draft-safe", "usable"),
        "adp_uncertainty_score": ("ADP stdev divided by ADP", "derived from FFCalc raw ADP", "pre-draft-safe", "usable"),
        "team_position_adp_count": ("same-team same-position drafted players", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "same_team_better_adp_count": ("same-team same-position players drafted earlier", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "same_team_top48_count": ("same-team same-position players in top 48 overall", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "same_team_top120_count": ("same-team same-position players in top 120 overall", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "teammate_best_adp": ("best same-team same-position teammate ADP", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "teammate_second_best_adp": ("second same-team same-position teammate ADP", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "adp_gap_to_next_teammate": ("ADP gap to next later teammate", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "target_competition_score": ("WR same-team ADP competition", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "backfield_competition_score": ("RB same-team ADP competition", "derived from FFCalc raw ADP team", "pre-draft-safe", "usable"),
        "prior_wr_targets_expanded": ("previous-season WR targets", "WR opportunity study shifted one season", "pre-draft-safe", "usable"),
        "prior_wr_target_share_expanded": ("previous-season WR target share", "WR opportunity study shifted one season", "pre-draft-safe", "usable"),
        "prior_wr_team_targets_expanded": ("previous-season WR team targets", "WR opportunity study shifted one season", "pre-draft-safe", "usable"),
        "prior_wr_games_expanded": ("previous-season WR games", "WR opportunity study shifted one season", "pre-draft-safe", "usable"),
        "prior_wr_fantasy_ppg_expanded": ("previous-season WR fantasy PPG", "WR opportunity study shifted one season", "pre-draft-safe but production-derived", "usable"),
        "prior_wr_late_season_target_growth": ("late-season target growth", "Weekly historical target split not found", "pre-draft-safe if prior-season weekly data", "unavailable"),
        "prior_team_from_opportunity": ("prior-season WR team", "WR opportunity study shifted one season", "pre-draft-safe", "usable"),
        "adp_team_change_from_prior_adp": ("current ADP team differs from previous preseason ADP team", "FFCalc raw ADP archive", "pre-draft-safe proxy", "usable"),
        "wr_team_change_from_prior_opportunity": ("current ADP team differs from prior WR opportunity team", "FFCalc plus prior WR opportunity", "pre-draft-safe", "usable"),
        "player_first_dataset_season": ("first year in local outcome dataset", "local historical outcome cache", "questionable proxy", "questionable"),
        "years_in_league_proxy": ("season minus first local outcome season", "local historical outcome cache", "questionable proxy", "questionable"),
        "rookie_or_first_year_flag": ("first year in local outcome dataset", "local historical outcome cache", "questionable proxy", "questionable"),
        "age_bucket_code": ("bucketed age", "age field", "pre-draft-safe", "usable"),
        "prior_games_missed_proxy": ("17 minus prior games when available", "prior season games", "pre-draft-safe if games available", "poor coverage"),
    }
    total = len(expanded)
    for feature in ADDED_FEATURES:
        meaning, source, safety, status = definitions[feature]
        coverage = int(expanded[feature].notna().sum()) if feature in expanded.columns else 0
        rows.append({
            "feature_name": feature,
            "meaning": meaning,
            "source": source,
            "safety_classification": safety,
            "status": status,
            "coverage_rows": coverage,
            "coverage_pct": round(coverage / max(total, 1) * 100.0, 2),
            "added_in_expansion_v1": feature not in base_cols,
        })
    unavailable_requested = [
        ("projected_targets", "Historical projected target file not found"),
        ("projected_receptions", "Historical projected reception file not found"),
        ("projected_receiving_yards", "Historical projected receiving-yard file not found"),
        ("projected_receiving_tds", "Historical projected receiving-TD file not found"),
        ("projected_carries", "Historical projected carry file not found"),
        ("projected_rushing_yards", "Historical projected rushing-yard file not found"),
        ("projected_total_tds", "Historical projected TD file not found"),
        ("prior_routes", "No local historical route data found"),
        ("prior_route_participation", "No local historical route participation data found"),
        ("prior_air_yards", "No local historical air-yards data found"),
        ("prior_snap_share", "No local historical snap share data found"),
        ("prior_red_zone_usage", "No local historical red-zone usage data found"),
        ("prior_goal_line_usage", "No local historical goal-line usage data found"),
        ("team_implied_points", "No historical preseason betting/team implied point file found"),
        ("projected_team_points", "No historical projected team scoring file found"),
        ("prior_team_rush_attempts", "No local team rush attempts file found"),
        ("prior_team_touchdowns", "No local team TD file found"),
        ("offensive_pace", "No local pace file found"),
        ("qb_change_flag", "No historical QB/team continuity file found"),
        ("coach_change_flag", "No historical coach/coordinator file found"),
        ("draft_capital", "No historical NFL draft capital file found"),
        ("injury_flag_entering_season", "No historical preseason injury file found"),
        ("suspension_flag_entering_season", "No historical preseason suspension file found"),
    ]
    for name, reason in unavailable_requested:
        rows.append({
            "feature_name": name,
            "meaning": name.replace("_", " "),
            "source": reason,
            "safety_classification": "pre-draft-safe if sourced historically; unsafe if current/post-season",
            "status": "unavailable",
            "coverage_rows": 0,
            "coverage_pct": 0.0,
            "added_in_expansion_v1": False,
        })
    return pd.DataFrame(rows)


def coverage_report(expanded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ADDED_FEATURES:
        for position in ["WR", "RB", "ALL"]:
            subset = expanded if position == "ALL" else expanded[expanded["position"].eq(position)]
            coverage = int(subset[col].notna().sum()) if col in subset.columns else 0
            rows.append({
                "feature_name": col,
                "position": position,
                "rows": int(len(subset)),
                "coverage_rows": coverage,
                "coverage_pct": round(coverage / max(len(subset), 1) * 100.0, 2),
            })
    return pd.DataFrame(rows)


def write_report(inventory: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> None:
    usable = inventory[inventory["status"].isin(["usable", "questionable", "poor coverage"])]
    unavailable = inventory[inventory["status"].eq("unavailable")]
    lines = [
        "# Feature Expansion V1 Report",
        "",
        "Date: 2026-07-07",
        "",
        "Scope: research-only feature expansion. No Streamlit app changes, no UI, and no new model families.",
        "",
        "## Summary",
        "",
        "Historical preseason projection files were not found locally. Current 2026 projection/app files exist, but they are not safe for historical validation and were not used as backtest features.",
        "",
        "Expansion V1 therefore adds only pre-draft-safe or prior-season-safe features available from local historical sources: FFCalc preseason ADP raw team/spread fields, same-team ADP competition, prior-season WR opportunity, age buckets, team-change proxies, and years-in-data proxies.",
        "",
        "## Successfully Added Or Populated",
        "",
        usable[["feature_name", "source", "safety_classification", "status", "coverage_rows", "coverage_pct"]].to_markdown(index=False),
        "",
        "## Requested But Unavailable Or Excluded",
        "",
        unavailable[["feature_name", "source", "safety_classification", "status"]].to_markdown(index=False),
        "",
        "## Coverage By Position",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Leakage Review",
        "",
        "- Used: previous-season production/opportunity shifted forward one season.",
        "- Used: preseason ADP raw metadata from Fantasy Football Calculator draft windows before the season.",
        "- Used with caution: first-season and years-in-league proxies from the local outcome cache, marked questionable because the cache begins in 1999 and may not represent true rookie year for older players.",
        "- Excluded: current 2026 projections, final rankings, final finishes as projection inputs, post-season data, and requested features without historical preseason/prior-season sources.",
        "",
        "## Next Step",
        "",
        "Run `evaluate_wr_models.py`, `evaluate_rb_models.py`, and `analyze_draft_window_edges.py` with `predraft_validation_dataset_expanded.csv`, then compare lift over ADP against prior reports.",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def build() -> pd.DataFrame:
    base = pd.read_csv(BASE_DATASET)
    base_cols = set(base.columns)
    raw = load_raw_adp()
    expanded = attach_projection_placeholders(base)
    expanded = attach_raw_adp_features(expanded, raw)
    expanded = attach_competition_features(expanded, raw)
    expanded = attach_wr_prior_opportunity(expanded)
    expanded = attach_player_context(expanded, raw)

    # Backfill existing placeholder columns with the expanded prior-safe values where appropriate.
    for source, dest in [
        ("prior_wr_targets_expanded", "prior_targets"),
        ("prior_wr_target_share_expanded", "prior_target_share"),
        ("prior_wr_team_targets_expanded", "prior_team_pass_attempts"),
        ("prior_wr_games_expanded", "prior_games"),
        ("prior_wr_fantasy_ppg_expanded", "prior_fantasy_ppg"),
    ]:
        if source in expanded.columns and dest in expanded.columns:
            expanded[dest] = expanded[dest].combine_first(expanded[source])
    prior_games = pd.to_numeric(expanded.get("prior_games", np.nan), errors="coerce")
    expanded["prior_games_missed_proxy"] = np.where(prior_games.notna(), 17 - prior_games, np.nan)
    expanded["prior_games_missed_proxy"] = pd.to_numeric(expanded["prior_games_missed_proxy"], errors="coerce").clip(lower=0)

    inventory = build_inventory(expanded, base_cols)
    coverage = coverage_report(expanded)
    expanded.to_csv(EXPANDED_DATASET, index=False)
    inventory.to_csv(VALIDATION_DIR / "feature_expansion_inventory.csv", index=False)
    coverage.to_csv(VALIDATION_DIR / "feature_coverage_report.csv", index=False)
    write_report(inventory, coverage, VALIDATION_DIR / "feature_expansion_v1_report.md")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Preseason Feature Expansion V1 dataset.")
    parser.parse_args()
    expanded = build()
    print(f"Expanded dataset written: {EXPANDED_DATASET}")
    print(f"Rows: {len(expanded)}")
    for col in ADDED_FEATURES:
        if col in expanded.columns:
            print(f"{col}: {int(expanded[col].notna().sum())} rows")


if __name__ == "__main__":
    main()

