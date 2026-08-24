from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from validation_utils import PROJECT_ROOT, VALIDATION_DIR, clean_name, ensure_columns, initial_last_key

OUTCOME_SOURCE_CANDIDATES = [
    PROJECT_ROOT / "case_studies" / "data" / "qb_rb_te_wr_elite_age_player_seasons_ppr.csv",
    PROJECT_ROOT / "case_studies" / "data" / "qb_rb_te_wr_elite_age_player_seasons_half_ppr.csv",
    PROJECT_ROOT / "case_studies" / "case_studies" / "data" / "qb_rb_te_wr_elite_age_player_seasons_half_ppr.csv",
]
WR_OPPORTUNITY_SOURCE = PROJECT_ROOT / "case_studies" / "data" / "opportunity_wr_target_share_player_seasons_half_ppr.csv"
STATS_DIR = VALIDATION_DIR / "data" / "stats_player_reg_by_season"
MARKET_SOURCE_CANDIDATES = [
    VALIDATION_DIR / "historical_adp.csv",
    VALIDATION_DIR / "historical_market.csv",
    PROJECT_ROOT / "data" / "research" / "historical_adp.csv",
]

IDENTITY_COLS = ["season", "player_name", "player_id", "position", "team", "age"]
MARKET_COLS = [
    "preseason_adp", "positional_adp", "overall_adp", "estimated_draft_round",
    "adp_source", "projection_source", "half_ppr_adp", "ppr_adp", "standard_adp",
]
# preseason_projection was dropped (Iteration 2, projection_model_iteration_plan.pdf):
# historical_adp.csv defines the column but it was 0/2,667 populated -- no
# projection data was ever loaded into the market source, for any position or
# season. Not a merge bug; the field was genuinely always empty. Downstream
# consumers (evaluate_wr_models.py / evaluate_rb_models.py) already filter
# FEATURE_GROUPS to `f in df.columns`, so dropping it is a no-op for them --
# it was contributing nothing as an always-null column either.
PRIOR_COLS = [
    "prior_targets", "prior_receptions", "prior_receiving_yards", "prior_receiving_tds", "prior_carries",
    "prior_rushing_yards", "prior_rushing_tds", "prior_passing_yards", "prior_passing_tds", "prior_total_tds",
    "prior_fantasy_points", "prior_fantasy_ppg",
    "prior_games", "prior_target_share", "prior_air_yards", "prior_routes", "prior_snap_share",
    "prior_route_participation", "prior_red_zone_usage", "prior_goal_line_usage", "prior_team_pass_attempts",
    "prior_team_rush_attempts", "prior_team_scoring", "qb_team_continuity", "oc_hc_continuity",
    "vacated_targets", "vacated_touches", "depth_chart_competition",
]
# Positional finish thresholds used for the Top-N outcome labels. WR/RB use the
# original 24/12 lines (deep-starter / true-starter). QB/TE only have ~1
# fantasy-relevant starter per team, so 24/12 would be meaningless -- use
# 12/6 instead (backup-relevant / true-starter for those positions).
POSITION_TOP_N_THRESHOLDS = {
    "WR": (24, 12),
    "RB": (24, 12),
    "QB": (12, 6),
    "TE": (12, 6),
}
OUTCOME_COLS = [
    "final_fantasy_points", "final_fantasy_ppg", "final_positional_finish", "games_played",
    "Top24", "Top12", "Underpriced_Top24", "Underpriced_Top12", "Beat_ADP_By_12",
    "WR_Top24", "WR_Top12", "WR_Underpriced_Top24", "WR_Underpriced_Top12", "WR_Beat_ADP_By_12",
    "RB_Top24", "RB_Top12", "RB_Underpriced_Top24", "RB_Underpriced_Top12", "RB_Beat_ADP_By_12",
    "QB_Top12", "QB_Top6", "QB_Underpriced_Top12", "QB_Underpriced_Top6", "QB_Beat_ADP_By_12",
    "TE_Top12", "TE_Top6", "TE_Underpriced_Top12", "TE_Underpriced_Top6", "TE_Beat_ADP_By_12",
]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_string(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].astype(str).replace({"nan": np.nan, "None": np.nan})


def _load_receptions_by_season_player() -> pd.DataFrame:
    """Real receptions by (season, player_id), read directly from
    stats_player_reg_by_season/*.csv -- used to derive genuine half-PPR
    fantasy_points (Iteration 2, projection_model_iteration_plan.pdf).

    The outcome source's own "half_ppr" candidate
    (case_studies/case_studies/data/qb_rb_te_wr_elite_age_player_seasons_half_ppr.csv)
    was checked directly and found byte-identical to the full-PPR file for
    every real player-season sampled (Jefferson, Kelce, Gibbs, McCaffrey) --
    its own generation script (rb_elite_age_analysis.py's
    SCORING_COLUMN_CANDIDATES) looks for a fantasy_points_half_ppr column
    that does not exist anywhere in stats_player_reg_by_season, and silently
    falls back to fantasy_points_ppr. Deriving directly from real receptions
    counts here sidesteps that broken intermediate file entirely.
    """
    rows = []
    for path in sorted(STATS_DIR.glob("*.csv")):
        cols = pd.read_csv(path, nrows=0).columns
        if "receptions" not in cols or "player_id" not in cols:
            continue
        df = pd.read_csv(path, usecols=["season", "player_id", "receptions"])
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["season", "player_id", "receptions"])
    out = pd.concat(rows, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["receptions"] = pd.to_numeric(out["receptions"], errors="coerce")
    # A handful of (season, player_id) pairs repeat across multi-team/multi-stint
    # rows in the raw source; sum real receptions across stints for a true
    # season total rather than picking one arbitrarily.
    return out.groupby(["season", "player_id"], as_index=False)["receptions"].sum()


def _load_games_by_season_player() -> pd.DataFrame:
    """Real games-played by (season, player_id) from stats_player_reg_by_season.

    Found while building Iteration 3 (baseline projection model): the
    outcome source's own "games" column was requested via _safe_numeric()
    but never existed in any OUTCOME_SOURCE_CANDIDATES file (confirmed --
    those files carry season/player_id/name/position/age/fantasy_points/
    positional_finish/passing/rushing/receiving stat totals, no games
    column), so games_played and everything derived from it
    (final_fantasy_ppg) were silently 100% null the whole time. Same fix
    pattern as _load_receptions_by_season_player() above.
    """
    rows = []
    for path in sorted(STATS_DIR.glob("*.csv")):
        cols = pd.read_csv(path, nrows=0).columns
        if "games" not in cols or "player_id" not in cols:
            continue
        df = pd.read_csv(path, usecols=["season", "player_id", "games"])
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["season", "player_id", "games"])
    out = pd.concat(rows, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["games"] = pd.to_numeric(out["games"], errors="coerce")
    # Sum across multi-team/multi-stint rows for a true season total, same
    # reasoning as the receptions loader above.
    return out.groupby(["season", "player_id"], as_index=False)["games"].sum()


def _load_team_by_season_player() -> pd.DataFrame:
    """Real team by (season, player_id), from stats_player_reg_by_season's
    recent_team column. Found alongside the games_played gap while building
    Iteration 3: the outcome source never carried a team column either
    (out["team"] silently defaulted to all-NaN via the same "column not in
    df.columns" branch), so any team-context feature join was a guaranteed
    100% miss. One row per (season, player_id) in the raw source (no
    mid-season multi-team stint rows), so no aggregation ambiguity here."""
    rows = []
    for path in sorted(STATS_DIR.glob("*.csv")):
        cols = pd.read_csv(path, nrows=0).columns
        if "recent_team" not in cols or "player_id" not in cols:
            continue
        df = pd.read_csv(path, usecols=["season", "player_id", "recent_team"])
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["season", "player_id", "recent_team"])
    out = pd.concat(rows, ignore_index=True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    return out.rename(columns={"recent_team": "real_team"})


def _load_outcomes() -> pd.DataFrame:
    path = _first_existing(OUTCOME_SOURCE_CANDIDATES)
    if path is None:
        raise FileNotFoundError("No historical player-season outcome CSV found in case_studies/data.")
    df = pd.read_csv(path)
    df.attrs["source_path"] = str(path)
    required = ["season", "player_name", "position", "age", "fantasy_points", "positional_finish"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Outcome source missing required columns: {missing}")
    out = df.copy()
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["player_name"] = out["player_name"].astype(str)
    out["player_key"] = out["player_name"].apply(clean_name)
    out["player_id"] = out["player_id"] if "player_id" in out.columns else np.nan
    out["position"] = out["position"].astype(str).str.upper()
    out["age"] = _safe_numeric(out, "age")
    out["final_fantasy_points"] = _safe_numeric(out, "fantasy_points")
    out["final_positional_finish"] = _safe_numeric(out, "positional_finish")

    games = _load_games_by_season_player()
    out = out.merge(games, on=["season", "player_id"], how="left")
    out["games_played"] = out["games"]
    out = out.drop(columns=["games"])

    real_team = _load_team_by_season_player()
    out = out.merge(real_team, on=["season", "player_id"], how="left")
    out["team"] = out["real_team"]
    out = out.drop(columns=["real_team"])

    # Half-PPR correction (Iteration 2): out["final_fantasy_points"] above is
    # sourced from the outcome file's full-PPR "fantasy_points" column. Real
    # league is half-PPR (confirmed this session) -- subtract 0.5/reception
    # using real receptions counts from stats_player_reg_by_season. Rows with
    # no matching (season, player_id) in that source (pre-1999 gaps, ID
    # mismatches) keep the uncorrected full-PPR value rather than being
    # silently dropped or NaN'd -- flagged via half_ppr_corrected for
    # transparency, not hidden.
    receptions = _load_receptions_by_season_player()
    out = out.merge(receptions, on=["season", "player_id"], how="left")
    out["half_ppr_corrected"] = out["receptions"].notna()
    out.loc[out["half_ppr_corrected"], "final_fantasy_points"] = (
        out.loc[out["half_ppr_corrected"], "final_fantasy_points"]
        - 0.5 * out.loc[out["half_ppr_corrected"], "receptions"]
    )
    out = out.drop(columns=["receptions"])

    out["final_fantasy_ppg"] = out["final_fantasy_points"] / out["games_played"].replace(0, np.nan)

    # Duplicate-row fix (Iteration 2): 297 rows share a real (season,
    # player_id, position) key with a different fantasy_points value --
    # the same upstream ID-assignment bug _build_prior_features() below
    # already documents and fixes for the `prior` join frame, but that fix
    # was never applied to `outcomes`/`current` itself, so it leaked
    # straight into final_fantasy_points. NOT the same as the 243 groups
    # that only collide on abbreviated player_name with a genuinely
    # different real player_id -- those are real distinct players and must
    # not be touched here (this key includes player_id, so it never matches
    # them). Keep the highest-production row per key, matching the existing
    # precedent exactly.
    out = out.sort_values("final_fantasy_points", ascending=False, na_position="last")
    out = out.drop_duplicates(["season", "player_id", "position"], keep="first")
    out = out.sort_index()
    return out


def _load_wr_opportunity_prior() -> pd.DataFrame:
    if not WR_OPPORTUNITY_SOURCE.exists():
        return pd.DataFrame()
    df = pd.read_csv(WR_OPPORTUNITY_SOURCE)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    # Joined on player_id (gsis_id), not player_key: this source uses real
    # full names ("Mike Evans") while the base outcome dataset uses
    # PFR-style abbreviated names ("M.Evans") -- clean_name()/player_key
    # never matched between them, which silently left prior_target_share at
    # 0% coverage. Both files share the same gsis_id player_id format, which
    # is a more reliable join than any name-matching anyway.
    cols = ["season", "player_id", "targets", "receiving_tds", "fantasy_points", "fantasy_ppg", "games", "target_share", "team_targets"]
    df = ensure_columns(df, cols)
    df = df[cols].copy()
    return df.rename(columns={
        "targets": "source_targets",
        "receiving_tds": "source_receiving_tds",
        "fantasy_points": "source_fantasy_points",
        "fantasy_ppg": "source_fantasy_ppg",
        "games": "source_games",
        "target_share": "source_target_share",
        "team_targets": "source_team_targets",
    })


def _normalize_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    candidates = {
        "season": ["season", "year"],
        "player_name": ["player_name", "player", "name", "player name"],
        "position": ["position", "pos"],
        "preseason_adp": ["preseason_adp", "adp", "average_draft_position", "average draft position"],
        "overall_adp": ["overall_adp", "adp", "adp_rank", "overall_rank", "rank"],
        "positional_adp": ["positional_adp", "pos_adp", "position_adp", "position_rank", "pos_rank"],
        "projection_source": ["projection_source", "proj_source"],
        "adp_source": ["adp_source", "source"],
        "half_ppr_adp": ["half_ppr_adp", "half_ppr", "half ppr adp"],
        "ppr_adp": ["ppr_adp", "ppr adp"],
        "standard_adp": ["standard_adp", "std_adp", "standard adp"],
    }
    for target, names in candidates.items():
        for name in names:
            if name in lower:
                rename[lower[name]] = target
                break
    return df.rename(columns=rename)


def _load_market() -> pd.DataFrame:
    path = _first_existing(MARKET_SOURCE_CANDIDATES)
    if path is None:
        out = pd.DataFrame()
        out.attrs["source_path"] = "not_found"
        out.attrs["source_name"] = "not_found"
        return out
    raw = pd.read_csv(path)
    df = _normalize_market_columns(raw)
    if "player_name" not in df.columns or "season" not in df.columns:
        out = pd.DataFrame()
        out.attrs["source_path"] = str(path)
        out.attrs["source_name"] = "invalid_missing_required_columns"
        out.attrs["raw_rows"] = int(len(raw))
        return out
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["player_key"] = df["player_name"].apply(clean_name)
    df["initial_last_key"] = df["player_name"].apply(initial_last_key)
    df["position"] = df["position"].astype(str).str.upper() if "position" in df.columns else np.nan
    keep = ["season", "player_key", "initial_last_key", "position"] + MARKET_COLS
    df = ensure_columns(df, keep)[keep]
    for col in ["preseason_adp", "positional_adp", "overall_adp", "half_ppr_adp", "ppr_adp", "standard_adp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["overall_adp"].isna().all() and df["preseason_adp"].notna().any():
        df["overall_adp"] = df["preseason_adp"]
    if df["preseason_adp"].isna().all() and df["overall_adp"].notna().any():
        df["preseason_adp"] = df["overall_adp"]
    df["adp_source"] = df["adp_source"].fillna(path.name)
    before = len(df)
    df = df.dropna(subset=["season", "player_key", "position"])
    df = df.sort_values(["season", "position", "overall_adp", "positional_adp"], na_position="last")
    duplicate_count = int(df.duplicated(["season", "player_key", "position"]).sum())
    df = df.drop_duplicates(["season", "player_key", "position"], keep="first")
    df.attrs["source_path"] = str(path)
    df.attrs["source_name"] = str(df["adp_source"].dropna().iloc[0]) if df["adp_source"].notna().any() else path.name
    df.attrs["raw_rows"] = int(len(raw))
    df.attrs["usable_rows_before_dedupe"] = int(before)
    df.attrs["duplicate_rows_dropped"] = duplicate_count
    return df


def _build_prior_features(outcomes: pd.DataFrame) -> pd.DataFrame:
    base = outcomes.copy()
    base = base[base["position"].isin(["WR", "RB", "QB", "TE"])].copy()
    base["receiving_yards"] = _safe_numeric(base, "receiving_yards")
    base["receiving_tds"] = _safe_numeric(base, "receiving_tds")
    base["rushing_yards"] = _safe_numeric(base, "rushing_yards")
    base["rushing_tds"] = _safe_numeric(base, "rushing_tds")
    base["passing_yards"] = _safe_numeric(base, "passing_yards")
    base["passing_tds"] = _safe_numeric(base, "passing_tds")
    base["prior_source_season"] = base["season"] + 1
    prior = base[[
        "prior_source_season", "player_key", "player_id", "position", "final_fantasy_points", "final_fantasy_ppg", "games_played",
        "receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds", "passing_yards", "passing_tds",
    ]].copy()
    prior = prior.rename(columns={
        "prior_source_season": "season",
        "final_fantasy_points": "prior_fantasy_points",
        "final_fantasy_ppg": "prior_fantasy_ppg",
        "games_played": "prior_games",
        "receiving_yards": "prior_receiving_yards",
        "receiving_tds": "prior_receiving_tds",
        "rushing_yards": "prior_rushing_yards",
        "rushing_tds": "prior_rushing_tds",
        "passing_yards": "prior_passing_yards",
        "passing_tds": "prior_passing_tds",
    })
    # QB fantasy production is passing+rushing TDs; WR/RB/TE is receiving+rushing.
    # total_tds intentionally excludes passing_tds for non-QB rows (passing_tds
    # is 0 for them in the source data already, so summing all three is
    # equivalent and simpler than branching on position here).
    prior["prior_total_tds"] = (
        prior["prior_receiving_tds"].fillna(0)
        + prior["prior_rushing_tds"].fillna(0)
        + prior["prior_passing_tds"].fillna(0)
    )
    wr_prior = _load_wr_opportunity_prior()
    if not wr_prior.empty:
        wr_prior["season"] = wr_prior["season"] + 1
        prior = prior.merge(wr_prior, on=["season", "player_id"], how="left")
        prior["prior_targets"] = prior["source_targets"]
        prior["prior_target_share"] = prior["source_target_share"]
        prior["prior_team_pass_attempts"] = prior["source_team_targets"]
        prior["prior_receiving_tds"] = prior["prior_receiving_tds"].fillna(prior["source_receiving_tds"])
        prior["prior_fantasy_points"] = prior["prior_fantasy_points"].fillna(prior["source_fantasy_points"])
        prior["prior_fantasy_ppg"] = prior["prior_fantasy_ppg"].fillna(prior["source_fantasy_ppg"])
        prior["prior_games"] = prior["prior_games"].fillna(prior["source_games"])
    # player_id was only needed here to join wr_prior above. Drop it before
    # returning: build_dataset()'s `current` frame (built straight from
    # outcomes) already carries the authoritative real player_id, and
    # merging two frames that both have a "player_id" column produces
    # player_id_x/player_id_y -- ensure_columns() then "fixes" the now-
    # missing plain "player_id" name by fabricating a fresh all-NaN one.
    # Hit this exact bug wiring up the fix above; keep player_id single-
    # sourced from `current` to avoid it recurring.
    # Deduplicate on the key the caller merges by. The upstream outcome
    # source has genuinely ambiguous rows -- 588 rows share a (season,
    # player_id) with a *different* player (e.g. 2017 has two distinct RBs
    # both carrying gsis_id 00-0032257, "D.Johnson" and "D.Johnson Jr."),
    # and abbreviated names collapse further on player_key. Left unhandled,
    # the current<->prior merge cross-products and silently invents rows
    # (measured: +403 phantom rows, 15,068 -> 15,471). Keep the
    # highest-production row per key so the survivor is the real starter
    # rather than an arbitrary pick. This masks, but does not fix, the
    # upstream ID-assignment bug -- worth fixing at the source separately.
    prior = prior.sort_values("prior_fantasy_points", ascending=False, na_position="last")
    prior = prior.drop_duplicates(["season", "player_id", "position"], keep="first")
    prior = prior.drop(columns=["player_id"])
    prior = prior.drop_duplicates(["season", "player_key", "position"], keep="first")
    return prior


def _adp_present(df: pd.DataFrame) -> pd.Series:
    return df[["overall_adp", "preseason_adp", "positional_adp"]].notna().any(axis=1)


def _write_merge_diagnostics(dataset: pd.DataFrame, market: pd.DataFrame, metadata: dict[str, object]) -> None:
    coverage = (
        dataset.assign(has_adp=_adp_present(dataset))
        .groupby(["season", "position"], dropna=False)
        .agg(rows=("player_name", "size"), rows_with_adp=("has_adp", "sum"))
        .reset_index()
    )
    coverage["adp_coverage_pct"] = np.where(coverage["rows"] > 0, coverage["rows_with_adp"] / coverage["rows"] * 100.0, np.nan)
    coverage.to_csv(VALIDATION_DIR / "adp_merge_coverage_by_season.csv", index=False)

    unmatched = dataset[~_adp_present(dataset)][["season", "player_name", "position", "team"]].head(100).copy()
    unmatched.to_csv(VALIDATION_DIR / "adp_unmatched_player_examples.csv", index=False)

    market_unmatched = pd.DataFrame()
    if not market.empty:
        dataset_keys = set(zip(dataset["season"].astype(str), dataset["player_key"].astype(str), dataset["position"].astype(str)))
        tmp = market.copy()
        tmp["_key"] = list(zip(tmp["season"].astype(str), tmp["player_key"].astype(str), tmp["position"].astype(str)))
        market_unmatched = tmp[~tmp["_key"].isin(dataset_keys)].head(100).drop(columns=["_key"])
    market_unmatched.to_csv(VALIDATION_DIR / "adp_market_rows_not_matched_examples.csv", index=False)

    with (VALIDATION_DIR / "adp_merge_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


def build_dataset() -> tuple[pd.DataFrame, dict[str, object]]:
    outcomes = _load_outcomes()
    prior = _build_prior_features(outcomes)
    market = _load_market()

    current = outcomes[outcomes["position"].isin(["WR", "RB", "QB", "TE"])].copy()
    keep_current = [
        "season", "player_name", "player_id", "player_key", "position", "team", "age", "final_fantasy_points",
        "final_fantasy_ppg", "final_positional_finish", "games_played", "half_ppr_corrected",
    ]
    current = ensure_columns(current, keep_current)[keep_current]
    dataset = current.merge(prior, on=["season", "player_key", "position"], how="left")
    dataset["initial_last_key"] = dataset["player_name"].apply(initial_last_key)
    if not market.empty:
        exact_market = market.drop(columns=["initial_last_key"], errors="ignore")
        dataset = dataset.merge(exact_market, on=["season", "player_key", "position"], how="left")
        missing_market = dataset["overall_adp"].isna() & dataset["preseason_adp"].isna() & dataset["positional_adp"].isna()
        fallback_market = market.copy()
        fallback_market = fallback_market[fallback_market["initial_last_key"].astype(str).ne("")]
        unique_key_count = fallback_market.groupby(["season", "position", "initial_last_key"])["player_key"].transform("nunique")
        fallback_market = fallback_market[unique_key_count.eq(1)].copy()
        fallback_market = fallback_market.drop_duplicates(["season", "position", "initial_last_key"], keep="first")
        fallback_cols = ["season", "position", "initial_last_key"] + MARKET_COLS
        fallback_market = ensure_columns(fallback_market, fallback_cols)[fallback_cols]
        fallback = dataset.loc[missing_market, ["season", "position", "initial_last_key"]].merge(
            fallback_market, on=["season", "position", "initial_last_key"], how="left"
        )
        if not fallback.empty:
            for col in MARKET_COLS:
                dataset.loc[missing_market, col] = fallback[col].to_numpy()
    else:
        dataset = ensure_columns(dataset, MARKET_COLS)

    for col in ["overall_adp", "preseason_adp", "positional_adp", "half_ppr_adp", "ppr_adp", "standard_adp"]:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    dataset["estimated_draft_round"] = np.ceil(dataset["overall_adp"] / 12.0)
    dataset.loc[dataset["estimated_draft_round"].isna() & dataset["preseason_adp"].notna(), "estimated_draft_round"] = np.ceil(dataset["preseason_adp"] / 12.0)

    # Generic Top24/Top12/Underpriced/Beat_ADP columns kept for backward
    # compatibility with existing WR/RB consumers -- these always use the
    # 24/12 finish lines regardless of position, so they're only meaningful
    # for WR/RB. QB/TE consumers should use the position-specific
    # QB_Top12/QB_Top6/TE_Top12/TE_Top6 columns built below instead.
    dataset["Top24"] = (dataset["final_positional_finish"] <= 24).astype(int)
    dataset["Top12"] = (dataset["final_positional_finish"] <= 12).astype(int)
    dataset["Underpriced_Top24"] = ((dataset["final_positional_finish"] <= 24) & (dataset["positional_adp"] > 24)).astype(float)
    dataset["Underpriced_Top12"] = ((dataset["final_positional_finish"] <= 12) & (dataset["positional_adp"] > 12)).astype(float)
    dataset["Beat_ADP_By_12"] = ((dataset["positional_adp"] - dataset["final_positional_finish"]) >= 12).astype(float)
    dataset.loc[dataset["positional_adp"].isna(), ["Underpriced_Top24", "Underpriced_Top12", "Beat_ADP_By_12"]] = np.nan

    for pos, (primary_n, secondary_n) in POSITION_TOP_N_THRESHOLDS.items():
        mask = dataset["position"].eq(pos)
        finish = dataset["final_positional_finish"]
        adp = dataset["positional_adp"]

        top_primary = (finish <= primary_n).astype(int)
        top_secondary = (finish <= secondary_n).astype(int)
        underpriced_primary = ((finish <= primary_n) & (adp > primary_n)).astype(float)
        underpriced_secondary = ((finish <= secondary_n) & (adp > secondary_n)).astype(float)
        beat_adp = ((adp - finish) >= 12).astype(float)
        no_adp = adp.isna()

        dataset.loc[mask, f"{pos}_Top{primary_n}"] = top_primary[mask]
        dataset.loc[mask, f"{pos}_Top{secondary_n}"] = top_secondary[mask]
        dataset.loc[mask, f"{pos}_Underpriced_Top{primary_n}"] = underpriced_primary.where(~no_adp)[mask]
        dataset.loc[mask, f"{pos}_Underpriced_Top{secondary_n}"] = underpriced_secondary.where(~no_adp)[mask]
        dataset.loc[mask, f"{pos}_Beat_ADP_By_12"] = beat_adp.where(~no_adp)[mask]

    dataset = ensure_columns(dataset, IDENTITY_COLS + MARKET_COLS + PRIOR_COLS + OUTCOME_COLS)
    ordered = IDENTITY_COLS + MARKET_COLS + PRIOR_COLS + OUTCOME_COLS
    extras = [c for c in dataset.columns if c not in ordered and not c.startswith("source_")]
    dataset = dataset[ordered + extras]

    has_adp = _adp_present(dataset)
    wr_mask = dataset["position"].eq("WR")
    rb_mask = dataset["position"].eq("RB")
    qb_mask = dataset["position"].eq("QB")
    te_mask = dataset["position"].eq("TE")
    coverage_by_season = (
        dataset.assign(has_adp=has_adp)
        .groupby("season", dropna=False)["has_adp"]
        .agg(["size", "sum"])
        .rename(columns={"size": "rows", "sum": "rows_with_adp"})
    )
    coverage_by_season["adp_coverage_pct"] = coverage_by_season["rows_with_adp"] / coverage_by_season["rows"] * 100.0

    metadata = {
        "outcome_source": outcomes.attrs.get("source_path"),
        "adp_file_used": market.attrs.get("source_path", "not_found"),
        "adp_source": market.attrs.get("source_name", "not_found"),
        "market_source_found": not market.empty,
        "market_raw_rows": market.attrs.get("raw_rows", 0),
        "market_duplicate_rows_dropped": market.attrs.get("duplicate_rows_dropped", 0),
        "seasons": sorted(dataset["season"].dropna().astype(int).unique().tolist()),
        "total_rows": int(len(dataset)),
        "wr_rows": int(wr_mask.sum()),
        "rb_rows": int(rb_mask.sum()),
        "rows_with_adp": int(has_adp.sum()),
        "wr_rows_with_adp": int((wr_mask & has_adp).sum()),
        "rb_rows_with_adp": int((rb_mask & has_adp).sum()),
        "wr_adp_coverage_pct": round(float((wr_mask & has_adp).sum() / max(wr_mask.sum(), 1) * 100.0), 2),
        "rb_adp_coverage_pct": round(float((rb_mask & has_adp).sum() / max(rb_mask.sum(), 1) * 100.0), 2),
        "qb_rows": int(qb_mask.sum()),
        "te_rows": int(te_mask.sum()),
        "qb_rows_with_adp": int((qb_mask & has_adp).sum()),
        "te_rows_with_adp": int((te_mask & has_adp).sum()),
        "qb_adp_coverage_pct": round(float((qb_mask & has_adp).sum() / max(qb_mask.sum(), 1) * 100.0), 2),
        "te_adp_coverage_pct": round(float((te_mask & has_adp).sum() / max(te_mask.sum(), 1) * 100.0), 2),
        "adp_coverage_by_season": coverage_by_season.reset_index().to_dict("records"),
        "unmatched_player_examples": dataset.loc[~has_adp, ["season", "player_name", "position", "team"]].head(20).to_dict("records"),
    }
    _write_merge_diagnostics(dataset, market, metadata)
    return dataset, metadata


def write_feature_inventory(path: Path) -> None:
    rows = [
        ("age", "WR/RB", "Player age in target season", "nflverse rosters/outcome cache", "objective", "yes", "low", "medium", "trusted"),
        ("preseason_adp", "WR/RB", "Market draft cost before season", "historical_adp.csv or historical_market.csv", "objective", "yes", "low", "high", "investigate"),
        ("overall_adp", "WR/RB", "Overall market draft rank", "historical_adp.csv or historical_market.csv", "objective", "yes", "low", "high", "investigate"),
        ("positional_adp", "WR/RB", "Market rank within position", "historical_adp.csv or historical_market.csv", "objective", "yes", "low", "high", "investigate"),
        ("prior_targets", "WR", "Previous-season WR targets", "prior season WR opportunity cache", "objective", "yes", "low", "medium", "trusted"),
        ("prior_target_share", "WR", "Previous-season share of team targets", "prior season WR opportunity cache", "objective", "yes", "low", "medium", "trusted"),
        ("prior_receiving_yards", "WR/RB", "Previous-season receiving yards", "prior season outcome cache", "objective", "yes", "low", "medium", "trusted"),
        ("prior_receiving_tds", "WR/RB", "Previous-season receiving TDs", "prior season outcome cache", "objective", "yes", "low", "medium", "trusted"),
        ("prior_rushing_yards", "RB", "Previous-season rushing yards", "prior season outcome cache", "objective", "yes", "low", "medium", "trusted"),
        ("prior_rushing_tds", "RB", "Previous-season rushing TDs", "prior season outcome cache", "objective", "yes", "low", "medium", "trusted"),
        ("prior_carries", "RB", "Previous-season carries", "not available in current local cache", "objective", "yes", "low", "high", "investigate"),
        ("final_positional_finish", "WR/RB", "Actual season finish by fantasy points", "same-season outcome label", "objective", "no", "high if used as feature", "low", "label only"),
        ("Top24/Top12 labels", "WR/RB", "Actual top finish labels", "same-season outcome label", "objective", "no", "high if used as feature", "low", "label only"),
        ("Underpriced/Beat_ADP labels", "WR/RB", "Actual finish compared with preseason positional ADP", "same-season label plus preseason ADP", "objective", "label only", "medium", "high", "label only"),
        ("prior_passing_yards", "QB", "Previous-season passing yards", "prior season outcome cache", "objective", "yes", "low", "medium", "trusted"),
        ("prior_passing_tds", "QB", "Previous-season passing TDs", "prior season outcome cache", "objective", "yes", "low", "medium", "trusted"),
        ("QB_Top12/QB_Top6 labels", "QB", "Actual top finish labels using QB-appropriate thresholds (12/6, not 24/12)", "same-season outcome label", "objective", "no", "high if used as feature", "low", "label only"),
        ("TE_Top12/TE_Top6 labels", "TE", "Actual top finish labels using TE-appropriate thresholds (12/6, not 24/12)", "same-season outcome label", "objective", "no", "high if used as feature", "low", "label only"),
    ]
    cols = ["feature_name", "position_model", "what_it_represents", "source_or_calculation", "objective_or_subjective", "pre_draft_available", "leakage_risk", "missing_value_risk", "trust_status"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def write_import_template(path: Path) -> None:
    if path.exists():
        return
    cols = [
        "season", "player_name", "position", "overall_adp", "positional_adp",
        "projection_source", "adp_source", "half_ppr_adp", "ppr_adp", "standard_adp",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pre-draft-safe validation dataset for Guaranteed Play research.")
    parser.add_argument("--output", default=str(VALIDATION_DIR / "predraft_validation_dataset.csv"))
    args = parser.parse_args()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    write_import_template(VALIDATION_DIR / "historical_adp_TEMPLATE.csv")
    dataset, metadata = build_dataset()
    output_path = Path(args.output)
    dataset.to_csv(output_path, index=False)
    write_feature_inventory(VALIDATION_DIR / "feature_inventory.csv")
    pd.DataFrame([{k: v for k, v in metadata.items() if k not in {"adp_coverage_by_season", "unmatched_player_examples"}}]).to_csv(VALIDATION_DIR / "dataset_metadata.csv", index=False)

    print(f"Dataset written: {output_path}")
    if metadata["seasons"]:
        print(f"Seasons used: {len(metadata['seasons'])} ({metadata['seasons'][0]}-{metadata['seasons'][-1]})")
    else:
        print("Seasons used: 0")
    print(f"ADP file used: {metadata['adp_file_used']}")
    print(f"ADP source: {metadata['adp_source']}")
    print(f"Total rows: {metadata['total_rows']}")
    print(f"Rows with ADP: {metadata['rows_with_adp']}")
    print(f"WR rows: {metadata['wr_rows']}")
    print(f"WR rows with ADP: {metadata['wr_rows_with_adp']} ({metadata['wr_adp_coverage_pct']}%)")
    print(f"RB rows: {metadata['rb_rows']}")
    print(f"RB rows with ADP: {metadata['rb_rows_with_adp']} ({metadata['rb_adp_coverage_pct']}%)")
    print(f"QB rows: {metadata['qb_rows']}")
    print(f"QB rows with ADP: {metadata['qb_rows_with_adp']} ({metadata['qb_adp_coverage_pct']}%)")
    print(f"TE rows: {metadata['te_rows']}")
    print(f"TE rows with ADP: {metadata['te_rows_with_adp']} ({metadata['te_adp_coverage_pct']}%)")
    print("ADP coverage by season written: research/validation_v1/adp_merge_coverage_by_season.csv")
    print("Unmatched player examples written: research/validation_v1/adp_unmatched_player_examples.csv")
    if metadata["rows_with_adp"] == 0:
        print("Historical ADP/projection source not found or did not match. Add research/validation_v1/historical_adp.csv using historical_adp_TEMPLATE.csv format.")


if __name__ == "__main__":
    main()


