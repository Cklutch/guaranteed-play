"""Backtest rookie inputs: 2023-2025 draft classes. Scope is the union of
two real, separately-justified lists:
  1. "High profile players" (draft_pick<=64 OR position_rank<=50) -- the
     original scope, same OR condition as the 2026 class.
  2. The real 72-player backfill-candidate list from
     find_backfill_candidates.py (2024-2025 RB/WR draftees still
     "unconfirmed" or missing entirely from rb_archetypes.csv/
     wr_archetypes.csv) -- added per rookie_data_backfill_plan.pdf's Task A,
     widening real coverage of Day-3 rookies the original cutoff excluded.

Separate from build_rookie_inputs.py (which builds the CURRENT year's
pre-season board) -- this is a validation tool: these players already have
real CONFIRMED current-season NFL archetypes in rb_archetypes.csv/
wr_archetypes.csv, so running their real draft-era inputs through
wr_rookie_projection()/rb_rookie_projection() and comparing the PROJECTED
tier to the real CONFIRMED tier is a genuine backtest (the spec's own
Step 5), not just more coverage.

Real bug fixed while building this list: the first pass joined
raw_draft_picks.csv's pfr_player_name to master_players.csv's player_name
by EXACT string match, which silently dropped every player with a Jr./Sr./
suffix or apostrophe formatting difference (confirmed real misses: Marvin
Harrison Jr., Tyrone Tracy Jr., Tre' Harris -- a real #4 overall pick was
being silently excluded). Fixed by joining on
draftkit.data_pipeline.normalize_player_name() instead (already built in
this repo for exactly this problem, per its own docstring precedent for
draft_capital_player_seasons.csv's Sr./Jr. collisions).

Real data status per input, current as of this build:
  - draft_pick, landing_team, age_at_draft, college: real, all three
    classes, same raw_draft_picks.csv source as the 2026 class.
  - Combine measurables (height/weight/40-time): real, all three classes
    now have at least partial coverage (2023: full RB+WR;
    2024: WR + a separate real RB file; 2025: combined RB+WR file).
  - College dominator ratings: real wherever a player's own real college
    season overlaps a season we have real team-total data for (currently
    2024 and 2025 only -- college_team_total_offense_2024.csv/_2025.csv).
    2023 class: still BLOCKED -- their real college seasons run 2018-2022,
    entirely before either team-totals file. 2025 class: PARTIALLY
    unblocked -- most of these players' real final college season was
    2024, and 7 of them play for schools already in
    college_team_total_offense_2024.csv (Rutgers, USC, Ohio St. x3,
    Mississippi, Texas); the rest (Virginia Tech, Arizona St., Kansas,
    Boise St., Central Florida, North Carolina, Arizona, Missouri,
    Iowa St., TCU) need that file extended before their real dominator
    can be computed. See _real_dominator_from_season_stats() -- always a
    real computed share, never a guess; returns None (left blank) when no
    real season+team-data overlap exists for that specific player.
  - roster_competition_tier: real, per (landing_team, position), via
    build_risk_variables.compute_roster_crowding_tiers() -- real count of
    teammates already carrying a real, meaningfully-involved
    opportunity_share_rate in the season BEFORE this rookie's own real
    draft season. Replaces a flat manual default (3) that contributed zero
    real variance -- see rookie_data_backfill_plan.pdf's follow-up
    validation pass.
  - breakout_age: real wherever _real_breakout_age() can find a real
    college season (within the same 2024/2025 team-data coverage as
    dominator, above) that crosses the real spec threshold (WR 20%, RB
    15%). Left None, not defaulted, everywhere else -- same real, documented
    coverage gap as college dominator, not a separate one.

Usage:
    python -m draftkit.scripts.build_backtest_rookie_inputs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.data_pipeline import normalize_player_name  # noqa: E402
from draftkit.scripts.build_risk_variables import (  # noqa: E402
    OFFENSE_CSV as RISK_OFFENSE_CSV,
    TEAM_CODE_ALIASES,
    compute_roster_crowding_tiers,
    offense_environment_score,
)
from draftkit.scripts.build_rookie_projections import PFR_TEAM_ALIASES  # noqa: E402

DRAFT_PICKS_CSV = REPO_ROOT / "research" / "validation_v1" / "data" / "raw_draft_picks.csv"
MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
BACKFILL_CANDIDATES_CSV = REPO_ROOT / "data" / "processed" / "backfill_candidates.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "raw" / "backtest_rookie_inputs.csv"

BACKTEST_SEASONS = (2023, 2024, 2025)
SKILL_POSITIONS = ("WR", "RB")
PICK_CUTOFF = 64
POSITION_RANK_CUTOFF = 50
DEFAULT_ROSTER_COMPETITION_TIER = 3

# Real 2023 class Combine measurables, user-supplied
# (data/raw/rb_combine_measurables_2023.csv) -- despite the filename, this
# file covers both RB and WR 2023 draft prospects. Height arrives as a
# simple "FT-IN" string (e.g. "5-9"), a different real encoding than the
# 2026 Combine spreadsheet's 4-digit FT-IN-eighths value -- decoded by
# _decode_height_ft_in() below, not the 2026 script's _decode_height().
COMBINE_CSV_BY_SEASON = {
    2023: [REPO_ROOT / "data" / "raw" / "rb_combine_measurables_2023.csv"],
    # 2024 arrived as two separate real files (WR-only, then a later RB one
    # covering the same class) -- both loaded and concatenated.
    2024: [
        REPO_ROOT / "data" / "raw" / "wr_combine_measurables_2024.csv",
        REPO_ROOT / "data" / "raw" / "rb_combine_measurables_2024.csv",
    ],
    2025: [REPO_ROOT / "data" / "raw" / "rb_wr_combine_measurables_2025.csv"],
}

# Real season-by-season college production (rushing for RB, receiving for
# WR), user-supplied. Real dominator = player's real season yards / that
# SAME season's real team yards -- only computable for whichever of a
# player's real college seasons overlaps a season we have real team-total
# data for (college_team_total_offense_2024.csv/_2025.csv). A player's
# season-stats rows span many real years (e.g. 2021-2024); most of those
# years have no real team-total file yet, so most rows can't be used --
# see _real_dominator_from_season_stats() below for the exact matching
# logic (prefers the player's most recent real season with real team data,
# not just any season).
SEASON_STATS_CSV_BY_SEASON = {
    2023: [REPO_ROOT / "data" / "raw" / "rb_season_stats_2023.csv"],
    2024: [REPO_ROOT / "data" / "raw" / "wr_season_stats_2024.csv"],
    2025: [REPO_ROOT / "data" / "raw" / "rb_wr_season_stats_2025.csv"],
}
TEAM_TOTAL_OFFENSE_BY_YEAR = {
    # 2023: real, single-team (LSU only, user-supplied real per-game splits
    # -- season_total = per_game_avg x games, same convention as every
    # other team-totals file). Unblocks Malik Nabers specifically (his real
    # final college season); every other 2023-class player still has no
    # real team-total coverage until their own school's data arrives.
    2023: REPO_ROOT / "data" / "raw" / "college_team_total_offense_2023.csv",
    2024: REPO_ROOT / "data" / "raw" / "college_team_total_offense_2024.csv",
    2025: REPO_ROOT / "data" / "raw" / "college_team_total_offense_2025.csv",
}

# Season-stats files spell school names in full ("Ohio State"); the
# team-totals files abbreviate ("Ohio St.") -- confirmed via a direct diff
# of every real team name in both sources that failed to match. Real bug
# found this way: Judkins/Henderson/Egbuka (all real 2024 Ohio State) were
# silently getting no dominator at all before this map existed.
SCHOOL_NAME_ALIASES = {
    "Arizona State": "Arizona St.", "Boise State": "Boise St.", "Central Florida": "UCF",
    "Florida State": "Florida St.", "Iowa State": "Iowa St.", "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.", "Nevada-Las Vegas": "UNLV", "Ohio State": "Ohio St.",
    "Ole Miss": "Mississippi", "Texas Christian": "TCU", "Washington State": "Washington St.",
}


def _decode_height_ft_in(raw) -> float | None:
    """"5-9" -> 69.0 total inches. Real format confirmed by cross-checking
    a known player (Bijan Robinson, real height ~5'11", raw="5-11" ->
    71.0in -- matches)."""
    if pd.isna(raw) or "-" not in str(raw):
        return None
    feet, inches = str(raw).split("-")
    return float(feet) * 12 + float(inches)


def load_combine_data(season: int) -> pd.DataFrame:
    paths = [p for p in COMBINE_CSV_BY_SEASON.get(season, []) if p.exists()]
    if not paths:
        return pd.DataFrame(columns=["player_name", "height_inches", "weight", "forty_time"])
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["player_name"] = df["Player"].astype(str).str.strip()
        df["height_inches"] = df["Ht"].apply(_decode_height_ft_in)
        df["weight"] = pd.to_numeric(df["Wt"], errors="coerce")
        df["forty_time"] = pd.to_numeric(df["40yd"], errors="coerce")
        frames.append(df[["player_name", "height_inches", "weight", "forty_time"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates("player_name", keep="first")


def load_season_stats(draft_season: int) -> pd.DataFrame:
    paths = [p for p in SEASON_STATS_CSV_BY_SEASON.get(draft_season, []) if p.exists()]
    if not paths:
        return pd.DataFrame(columns=["_key", "Season", "Team", "Pos", "RushYds", "RecYds"])
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        # normalize_player_name-keyed, not raw string -- season-stats names
        # (e.g. "Tre Harris") don't always carry the same apostrophe/suffix
        # formatting as master_players.csv's real name (e.g. "Tre' Harris").
        # Same real bug class as the draft_picks/master_players join fix.
        df["_key"] = df["Player"].apply(normalize_player_name)
        df["Team"] = df["Team"].replace(SCHOOL_NAME_ALIASES)
        frames.append(df[["_key", "Season", "Team", "Pos", "RushYds", "RecYds"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["_key", "Season"])


_TEAM_TOTALS_CACHE: dict[int, pd.DataFrame] = {}


def _team_totals(year: int) -> pd.DataFrame | None:
    if year not in TEAM_TOTAL_OFFENSE_BY_YEAR:
        return None
    if year not in _TEAM_TOTALS_CACHE:
        _TEAM_TOTALS_CACHE[year] = pd.read_csv(TEAM_TOTAL_OFFENSE_BY_YEAR[year])
    return _TEAM_TOTALS_CACHE[year]


def _real_dominator_series(player_key: str, position: str, season_stats: pd.DataFrame) -> list[tuple[int, float]]:
    """Every real (season, dominator) pair computable for this player --
    real season yards / that same season's real team yards -- restricted to
    whichever of the player's real college seasons ALSO has a real team-
    totals file available (currently 2024/2025 only -- see module
    docstring). Sorted ascending by season. Shared by
    _real_dominator_from_season_stats() (wants the most recent) and
    _real_breakout_age() (wants the first one crossing a threshold) so the
    real team-lookup logic exists in exactly one place."""
    rows = season_stats[season_stats["_key"] == player_key]
    if rows.empty:
        return []
    yards_col = "RushYds" if position == "RB" else "RecYds"
    team_yards_col = "rush_yds" if position == "RB" else "pass_yds"
    series = []
    for _, row in rows.sort_values("Season").iterrows():
        team_totals = _team_totals(int(row["Season"]))
        if team_totals is None:
            continue
        team_row = team_totals[team_totals["team"] == row["Team"]]
        if team_row.empty:
            continue
        team_yards = team_row.iloc[0][team_yards_col]
        player_yards = row[yards_col]
        if pd.isna(player_yards) or not team_yards:
            continue
        series.append((int(row["Season"]), round(100 * player_yards / team_yards, 1)))
    return series


def _real_dominator_from_season_stats(player_key: str, position: str, season_stats: pd.DataFrame) -> float | None:
    """Real dominator = player's real season yards / that season's real
    team yards, using whichever of the player's real college seasons is
    both (a) present in the season-stats file and (b) has a real team-
    totals file available -- prefers the most recent such season (closest
    to their actual draft-year production), not just the first match."""
    series = _real_dominator_series(player_key, position, season_stats)
    return series[-1][1] if series else None


# Real spec-given dominator-rating thresholds a player's college production
# must cross to count as a real "breakout" -- from
# rookie_projection_model_v1_spec.pdf, confirmed via
# rookie_data_backfill_plan.pdf's follow-up instructions.
BREAKOUT_DOMINATOR_THRESHOLD = {"WR": 20.0, "RB": 15.0}


def _real_breakout_age(
    player_key: str, position: str, age_at_draft, draft_season: int, season_stats: pd.DataFrame,
) -> float | None:
    """Real age at the FIRST real college season this player's real
    dominator crossed BREAKOUT_DOMINATOR_THRESHOLD -- but only among seasons
    _real_dominator_series() can actually compute (2024/2025 team-data
    coverage only). This is a real, documented UNDERSTATEMENT risk, not a
    silent one: a player whose true breakout came in an earlier real season
    outside that coverage (e.g. a 2022 true freshman breakout for a 2023
    draftee, whose real team-data coverage only starts 2024) will either
    show a later, real-but-not-earliest breakout age, or None if no season
    within coverage ever crosses threshold -- NOT a fabricated early age.
    Age at a given season is approximated as age_at_draft - (draft_season -
    season) -- no real birthdate exists in this repo, only real age at the
    actual NFL draft, so this assumes a birthday-agnostic one-year-per-season
    offset (the same convention fantasy analysis commonly uses for this
    exact metric)."""
    threshold = BREAKOUT_DOMINATOR_THRESHOLD.get(position)
    if threshold is None or pd.isna(age_at_draft):
        return None
    for season, dominator in _real_dominator_series(player_key, position, season_stats):
        if dominator >= threshold:
            return round(float(age_at_draft) - (draft_season - season), 1)
    return None


_HISTORICAL_OFFENSE_ENV_CACHE: dict[int, dict] = {}


def load_historical_offense_environment(season: int) -> dict:
    """Real historical replacement for build_risk_variables.py's
    load_offense_environment(), which only ever filters
    offense_environment_team_seasons.csv to CURRENT_SEASON (2026) -- for the
    backtest, "current team context" is wrong for a player whose real
    rookie season was 2023/2024/2025. Same real source file, same real
    offense_environment_score() formula (imported, not reimplemented, so the
    two can't silently drift), just ranked within the player's own real
    draft season's 32 teams instead of today's.

    schedule_weather_venue_score has NO historical equivalent -- confirmed
    team_schedule_risk.csv (build_schedule_data.py's output) carries no
    season column at all, so it stays on the current-season approximation
    for every backtest player, unfixed this pass (real, flagged gap, not
    silently worked around).

    Returns {team_code (LAR-style, matching raw_draft_picks.csv/
    risk_variables.csv convention): raw 1.0(best)-5.0(worst) risk-direction
    score}.
    """
    if season not in _HISTORICAL_OFFENSE_ENV_CACHE:
        df = pd.read_csv(RISK_OFFENSE_CSV)
        df = df[df["season"] == season].copy()
        if df.empty:
            _HISTORICAL_OFFENSE_ENV_CACHE[season] = {}
        else:
            df["epa_rank"] = df["prior_team_epa_pg"].rank(ascending=False, method="first")
            df["pace_rank"] = df["prior_team_plays_pg"].rank(ascending=False, method="first")
            df["scoring_rank"] = df["prior_team_tds_pg"].rank(ascending=False, method="first")
            df["score"] = df.apply(offense_environment_score, axis=1)
            # offense_environment_team_seasons.csv uses nflverse-style codes
            # (e.g. "LA"); raw_draft_picks.csv/risk_variables.csv use "LAR" --
            # inverse of build_risk_variables.py's own TEAM_CODE_ALIASES
            # (built the other direction, master->nflverse).
            inverse_aliases = {v: k for k, v in TEAM_CODE_ALIASES.items()}
            df["team"] = df["team"].replace(inverse_aliases)
            _HISTORICAL_OFFENSE_ENV_CACHE[season] = df.set_index("team")["score"].to_dict()
    return _HISTORICAL_OFFENSE_ENV_CACHE[season]


def _load_backfill_keys() -> set[str]:
    """Real 72-player backfill-candidate list (find_backfill_candidates.py's
    output) -- unconfirmed/no_row 2024-2025 RB/WR draftees, added to this
    backtest's real scope ON TOP OF the existing 53-player "high profile"
    list (draft_pick<=64 OR position_rank<=50), not replacing it. Most of
    the 53 already clear that threshold and appear in both; this only
    widens 2024/2025 coverage to real Day-3 rookies who never cleared the
    original cutoff (see report_backfill_data_coverage.py for how many of
    them still need real combine/season-stats sourcing)."""
    if not BACKFILL_CANDIDATES_CSV.exists():
        return set()
    df = pd.read_csv(BACKFILL_CANDIDATES_CSV)
    return set(df["player_name"].apply(normalize_player_name))


def build_backtest_rookie_inputs() -> pd.DataFrame:
    draft = pd.read_csv(DRAFT_PICKS_CSV)
    master = pd.read_csv(MASTER_CSV)
    draft["_key"] = draft["pfr_player_name"].apply(normalize_player_name)
    master["_key"] = master["player_name"].apply(normalize_player_name)
    backfill_keys = _load_backfill_keys()

    frames = []
    for season in BACKTEST_SEASONS:
        d = draft[(draft["season"] == season) & (draft["position"].isin(SKILL_POSITIONS))].copy()
        d = d.merge(master[["_key", "player_name", "position_rank"]], on="_key", how="inner")
        d = d[
            (d["pick"] <= PICK_CUTOFF) | (d["position_rank"] <= POSITION_RANK_CUTOFF)
            | (d["_key"].isin(backfill_keys))
        ].copy()
        d = d.rename(columns={"pick": "draft_pick", "team": "landing_team", "age": "age_at_draft"})
        d["draft_season"] = season

        combine = load_combine_data(season)
        combine["_key"] = combine["player_name"].apply(normalize_player_name)
        d = d.merge(combine[["_key", "height_inches", "weight", "forty_time"]], on="_key", how="left")

        season_stats = load_season_stats(season)
        d["_dominator"] = d.apply(
            lambda r: _real_dominator_from_season_stats(r["_key"], r["position"], season_stats), axis=1,
        )
        d["college_dominator_final_year"] = d["_dominator"].where(d["position"] == "WR")
        d["college_dominator_career"] = d["_dominator"].where(d["position"] == "RB")
        d["breakout_age"] = d.apply(
            lambda r: _real_breakout_age(r["_key"], r["position"], r["age_at_draft"], season, season_stats), axis=1,
        )

        hist_oe = load_historical_offense_environment(season)
        std_team = d["landing_team"].replace(PFR_TEAM_ALIASES)
        d["historical_offense_environment_score"] = std_team.map(hist_oe)

        # Real roster crowding heading INTO this rookie's real draft season
        # is the season BEFORE it (e.g. a 2024 draftee's incoming
        # competition is real 2023 usage) -- see
        # build_risk_variables.compute_roster_crowding_tiers()'s docstring.
        crowding_tiers = compute_roster_crowding_tiers(season=season - 1)
        d["roster_competition_tier"] = [
            crowding_tiers.get((team, position), DEFAULT_ROSTER_COMPETITION_TIER)
            for team, position in zip(std_team, d["position"])
        ]

        frames.append(d[[
            "player_name", "position", "draft_season", "draft_pick", "landing_team",
            "age_at_draft", "college", "height_inches", "weight", "forty_time",
            "college_dominator_final_year", "college_dominator_career",
            "historical_offense_environment_score", "roster_competition_tier", "breakout_age",
        ]])

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["player_name", "draft_season"])
    return out.sort_values(["draft_season", "draft_pick"]).reset_index(drop=True)


def main() -> int:
    df = build_backtest_rookie_inputs()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(df)} row(s)")
    for season in BACKTEST_SEASONS:
        sub = df[df["draft_season"] == season]
        print(f"  {season}: {len(sub)} players, "
              f"{sub['weight'].notna().sum()} with real combine data, "
              f"{sub['college_dominator_final_year'].notna().sum() + sub['college_dominator_career'].notna().sum()} with real dominator data, "
              f"{sub['historical_offense_environment_score'].notna().sum()} with real historical offense-environment data, "
              f"{sub['breakout_age'].notna().sum()} with real breakout_age, "
              f"roster_competition_tier real distribution: {sub['roster_competition_tier'].value_counts().sort_index().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
