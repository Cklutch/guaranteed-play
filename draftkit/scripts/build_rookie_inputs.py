"""One-time build: data/raw/rookie_inputs.csv (rookie_projection_model_v1_spec.pdf).

Per the spec's own Section 1 instruction, this is a manually-assembled file
(rookie classes are small, ~15-20 relevant skill players/position/year), not
a repeatable pipeline output like the rest of this repo -- this script is a
one-time assembly of real sources, re-run manually once a year after the
draft (or whenever new real data becomes available), not wired into the
regular scoring build.

Real sources, no fabricated values anywhere:
  - player_name, position, draft_pick, landing_team, age_at_draft, college:
    research/validation_v1/data/raw_draft_picks.csv, filtered to the real
    2026 skill-position class already present in data/processed/
    master_players.csv (same 74-player fantasy-relevance filter the RB/WR
    archetype builds already use).
  - height, weight, forty_time: real, joined by name from the user-supplied
    "NFL Combine 2026 @JordanSportGuy Twitter.xlsx". 67/80 real skill
    draftees matched; height/weight populated for all matches, forty_time
    for 41 (not every prospect ran it -- left blank, not guessed, for the
    rest). Height arrives encoded as a 4-digit FT-IN-eighths value (e.g.
    "6052" = 6'5 2/8") -- confirmed against known real player heights
    (Drew Allar ~6'5", Aaron Anderson ~5'8") before trusting the decode.
  - conference_strength_tier: real, from the college field via a static
    Power4/G5/FCS mapping of the actual 39 real schools appearing in this
    draft class (post-2023/2024 realignment: Oregon/USC/UCLA/Washington ->
    Big Ten, Texas/Oklahoma -> SEC, Stanford/Cal/SMU -> ACC, Utah/Arizona
    St./BYU/Colorado/Cincinnati/Houston -> Big 12, all reflected below).
    Notre Dame is football-independent but Power4-equivalent by resources/
    talent level, classified as Power4 here. Any school not in the table
    falls back to "Unknown", not a guessed tier.
  - roster_competition_tier: real, per (landing_team, position), via
    build_risk_variables.compute_roster_crowding_tiers() -- counts real
    2025 (most recent completed season) teammates at the same team/position
    who already carried a real, meaningfully-involved opportunity_share_rate
    (>=0.15), bucketed 1 (0 established teammates, clear path) to 5 (4+,
    most crowded). Replaces the flat manual default (3, "average") every
    rookie previously got -- see rookie_data_backfill_plan.pdf's follow-up
    validation pass, which found that flat default was contributing zero
    real variance to the composite. QB/TE fall back to the flat default
    (3) -- compute_roster_crowding_tiers() only covers RB/WR, matching this
    repo's real archetype taxonomy scope.
  - college_dominator_final_year, college_dominator_career, breakout_age:
    left BLANK for v1, per explicit user direction ("skip for now as I find
    it") -- the college_statistics.csv the user also supplied only covers
    2014-2020 seasons, five to six years too old for this class. See
    draftkit/rookie_projection.py's dominator_score()/breakout_age_score()
    for the documented neutral-default handling of these blanks.

Usage:
    python -m draftkit.scripts.build_rookie_inputs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.scripts.build_risk_variables import (  # noqa: E402
    RECENT_SEASON,
    compute_roster_crowding_tiers,
)
from draftkit.scripts.build_rookie_projections import PFR_TEAM_ALIASES  # noqa: E402

DRAFT_PICKS_CSV = REPO_ROOT / "research" / "validation_v1" / "data" / "raw_draft_picks.csv"
MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
COMBINE_XLSX = Path.home() / "Downloads" / "NFL Combine 2026 @JordanSportGuy Twitter.xlsx"
OUTPUT_CSV = REPO_ROOT / "data" / "raw" / "rookie_inputs.csv"

TEAM_TOTAL_OFFENSE_2025_CSV = REPO_ROOT / "data" / "raw" / "college_team_total_offense_2025.csv"

DRAFT_SEASON = 2026
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_ROSTER_COMPETITION_TIER = 3  # "average" -- see module docstring

# Real 2025 college production, user-supplied (site's own season selector
# confirmed "2025" -- no season-ambiguity risk here, unlike an earlier
# batch that mixed seasons and was NOT used). {player_name: real season
# receiving/rushing yards}. Turned into a real dominator RATING at build
# time by dividing by that player's real team total (see
# _dominator_share() below) -- not a hardcoded percentage, so it stays
# correct if the team-totals file is ever corrected or extended.
#
# Simplification, documented: a true "Dominator Rating" is yards+TDs as a
# share of team offense; team-level offensive TD counts aren't available in
# any real source gathered so far (college_team_total_offense_2025.csv only
# has team season yards, not TDs -- points include kicking, not a clean
# TD proxy). This uses a yards-only share instead, real but a simplified
# substitute for the full TD-weighted version. RB uses this season's real
# rushing yards for college_dominator_career too (the spec wants a true
# career figure; only one real season is available so far).
REAL_WR_RECEIVING_YARDS_2025 = {
    "Skyler Bell": 1278, "Makai Lemon": 1156, "KC Concepcion": 919, "Denzel Boston": 881,
    "Carnell Tate": 875, "Germie Bernard": 862, "De'Zhaun Stribling": 811, "Zachariah Branch": 811,
    "Chris Bell": 917, "Ja'Kobi Lane": 745, "Jordyn Tyson": 711, "Reggie Virgil": 705,
    "CJ Daniels": 557, "Malik Benson": 719, "Lewis Bond": 993, "Kendrick Law": 540,
}
REAL_RB_RUSHING_YARDS_2025 = {
    "Kaytron Allen": 1303, "Emmett Johnson": 1451, "Seth McGowan": 725, "Jonah Coleman": 758,
}

# True multi-year career dominator -- real player rushing yards summed across
# real seasons, divided by the SAME seasons' real team rushing yards (both
# from a user-supplied sports-reference-style career table with real
# season-by-season splits). Only used where real team-season data exists for
# every season being summed (2024 + 2025 -- college_team_total_offense_2024.csv
# doesn't go back to 2023, so a real 2023 season, where available, isn't
# folded in even though the player's own 2023 line was given -- team-side
# 2023 data isn't available yet). Overrides the single-season dict above for
# these two players; more accurate than a single season, per the spec's own
# "career" field name for RB.
REAL_RB_CAREER_RUSHING = {
    # (Notre Dame) 2024: 16 real games, 120att/746yds/7TD; 2025: 12 real
    # games, 113att/674yds/11TD -- real career table, confirmed no missed-
    # time distortion (16 games is a full/extended season, not a shortened
    # one). "college_dominator_career" from the true career total.
    "Jadarian Price": {2024: 746, 2025: 674},
    # (Notre Dame) 2024: 16 real games, 163att/1125yds/17TD; 2025: 12 real
    # games, 199att/1372yds/18TD -- same real games-played as his real
    # backfield mate Jadarian Price above, confirmed no missed-time
    # distortion. Upgrades the single-season 56.2% figure used in an
    # earlier build pass.
    "Jeremiyah Love": {2024: 1125, 2025: 1372},
}
TEAM_TOTAL_OFFENSE_BY_SEASON = {
    2024: REPO_ROOT / "data" / "raw" / "college_team_total_offense_2024.csv",
    2025: REPO_ROOT / "data" / "raw" / "college_team_total_offense_2025.csv",
}


def _dominator_share(player_yards: int, college: str, team_totals: pd.DataFrame, yards_col: str) -> float:
    row = team_totals[team_totals["team"] == college]
    if row.empty:
        return None
    team_yards = row.iloc[0][yards_col]
    return round(100 * player_yards / team_yards, 1)


def _career_rushing_dominator_share(season_yards: dict[int, int], college: str) -> float:
    player_sum, team_sum = 0, 0
    for season, yards in season_yards.items():
        team_totals = pd.read_csv(TEAM_TOTAL_OFFENSE_BY_SEASON[season])
        row = team_totals[team_totals["team"] == college]
        if row.empty:
            continue
        player_sum += yards
        team_sum += row.iloc[0]["rush_yds"]
    return round(100 * player_sum / team_sum, 1) if team_sum else None

# Real conference membership (post-2023/2024 realignment) for the 39 real
# schools appearing in this draft class -- see module docstring for the
# specific realignment moves this reflects.
CONFERENCE_TIER = {
    "Alabama": "Power4", "Arizona St.": "Power4", "Arkansas": "Power4", "BYU": "Power4",
    "Baylor": "Power4", "Boston Col.": "Power4", "Cincinnati": "Power4", "Clemson": "Power4",
    "Georgia": "Power4", "Houston": "Power4", "Indiana": "Power4", "Iowa": "Power4",
    "Kentucky": "Power4", "LSU": "Power4", "Louisville": "Power4", "Miami (FL)": "Power4",
    "Michigan": "Power4", "Mississippi": "Power4", "Mississippi St.": "Power4",
    "Nebraska": "Power4", "North Carolina St.": "Power4", "Notre Dame": "Power4",
    "Ohio St.": "Power4", "Oklahoma": "Power4", "Oregon": "Power4", "Penn St.": "Power4",
    "Rutgers": "Power4", "Stanford": "Power4", "Texas": "Power4", "Texas A&M": "Power4",
    "Texas Tech": "Power4", "USC": "Power4", "Utah": "Power4", "Vanderbilt": "Power4",
    "Wake Forest": "Power4", "Washington": "Power4",
    # G5 / independent (treated as G5 -- no P4 conference membership)
    "East Carolina": "G5", "Georgia St.": "G5", "Connecticut": "G5",
    # FCS
    "North Dakota St.": "FCS",
}


def _decode_height(raw) -> float | None:
    """4-digit FT-IN-eighths -> total inches. "6052" = 6'5 2/8" = 77.25in.
    Confirmed against known real player heights before trusting this (see
    module docstring)."""
    if pd.isna(raw):
        return None
    digits = str(int(raw)).zfill(4)
    feet, inches, eighths = int(digits[0]), int(digits[1:3]), int(digits[3])
    return round(feet * 12 + inches + eighths / 8, 2)


def load_combine_data() -> pd.DataFrame:
    xl = pd.ExcelFile(COMBINE_XLSX)
    frames = []
    for sheet in ("QBs", "RBs", "WRs", "TEs"):
        d = xl.parse(sheet)
        d["player_name"] = d["NAME:"].astype(str).str.strip()
        d["height_inches"] = d["HEIGHT"].apply(_decode_height)
        d["weight"] = pd.to_numeric(d["WEIGHT"], errors="coerce")
        d["forty_time"] = pd.to_numeric(d["40 Yard Dash"], errors="coerce")
        frames.append(d[["player_name", "height_inches", "weight", "forty_time"]])
    combine = pd.concat(frames, ignore_index=True).drop_duplicates("player_name", keep="first")
    return combine


def build_rookie_inputs() -> pd.DataFrame:
    draft = pd.read_csv(DRAFT_PICKS_CSV)
    rookies = draft[
        (draft["season"] == DRAFT_SEASON) & (draft["position"].isin(SKILL_POSITIONS))
    ].copy()

    master = pd.read_csv(MASTER_CSV)
    rookies = rookies[rookies["pfr_player_name"].isin(master["player_name"])].copy()
    rookies = rookies.rename(columns={
        "pfr_player_name": "player_name", "pick": "draft_pick", "team": "landing_team", "age": "age_at_draft",
    })

    combine = load_combine_data()
    rookies = rookies.merge(combine, on="player_name", how="left")

    rookies["conference_strength_tier"] = rookies["college"].map(CONFERENCE_TIER).fillna("Unknown")

    crowding_tiers = compute_roster_crowding_tiers(season=RECENT_SEASON)
    std_team = rookies["landing_team"].replace(PFR_TEAM_ALIASES)
    rookies["roster_competition_tier"] = [
        crowding_tiers.get((team, position), DEFAULT_ROSTER_COMPETITION_TIER)
        for team, position in zip(std_team, rookies["position"])
    ]

    # Real dominator ratings for the players real 2025 production data has
    # been gathered for so far (see REAL_WR_RECEIVING_YARDS_2025/
    # REAL_RB_RUSHING_YARDS_2025 above) -- everyone else stays blank, per
    # explicit user direction ("skip for now as I find it"), not guessed.
    team_totals = pd.read_csv(TEAM_TOTAL_OFFENSE_2025_CSV)
    rookies["college_dominator_final_year"] = rookies.apply(
        lambda r: _dominator_share(REAL_WR_RECEIVING_YARDS_2025[r["player_name"]], r["college"], team_totals, "pass_yds")
        if r["player_name"] in REAL_WR_RECEIVING_YARDS_2025 else None, axis=1,
    )
    rookies["college_dominator_career"] = rookies.apply(
        lambda r: (
            _career_rushing_dominator_share(REAL_RB_CAREER_RUSHING[r["player_name"]], r["college"])
            if r["player_name"] in REAL_RB_CAREER_RUSHING
            else _dominator_share(REAL_RB_RUSHING_YARDS_2025[r["player_name"]], r["college"], team_totals, "rush_yds")
            if r["player_name"] in REAL_RB_RUSHING_YARDS_2025 else None
        ), axis=1,
    )
    rookies["breakout_age"] = pd.NA
    rookies["draft_season"] = DRAFT_SEASON

    out = rookies[[
        "player_name", "position", "draft_season", "draft_pick", "landing_team", "age_at_draft", "college",
        "conference_strength_tier", "height_inches", "weight", "forty_time",
        "roster_competition_tier",
        "college_dominator_final_year", "college_dominator_career", "breakout_age",
    ]].sort_values("draft_pick").reset_index(drop=True)
    return out


def main() -> int:
    df = build_rookie_inputs()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[write] {OUTPUT_CSV}: {len(df)} row(s)")
    print(f"height/weight: {df['weight'].notna().sum()}/{len(df)}")
    print(f"forty_time: {df['forty_time'].notna().sum()}/{len(df)}")
    print(f"conference tier known: {(df['conference_strength_tier'] != 'Unknown').sum()}/{len(df)}")
    print(f"dominator/breakout_age (v1, expected 0): "
          f"{df['college_dominator_final_year'].notna().sum()}/{len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
