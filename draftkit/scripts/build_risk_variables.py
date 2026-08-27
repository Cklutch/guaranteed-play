"""One-time build (v2 base, injury/role/schedule now v5): join historical
NFL + schedule data into per-player risk scores across the four v2
categories, per the user's master spec.

v2 replaces v1's "many small quintile-bucketed sub-variables averaged per
category" approach with the spec's own explicit formulas: exactly one
directly-computed 1.0-5.0 score per category (injury now via
draftkit.injury_history.injury_risk_score_v4(), role_usage_td_score(),
offense_environment_score(), schedule_weather_venue_score()), adapted to
real available inputs.

v5 (risk_engine_v5_master_fix_spec.pdf) reworks Role/Usage/TD and
Schedule/Weather on top of v2's structure:

  - Role/Usage/TD used to source "target_share" from
    prior_route_participation_rate (route participation, which legitimately
    reaches 80-90%+) under a field literally named target_share -- but real
    target share tops out ~25-30% even for elite players, which is why
    (1-target_share)*2 never got close to zero for anyone. WORSE: even
    nflverse's own real target_share column in stats_player_reg_by_season
    has a season-total dilution bug for anyone who missed games -- confirmed
    directly, McCaffrey's 2024 row shows games=4, targets=19,
    target_share=0.037 (3.7%), an absurd number for an every-down receiving
    back, because it divides by the team's FULL 17-game season total, not
    just the games he was active for. Fixed here by recomputing
    target_share_rate manually as targets / (team_pass_attempts_per_game *
    games_played), then ranking it by PERCENTILE within the current pool's
    own position group (see build_position_percentiles) instead of judging
    it against an unreachable 100% ceiling. snap_share could NOT get the
    same fix -- snap_share_player_seasons.csv only carries a pre-aggregated
    rate, no raw snap counts and no games-played field alongside it, so if
    it has the same dilution bug there's no way to detect or correct it
    with what this repo has. Shipped as-is, a documented, unfixable-with-
    current-data limitation.

  - The spec's "weekly-refreshed, in-season" framing for depth-chart
    competition and small-sample confidence (Sections 4c/4d) has no real
    data behind it in this app -- this is a preseason draft-prep tool, and
    every "current season" field in this pipeline is actually the most
    recent COMPLETED season's real numbers repackaged under next year's
    label (confirmed by how the spec's own Nacua example works out: 28.8%
    vs 23.5% are real 2025 numbers, not live 2026 in-season data). Adapted
    here to use the most-recent-completed-season's real numbers throughout
    -- same substitution the spec's own worked example already relies on,
    just made explicit and systematic.

  - Schedule/Weather: home_climate (a team's own stadium type, a constant)
    and road_cold_games (only true away trips into cold outdoor stadiums)
    replace the old dome_games/outdoor_cold_games, which incorrectly
    counted a team's OWN home dome games as risk-reducing.

Injury v4 (season-range fix): the old Out-report-based counting undercounts
or entirely misses IR stints (players drop off the weekly injury report
once on IR). v4 instead builds games_missed from an explicitly constructed
season range (rookie year through last season) via real games-played data,
so a season with zero rows counts as fully missed instead of vanishing --
see draftkit/injury_history.py's own docstring for the confirmed Erick
All/McCaffrey cases this fixes. rookie_season is resolved by player_id, NOT
normalized name -- draft_capital_player_seasons.csv spans multiple NFL
generations and contains real Sr./Jr. pairs (Marvin Harrison 1996 vs.
Marvin Harrison Jr. 2024; Michael Pittman 1998 vs. Michael Pittman Jr.
2020) that normalize_player_name collapses to the same key (it strips
jr/sr/ii/iii suffixes by design, to match format variants of the SAME
person elsewhere). Confirmed by direct inspection: a normalized-name join
here silently attaches the Sr.'s decades-old draft_season to the Jr. who is
actually in the pool, producing an absurd 20+ year "fully missed" streak.
Joining by player_id sidesteps this entirely.

Every input gap was checked against real data before being filled with a
proxy -- see the plan's Context section for the full audit. Summary:
  - qb_tier / oline_pass_block_rank / oc_continuity: do not exist anywhere
    in this repo (no PFF/DVOA access, no O-line dataset, no coaching-change
    tracking). User's explicit decision: substitute prior_team_epa_pg
    (the standard single team-offense-quality metric) instead of
    fabricating these.
  - injury_severity: no true severity classification exists (injury
    reports only carry body part + games missed). Proxied from
    games-missed magnitude, documented as a proxy, not real severity data.
  - Season-ending IR stints are UNDER-COUNTED by the weekly injury report
    data itself (players drop off it once on IR, confirmed by tracing a
    real case). This is a data gap the formula cannot fix -- shipped
    anyway per the user's explicit decision, not silently hidden.
  - 2026 schedule (dome/outdoor, bye week, SOS, primetime): did not exist
    in this repo; built fresh by draftkit/scripts/build_schedule_data.py
    (verified against the real nflverse schedule release).
  - career_touches is computed (summed across all
    stats_player_reg_by_season/*.csv) and stored, but -- matching the
    spec's OWN function body, which lists it as an input in the docstring
    but never references it in the actual formula -- it is not fed into
    injury_risk_score. Likewise red_zone_share is computed/stored but not
    used by role_usage_td_score's actual formula, for the same reason.

market (value_signal_market) and volatility (volatility_diagnostic) are
computed but deliberately EXCLUDED from risk_index -- market is a value
signal, not a risk signal (v1 diluted the composite with it); volatility
is a downstream symptom of role instability, not an independent driver.

Usage:
    python -m draftkit.scripts.build_risk_variables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.injury_history import (  # noqa: E402
    FULL_SEASON_GAMES,
    NEUTRAL_SCORE,
    assemble_injury_inputs,
    injury_risk_score_v4_detailed,
)
from draftkit.draft_analysis import PROJECTION_MANUAL_ADJUSTMENTS  # noqa: E402
from draftkit.projection_enrichment import normalize_player_name  # noqa: E402
from draftkit.risk_scoring import (  # noqa: E402
    RiskCategory,
    load_weights,
    player_from_variable_row,
    risk_index,
)

DATA_DIR = REPO_ROOT / "research" / "validation_v1" / "data"
MASTER_CSV = REPO_ROOT / "data" / "processed" / "master_players.csv"
SPORTSBOOK_CSV = REPO_ROOT / "data" / "processed" / "sportsbook_vs_adp_comparison.csv"
OFFENSE_CSV = DATA_DIR / "offense_environment_team_seasons.csv"
SCHEDULE_CSV = REPO_ROOT / "data" / "processed" / "team_schedule_risk.csv"
DRAFT_CAPITAL_CSV = DATA_DIR / "draft_capital_player_seasons.csv"
STATS_DIR = DATA_DIR / "stats_player_reg_by_season"
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "risk_variables.csv"

CURRENT_SEASON = 2026
RECENT_SEASON = 2025          # most recent completed season
SEASON_BEFORE = 2024
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
DEPTH_CHART_SNAP_THRESHOLD = 0.15  # "meaningfully involved" cutoff for competition counting

# Materiality cutoff for load_target_share_rate()'s disclosed
# target_share_data_quality_flag -- the team_games/games rescaling only
# moves a player's target_share_rate when they missed real games (a
# uniform-rate-across-missed-games assumption, not a measured value), so
# only surface the flag once that correction is large enough to matter.
# 2 percentage points comfortably catches the real confirmed cases (e.g.
# Tee Higgins 16.1%->18.3%, Cedric Tillman 7.5%->10.6%) without flagging
# every player who missed a single meaningless late-season snap.
TARGET_SHARE_FLAG_THRESHOLD = 0.02

# master_players.csv uses "LAR" for the Rams; the nflverse-derived team
# files (offense_environment_team_seasons.csv, team_schedule_risk.csv) use
# "LA". Confirmed by diffing both team-code sets directly -- this is the
# only mismatch (aside from "FA" / free agents, which correctly has no
# team-level data to join).
TEAM_CODE_ALIASES = {"LAR": "LA"}


# ---------------------------------------------------------------------------
# Name/id crosswalks (unchanged from v1 -- same real join-key gaps exist)
# ---------------------------------------------------------------------------

def _build_id_name_crosswalk() -> pd.DataFrame:
    """player_id -> player_name/position, from files that carry both."""
    parts = []
    for fname in ("injury_durability_player_seasons.csv", "snap_share_player_seasons.csv"):
        df = pd.read_csv(DATA_DIR / fname, usecols=lambda c: c in {
            "player_id", "player_name", "position"
        })
        parts.append(df[["player_id", "player_name", "position"]])
    stats = pd.read_csv(
        DATA_DIR / "stats_player_reg_by_season" / f"{RECENT_SEASON}.csv",
        usecols=["player_id", "player_display_name", "position"],
    ).rename(columns={"player_display_name": "player_name"})
    parts.append(stats[["player_id", "player_name", "position"]])
    crosswalk = pd.concat(parts, ignore_index=True).dropna(subset=["player_id", "player_name"])
    return crosswalk.drop_duplicates("player_id", keep="first")


def _build_abbrev_name_crosswalk() -> pd.DataFrame:
    """(abbreviated "P.Rivers"-style name, team) -> full name.

    Keyed on (abbrev_name, team), not abbrev_name alone -- confirmed by
    inspection that abbreviations collide across real players (e.g.
    "B.Robinson" = both Bijan Robinson/ATL and Brian Robinson/SF).
    """
    stats = pd.read_csv(
        DATA_DIR / "stats_player_reg_by_season" / f"{RECENT_SEASON}.csv",
        usecols=["player_name", "player_display_name", "recent_team"],
    )
    stats = stats.rename(columns={
        "player_name": "abbrev_name", "player_display_name": "player_name", "recent_team": "team",
    })
    return stats.dropna(subset=["abbrev_name", "player_name", "team"]).drop_duplicates(
        ["abbrev_name", "team"], keep="first"
    )


def _latest_season(df: pd.DataFrame, season_col: str = "season", season: int = CURRENT_SEASON) -> pd.DataFrame:
    return df[df[season_col] == season].copy()


def _key(series: pd.Series) -> pd.Series:
    return series.apply(normalize_player_name)


# ---------------------------------------------------------------------------
# Injury inputs (v4 -- season-range fix, see module docstring)
# ---------------------------------------------------------------------------

def load_game_log_by_season(crosswalk: pd.DataFrame) -> dict[str, dict[int, int]]:
    """{player_id: {season: games_played}}, from every real
    stats_player_reg_by_season/*.csv row that has that player_id -- the
    single source of truth for "did this player have a row this season,
    and how many games did it show," which is exactly what
    injury_history.build_games_missed_by_season needs."""
    game_log: dict[str, dict[int, int]] = {}
    for path in sorted(STATS_DIR.glob("*.csv")):
        season = int(path.stem)
        df = pd.read_csv(path, usecols=["player_id", "games"])
        for player_id, games in zip(df["player_id"], df["games"]):
            if pd.isna(player_id):
                continue
            game_log.setdefault(player_id, {})[season] = int(games)
    return game_log


def load_draft_capital_by_id() -> dict[str, int]:
    """player_id -> draft_season, keyed by player_id (NOT normalized name --
    see module docstring for the confirmed Sr./Jr. collision this avoids)."""
    df = pd.read_csv(DRAFT_CAPITAL_CSV, usecols=["player_id", "draft_season"])
    df = df.dropna(subset=["player_id", "draft_season"]).drop_duplicates("player_id", keep="first")
    return df.set_index("player_id")["draft_season"].astype(int).to_dict()


def resolve_rookie_season(player_id, draft_capital_by_id: dict, game_log_by_id: dict) -> int | None:
    """draft_capital's draft_season (by player_id) is primary. Falls back to
    the earliest season this player_id has a real game-log row, for the
    ~193 UDFA-ish players not present in draft_capital_player_seasons.csv
    at all -- a documented approximation, not exact draft data. Returns
    None only when there's truly no signal at all (brand-new player with no
    draft record and no game-log row yet)."""
    if player_id in draft_capital_by_id:
        return draft_capital_by_id[player_id]
    log = game_log_by_id.get(player_id)
    if log:
        return min(log.keys())
    return None



# injury_status values that carry no real current-injury signal for this
# blend. Active/Inactive obviously don't. "Questionable" ALSO excluded
# (post-fix, real case: Jahmyr Gibbs was tagged Questionable in this data
# but is confirmed actually healthy and back at practice) -- this field is
# a point-in-time snapshot, not a live feed, and Questionable is the most
# short-lived, day-to-day designation of the set (tied to a single week's
# practice participation), so it's the most likely to have gone stale by
# the time anyone's looking at this board. IR/PUP/DNR/"Injured Reserve"/
# "Physically Unable to Perform"/Sus are more durable, season-defining
# designations, less likely to flip within days -- still blended in.
CURRENT_STATUS_BASELINE = {"Active", "Inactive", "Questionable"}


def _blend_current_injury_status(historical_score: float, injury_status, injury_risk_current) -> float:
    """Blends the purely historical, decay-weighted injury_risk_score_v4
    with master_players.csv's real, CURRENT injury_status (pre-existing
    field from the original data pull, e.g. Sleeper's own status feed --
    Active/Inactive=30, Questionable=65, IR/PUP=85 on a 0-100 scale).

    Found via direct user report, not assumed: George Kittle just tore his
    Achilles (injury_status="PUP") -- a real, forward-looking red flag a
    purely historical, completed-seasons-only formula structurally cannot
    see (his games_missed_by_season history alone gave him a defensible-
    looking but incomplete 50/100). The whole injury category had been
    ignoring this real, already-available current-status field entirely.

    Takes max(historical, current), not a blend or override -- either
    signal independently justifies concern, and a clean history shouldn't
    be allowed to mask an acute current injury. Only applies when status
    is something OTHER than Active/Inactive, so a healthy player's clean
    historical score isn't artificially floored at 30 just because
    "Active" carries a nonzero baseline in the pre-existing field."""
    if injury_status in CURRENT_STATUS_BASELINE or pd.isna(injury_risk_current):
        return historical_score
    historical_pct = (historical_score - 1) / 4 * 100
    blended_pct = max(historical_pct, float(injury_risk_current))
    return min(5.0, max(1.0, round(blended_pct / 100 * 4 + 1, 1)))


# Manual, dated overrides for real, currently-reported injury news that a
# purely historical/season-status pipeline structurally can't see yet --
# distinct from _blend_current_injury_status() above, which only reacts to
# master_players.csv's own injury_status field (itself a point-in-time
# Sleeper pull, and one that explicitly EXCLUDES "Questionable" from moving
# the score at all, see CURRENT_STATUS_BASELINE's docstring -- a genuinely
# recent, still-developing situation like this one is exactly the kind of
# day-to-day news that field is least likely to have caught).
#
# Jordyn Tyson (NO): real recurring hamstring history (2022, 2024, 2025 --
# cost him the Combine and Pro Day), re-injured the same hamstring in a
# real joint practice vs. JAX reported Aug 13-14 2026. Updated 2026-08-17:
# the initial "uncertain for Week 1" report (HC Kellen Moore, ESPN/NBC
# Sports/ProFootballRumors) has firmed up to a real, decisive out-until-
# Week-9 timeline (user-reported; the underlying re-injury and prior
# Week-1-uncertain reporting were independently corroborated via web search
# the same day, but no indexed source yet confirms the specific Week 9
# date -- treated as current per the user's direct report). His
# pre-override injury_score was 1.0 (the safest floor -- he has zero real
# NFL games_missed history on file as a 2026 rookie, and injury_status in
# master_players.csv still reads "Active", a stale pre-news snapshot).
# Forced to the ceiling per explicit instruction: this is real, current,
# reported information that should override a generic projection
# immediately, not something to model probabilistically. Score stays 5.0
# (already the ceiling of this scale) -- the update changes WHY it's
# floored (a confirmed multi-week absence, not just uncertainty) and the
# season-shape impact, which PROJECTION_MANUAL_ADJUSTMENTS in
# draft_analysis.py now carries separately (a real, distinct games-missed
# discount, not modeled by this 0-5 injury-risk scale at all).
# Sizing/mechanism/confidence rules for every entry below: see
# research/news_override_policy.md (written 2026-08-26 by extracting the
# rules already applied consistently across these entries -- read it before
# adding a new one, rather than re-deriving judgment calls from scratch).
INJURY_MANUAL_OVERRIDES = {
    "Jordyn Tyson": {
        "score": 5.0,
        "reason": "Hamstring re-injury -- real, decisive timeline: out until Week 9",
        "date": "2026-08-17",
    },
    # Ashton Jeanty (added 2026-08-26): real, current ankle sprain, helped
    # off the practice field without weight-bearing (Aug 24, per Rapoport/
    # Schefter). Reporting explicitly frames it as NOT long-term with a
    # subsequent "positive update," but the return timeline is still
    # unconfirmed as of this entry -- unlike Tyson's decisive out-until-
    # Week-9 case, there is no specific missed-games number to build a
    # PROJECTION_MANUAL_ADJUSTMENTS discount from yet. Scored 4.0, not the
    # 5.0 ceiling: real, meaningfully elevated near-term risk, deliberately
    # short of asserting a confirmed long absence the reporting itself
    # doesn't support. Revisit (both this score and a possible projection
    # discount) once a real return timeline firms up.
    "Ashton Jeanty": {
        "score": 4.0,
        "reason": "Sprained ankle at practice (Aug 24) -- not considered long-term per "
                  "reporting, but return timeline still unconfirmed",
        "date": "2026-08-26",
    },
    # Josh Jacobs (added 2026-08-26): TWO distinct real, current, undisclosed-
    # to-the-model risks compounding right now, neither visible to the
    # historical pipeline -- injury_status reads "Questionable" (explicitly
    # excluded from moving the score, see _blend_current_injury_status), so
    # his pre-override score (1.7) reflects neither:
    #   1. An existing groin injury, real and ongoing (reported return "late
    #      next week" as of Aug 14).
    #   2. A real, PENDING NFL suspension review (GM Gutekunst confirmed
    #      Green Bay is preparing a contingency RB plan) stemming from a May
    #      arrest -- no discipline has been announced and no charges have
    #      been filed as of this entry, so this is disclosed, elevated
    #      UNCERTAINTY, not a confirmed absence. No PROJECTION_MANUAL_
    #      ADJUSTMENTS discount applied for the same reason as Jeanty: no
    #      real, decisive length to build one from yet -- add one if/when a
    #      suspension (or its length) is confirmed.
    # Scored 3.5: real, compounding risk from two independent sources, kept
    # below Jeanty's 4.0 since neither individual factor here is as acute
    # as a practice-field injury with a diagnosis in hand.
    "Josh Jacobs": {
        "score": 3.5,
        "reason": "Existing groin injury (return eyed for late Aug) + a real, pending NFL "
                  "suspension review following a May arrest -- no discipline decided yet",
        "date": "2026-08-26",
    },
    # Luther Burden (added 2026-08-26, human-reviewed via research/pending_
    # news_adjustments.md): real groin injury at Bears practice (Aug 8),
    # missing the rest of the preseason, team "hopeful" for Week 1 -- short
    # of a decisive missed-games timeline (same standard as Jeanty), so
    # risk-only, no projection cut. His existing PROJECTION_MANUAL_
    # ADJUSTMENTS entry (+10%, DJ Moore trade/camp-role opportunity) is
    # unrelated -- opportunity, not this injury -- and stays untouched;
    # this is an ADDITIONAL, separate injury-score entry.
    "Luther Burden": {
        "score": 3.5,
        "reason": "Groin injury at practice (Aug 8), missing rest of preseason -- team "
                  "\"hopeful\" for Week 1 but no confirmed return date",
        "date": "2026-08-26",
    },
    # Puka Nacua (added 2026-08-26, human-reviewed via research/pending_
    # news_adjustments.md): real, still-unresolved suspension risk tied to
    # a civil lawsuit (alleged biting + antisemitic remark) -- Schefter says
    # missing Week 1 vs. SF "can't be ruled out," multiple outlets frame it
    # as a genuine coin flip. No discipline decided, no charges filed as of
    # this entry -- same shape as the Jacobs precedent (real, live,
    # undecided legal/suspension risk, not a physical injury), so
    # risk-only, no projection cut despite the high stakes (top-4 ADP).
    "Puka Nacua": {
        "score": 3.5,
        "reason": "Real, unresolved NFL suspension risk tied to a civil lawsuit -- "
                  "Schefter says missing Week 1 \"can't be ruled out,\" nothing decided",
        "date": "2026-08-26",
    },
    # Jayden Higgins (added 2026-08-27, auto-accept): CONFIRMED torn ACL,
    # suffered during joint practices with the Raiders (per multiple outlets
    # covering the Patriots-Texans Kayshon Boutte trade the same week -- "a
    # week after Houston lost wide receiver Jayden Higgins to a torn ACL
    # during joint practices with the Las Vegas Raiders" is stated as
    # established context, not a rumor, across ESPN/NFL.com/CBS Sports
    # coverage). Season-ending, no timeline ambiguity -- the most decisive
    # possible injury tier. Pre-override pipeline had him at 1.0/Active,
    # meaning the automated Sleeper-status feed had not yet caught this at
    # all. Ceiling score, same tier as Tyson's confirmed multi-week
    # absence -- if anything more decisive, since an ACL tear has no
    # realistic in-season return. A PROJECTION_MANUAL_ADJUSTMENTS cut is
    # also clearly warranted (near-total loss of 2026 value) but drafted to
    # research/pending_news_adjustments.json for review per SS1a rather than
    # applied here, same as every projection-affecting lever.
    "Jayden Higgins": {
        "score": 5.0,
        "reason": "Confirmed torn ACL during joint practices with the Raiders -- season-ending",
        "date": "2026-08-27",
    },
    # TreVeyon Henderson (added 2026-08-27, auto-accept): real, fresh
    # ankle/leg injury -- did not finish an August 2026 practice after
    # slipping on a cut, per multiple outlets covering Patriots training
    # camp. Return timeline unconfirmed as of this entry -- same shape as
    # the Jeanty precedent (real current injury, no decisive missed-games
    # number yet), scored the same 4.0 rather than the 5.0 ceiling. Notably,
    # real depth-chart reporting names Rhamondre Stevenson (not Henderson)
    # as the Patriots' clear early-down RB1 even before this injury --
    # documented here for context, but that's a role/usage read, not itself
    # grounds for an injury_score change on its own.
    "TreVeyon Henderson": {
        "score": 4.0,
        "reason": "Ankle/leg injury -- did not finish practice (late Aug 2026), return timeline unconfirmed",
        "date": "2026-08-27",
    },
}


def compute_injury_v4(row, game_log_by_id: dict, draft_capital_by_id: dict) -> pd.Series:
    player_id = row.get("player_id")
    if not isinstance(player_id, str) or not player_id:
        # No resolvable identity at all -- correct floor, matches the
        # true-rookie case in injury_history's own test suite. Still runs
        # through the current-status blend below -- a true unknown could
        # still carry a real current injury_status.
        acute_score = chronic_score = historical_score = 1.0
        games_missed, severity = {}, {}
    else:
        rookie_season = resolve_rookie_season(player_id, draft_capital_by_id, game_log_by_id)
        if rookie_season is None:
            rookie_season = CURRENT_SEASON  # no draft record, no game log -> treat as incoming rookie

        game_log = game_log_by_id.get(player_id, {})
        games_missed, severity = assemble_injury_inputs(
            rookie_season=rookie_season, current_season=CURRENT_SEASON, game_log_source=game_log,
        )
        age = row.get("age")
        age_val = int(age) if pd.notna(age) else 24
        acute_score, chronic_score, historical_score = injury_risk_score_v4_detailed(
            games_missed, severity, age=age_val, current_season=CURRENT_SEASON,
        )

    score = _blend_current_injury_status(historical_score, row.get("injury_status"), row.get("injury_risk"))

    override = INJURY_MANUAL_OVERRIDES.get(row.get("player_name"))
    override_note = None
    if override is not None:
        score = override["score"]
        override_note = f"{override['reason']} (reported {override['date']})"

    return pd.Series({
        "injury_score": score,
        # Real, disclosed sub-scores (plan_risk_index_reweight_dynamic.pdf
        # Phase 2) -- surfaced in Home.py's popover so a user can see
        # whether a score is acute- or chronic-driven, not just the final
        # blended number. Not touched by the current-status blend or manual
        # override above (those only affect the combined injury_score).
        "injury_acute_score": acute_score,
        "injury_chronic_score": chronic_score,
        "games_missed_by_season": json.dumps(games_missed),
        "injury_severity_by_season": json.dumps(severity),
        "injury_override_note": override_note,
    })


def load_career_touches(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Sum of carries+receptions across every available season file.

    Computed and stored for transparency/future use -- matching the spec's
    own injury_risk_score, which lists career_touches as an input in its
    docstring but never references it in the actual formula body, this is
    NOT fed into the injury formula here either.
    """
    frames = []
    for path in sorted((DATA_DIR / "stats_player_reg_by_season").glob("*.csv")):
        df = pd.read_csv(path, usecols=lambda c: c in {"player_id", "carries", "receptions"})
        if "player_id" not in df.columns:
            continue
        df["touches"] = df.get("carries", 0).fillna(0) + df.get("receptions", 0).fillna(0)
        frames.append(df[["player_id", "touches"]])
    all_touches = pd.concat(frames, ignore_index=True)
    career = all_touches.groupby("player_id")["touches"].sum().rename("career_touches").reset_index()
    career = career.merge(crosswalk[["player_id", "player_name"]], on="player_id", how="left")
    return career[["player_name", "career_touches"]].dropna(subset=["player_name"])


# ---------------------------------------------------------------------------
# Role/Usage/TD inputs (v5 -- real target_share_rate + percentile scoring)
# ---------------------------------------------------------------------------

def load_snap_share() -> pd.DataFrame:
    df = _latest_season(pd.read_csv(DATA_DIR / "snap_share_player_seasons.csv"))
    return df[["player_name", "team", "position", "prior_snap_share"]].rename(
        columns={"prior_snap_share": "snap_share"}
    )


def _build_team_plays_per_game(season: int, play_columns: list[str]) -> dict[str, float]:
    """{team: sum(play_columns) per game} for a real season, derived from
    stats_player_reg_by_season -- sum every player's plays credited to
    that team, divide by the team's own real games-played (taken as the
    max games value on the roster, robust across historical era game-count
    changes without hardcoding 16 vs 17).

    Every season file (confirmed across all of 1999-2025) carries exactly
    one junk row with player_id=NaN and a garbage games value (e.g. 272 for
    BAL/2025) -- almost certainly a league-total/placeholder row from the
    export process. load_game_log_by_season already guards against this via
    its own player_id check; this needed the same guard, since an
    unfiltered max(games) of 272 was corrupting BAL's denominator down to
    ~1.6 attempts/game (vs. a real ~30-40), which in turn made Derrick
    Henry's target_share_rate compute to an impossible 79.6% -- caught via
    the v5 full-pool spot check, not assumed correct. games is also clipped
    to 17 (this era's real max) as a second guard against smaller
    off-by-one anomalies (e.g. one real player logged at 18 games).

    KNOWN BROKEN for team-total reconstruction, kept only as
    build_team_games_per_season()'s source (team_games, the max-games
    value, is unaffected -- see that function) -- do not use this for a
    per-game PLAYS total. Grouping by recent_team silently drops a
    mid-season-traded player's attempts from their FIRST team entirely,
    since nflverse folds a traded player's whole season into their
    SEASON-ENDING team only. Real, confirmed 2025 case: Joe Flacco started
    with Cleveland, moved to Cincinnati mid-season after Joe Burrow's
    injury -- his 416 real attempts landed entirely in Cincinnati's total,
    none in Cleveland's, corrupting CLE's total to an impossibly-low 398
    attempts (lowest in the league) and CIN's to an impossibly-high 800
    (highest), while every other team fell in a tight, plausible 422-649
    range. See _load_team_offense_rates() for the real fix."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=["player_id", "recent_team", "games", *play_columns])
    df = df.dropna(subset=["player_id"])
    df["games"] = df["games"].clip(upper=FULL_SEASON_GAMES)
    df["plays"] = df[play_columns].fillna(0).sum(axis=1)
    grouped = df.groupby("recent_team").agg(total_plays=("plays", "sum"), team_games=("games", "max"))
    grouped = grouped[grouped["team_games"] > 0]
    return (grouped["total_plays"] / grouped["team_games"]).to_dict()


def build_team_games_per_season(season: int) -> dict[str, float]:
    """{team: real games played this season} -- the max real games value
    on that team's roster (clipped to FULL_SEASON_GAMES), same convention
    _build_team_plays_per_game already used for its own team_games. Unlike
    that function's PLAYS total, this max-games value is NOT corrupted by
    mid-season trades (a team's own game count doesn't change when one
    player on the roster gets traded away), so it's safe to keep sourcing
    from stats_player_reg_by_season directly."""
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=["player_id", "recent_team", "games"])
    df = df.dropna(subset=["player_id"])
    df["games"] = pd.to_numeric(df["games"], errors="coerce").clip(upper=FULL_SEASON_GAMES)
    return df.groupby("recent_team")["games"].max().to_dict()


def _load_team_offense_rates(season: int) -> tuple[dict[str, float], dict[str, float]]:
    """{team: real pass attempts per game}, {team: real total plays per
    game} for `season`, sourced from offense_environment_team_seasons.csv's
    own team-level PBP aggregation instead of _build_team_plays_per_game's
    reconstruction-by-summing-player-rows -- immune to the mid-season-trade
    bug documented on that function, since this file's totals are built
    directly from each week's real team, not a player's season-ending
    recent_team label.

    That file's own "prior_" columns hold real season `season` data filed
    under the season+1 label (the same repo-wide convention documented in
    this module's docstring and load_offense_environment) -- confirmed by
    diffing this source against _build_team_plays_per_game across all 32
    real 2025 teams: the two agree within rounding everywhere EXCEPT
    CLE/CIN, the exact two teams the real Flacco trade broke (CLE: 23.4 ->
    32.8 real pass attempts/game; CIN: 47.1 -> 37.6), which is itself
    confirmation this source doesn't share that bug."""
    df = pd.read_csv(OFFENSE_CSV)
    df = df[df["season"] == season + 1]
    total_pg = df.set_index("team")["prior_team_plays_pg"]
    pass_pg = total_pg * df.set_index("team")["prior_team_pass_rate"]
    return pass_pg.to_dict(), total_pg.to_dict()


def build_team_pass_attempts_per_game(season: int = RECENT_SEASON) -> dict[str, float]:
    """{team: real pass attempts per game} -- see _load_team_offense_rates."""
    return _load_team_offense_rates(season)[0]


def build_team_total_plays_per_game(season: int = RECENT_SEASON) -> dict[str, float]:
    """{team: real (pass attempts + carries) per game} -- the denominator
    RBs' opportunity_share_rate needs (see load_target_share_rate). A power
    back's real workload dominance shows up in carries, not targets; using
    pass-attempts-only would badly understate it. See
    _load_team_offense_rates for why this isn't _build_team_plays_per_game."""
    return _load_team_offense_rates(season)[1]


def load_target_share_rate(crosswalk: pd.DataFrame, season: int = RECENT_SEASON) -> pd.DataFrame:
    """Two real usage rates, both rate-per-game-played (never divided by
    games the player didn't play -- see module docstring for the dilution
    bug this avoids):
      - target_share_rate: pure receiving share. The right measure of
        receiving involvement for every position.
      - opportunity_share_rate: total workload share ((carries+targets) /
        team total plays pg) for RBs; for QBs, RUSHING share specifically
        (carries / team total plays pg) -- a QB's passing workload is
        already fully captured by snap_share (he throws or hands off on
        nearly every snap he's on the field for regardless of talent), so
        what's DISTINCTIVE and risk-reducing for a QB is his own added
        rushing floor. Identical to target_share_rate for WR/TE, where
        rushing volume isn't a meaningful role signal.

    target_share_rate is sourced from nflverse's own real `target_share`
    column (targets / team TARGETS, computed by nflverse directly from
    real weekly team data), rescaled by team_games/player_games -- NOT
    reconstructed from team pass attempts the way opportunity_share_rate
    still is. Two real, confirmed-independently bugs in the old attempts-
    based reconstruction, both fixed by this switch:
      1. Stint/trade bug: summing attempts by a player's SEASON-ENDING
         recent_team (see _build_team_plays_per_game's docstring for the
         real Flacco CLE->CIN 2025 case). nflverse's own target_share
         column is immune to this -- confirmed by diffing corrected values
         against real games-played-adjusted numbers for the 4 affected CLE/
         CIN receivers (Jeudy 26.6%->20.3%, Tillman 13.9%->7.5%->10.6%
         games-corrected, Chase 24.6%->30.4%->32.3%, Higgins 13.9%->16.1%
         ->18.3%), all landing in the plausible range independent sources
         (nflverse's own column) confirm.
      2. Denominator bug: the old code divided by team pass ATTEMPTS
         (includes throwaways/spikes/uncatchable heaves with no targeted
         receiver); nflverse's target_share denominates on team TARGETS,
         the structurally correct denominator for a *target* share.
    Same formula and same real McCaffrey-dilution rationale as
    build_te_archetypes.py's load_te_target_share_corrected() (see that
    function's docstring for the full original investigation) -- this
    generalizes that TE-only fix to every position sharing this function.

    opportunity_share_rate keeps the attempts/carries-based team-total
    denominator (no nflverse "opportunity_share" column exists to borrow),
    but now sources that team total from _load_team_offense_rates(), which
    fixes the same stint bug for RBs'/QBs' workload share.

    Why both target_share_rate and opportunity_share_rate, not just one
    (post-v5 fix, confirmed on real data, three rounds): a receiving-
    share-only view badly understated a power back's real workload
    dominance -- Jonathan Taylor's 2025 target share (10.1%) put him only
    ~6 points ahead of his closest target-share "competitor" Ameer
    Abdullah (4.1%), a modest-looking gap, but Abdullah is a passing-down
    specialist with almost no real touches (31 total) against Taylor's
    378 -- a 12x gap, not ~2.5x. Switching RBs to opportunity_share_rate
    alone fixed that, but then made the OPPOSITE mistake for a pure
    between-the-tackles back: Derrick Henry's total workload (328 touches,
    nearly all carries) correctly shows him as the clear lead back, but
    his real receiving involvement is next to nothing (21 targets in 17
    games) -- a distinct, real risk (limited on passing downs, exposed to
    negative game script) that opportunity share alone can't see, since
    rushing volume swamps it in the blend. Keeping both lets
    role_usage_td_score price each dimension separately instead of one
    hiding the other.

    QBs had the exact same blind spot as RBs originally did, just
    unnoticed until checked directly: Jayden Daniels and Joe Burrow (both
    real rushing/dual-threat value) showed target_share_percentile ==
    opportunity_share_percentile == their (near-meaningless) receiving
    share, since only RB got the opportunity split. A dual-threat QB's
    real rushing floor was invisible to both usage_risk and the
    td_reliance_risk discount.

    Also carries `games` (that season's real games played) for
    role_score_with_confidence's small-sample dampening, and
    `target_share_data_quality_flag` -- "corrected" when the games
    rescaling moved a player's target_share_rate by more than
    TARGET_SHARE_FLAG_THRESHOLD, empty string otherwise -- the disclosed
    marker consumers should surface rather than presenting every number at
    identical confidence."""
    team_plays_pg = build_team_total_plays_per_game(season)
    team_games = build_team_games_per_season(season)
    path = DATA_DIR / "stats_player_reg_by_season" / f"{season}.csv"
    df = pd.read_csv(path, usecols=[
        "player_id", "position", "recent_team", "targets", "carries", "games", "target_share",
    ])
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()
    # Clipped to FULL_SEASON_GAMES for the same real reason
    # _build_team_plays_per_game's own games column is -- confirmed one
    # real anomalous row here too (Rashid Shaheed, games=18, an impossible
    # value in a 17-game season), which without this clip understates his
    # own target_share_rate/opportunity_share_rate by dividing across a
    # game he's credited with but that can't really exist.
    df["games"] = pd.to_numeric(df["games"], errors="coerce").clip(upper=FULL_SEASON_GAMES)
    df["team_games"] = df["recent_team"].map(team_games)

    raw_target_share = pd.to_numeric(df["target_share"], errors="coerce")
    df["target_share_rate"] = np.where(
        (df["games"] > 0) & df["team_games"].notna(),
        raw_target_share * df["team_games"] / df["games"],
        np.nan,
    )
    df["target_share_data_quality_flag"] = np.where(
        df["target_share_rate"].notna() & raw_target_share.notna()
        & ((df["target_share_rate"] - raw_target_share).abs() > TARGET_SHARE_FLAG_THRESHOLD),
        "corrected", "",
    )

    df["team_plays_per_game"] = df["recent_team"].map(team_plays_pg)
    opportunity_denom = df["team_plays_per_game"] * df["games"]
    rb_opportunity_rate = np.where(
        (df["games"] > 0) & (opportunity_denom > 0),
        (df["carries"].fillna(0) + df["targets"].fillna(0)) / opportunity_denom,
        np.nan,
    )
    qb_rushing_rate = np.where(
        (df["games"] > 0) & (opportunity_denom > 0), df["carries"].fillna(0) / opportunity_denom, np.nan
    )
    df["opportunity_share_rate"] = np.select(
        [df["position"] == "RB", df["position"] == "QB"],
        [rb_opportunity_rate, qb_rushing_rate],
        default=df["target_share_rate"],
    )

    df = df.merge(crosswalk[["player_id", "player_name"]], on="player_id", how="left")
    df = df.rename(columns={"recent_team": "team", "games": "games_recent"})
    return df[["player_name", "team", "position", "target_share_rate", "opportunity_share_rate",
               "games_recent", "target_share_data_quality_flag"]].dropna(subset=["player_name"])


QB_DEFAULT_COMPETITION = 0.5

ROSTER_CROWDING_TIER_CAP = 5  # 4+ established teammates -> tier 5, matches real observed max (see caller)


def compute_roster_crowding_tiers(season: int, positions: tuple[str, ...] = ("RB", "WR")) -> dict:
    """Real per-(team, position) "how crowded is this room already" tier,
    for scoring an INCOMING rookie's roster_competition_tier -- built for
    rookie_data_backfill_plan.pdf's follow-up validation pass, which found
    roster_competition_tier was a flat manual default (3, "average") for
    every single rookie, contributing zero real variance to the composite.

    Different question, different function, from compute_depth_chart_competition()
    above: that one measures an EXISTING player's own week-to-week gap
    uncertainty against a teammate (needs the player's own real share, so
    it's meaningless for a rookie with zero real snaps). This instead counts
    how many teammates ALREADY had a real, meaningfully-involved share
    (opportunity_share_rate >= DEPTH_CHART_SNAP_THRESHOLD) in the season
    BEFORE the rookie's evaluation point -- a real proxy for "how much
    established competition is this rookie walking into," derivable purely
    from real NFL usage data, no rookie-specific data needed.

    season-parameterized (unlike compute_depth_chart_competition, which is
    hardcoded to the module's RECENT_SEASON) so callers can ask about ANY
    real historical season -- the 2026 board wants the most recent real
    completed season (2025); the 2023-2025 backtest wants the real season
    immediately before each rookie's own real draft year (e.g. a 2024
    draftee's incoming competition is real 2023 usage, not today's).

    Real observed distribution (2022-2025, confirmed before choosing these
    cutoffs): 0-4 real "established" teammates per team/position, no team
    ever hit 5+. Bucketed directly count->tier (0->1 clearest path, 4+->5
    most crowded) rather than an arbitrary formula, since the real range
    already spans the full 1-5 scale on its own.

    Returns {(team_code, position): tier} -- team_code matches this
    session's "standard" convention (LAR, not nflverse's LA -- see the
    inverse-alias step below, same fix as
    build_backtest_rookie_inputs.load_historical_offense_environment)."""
    crosswalk = _build_id_name_crosswalk()
    df = load_target_share_rate(crosswalk, season=season)
    df = df[df["position"].isin(positions)]
    crowded = df[df["opportunity_share_rate"] >= DEPTH_CHART_SNAP_THRESHOLD]
    counts = crowded.groupby(["team", "position"]).size()

    tiers = {}
    for team, position in df[["team", "position"]].drop_duplicates().itertuples(index=False):
        count = int(counts.get((team, position), 0))
        std_team = {v: k for k, v in TEAM_CODE_ALIASES.items()}.get(team, team)
        tiers[(std_team, position)] = min(ROSTER_CROWDING_TIER_CAP, count + 1)
    return tiers


def compute_depth_chart_competition(target_share_df: pd.DataFrame) -> pd.DataFrame:
    """Gap-MAGNITUDE-based competition risk vs. the closest same-team/
    -position teammate's real opportunity_share_rate (adapted from spec
    Section 4c's gap formula, adapted to most-recent-completed-season real
    data -- see module docstring for why no live in-season data exists to
    refresh this weekly here).

    Uses abs(gap), not signed gap (post-v5 fix, real case: Tee Higgins --
    a clear, entrenched WR2 trailing Ja'Marr Chase by a real 10.7-point
    target-share gap was scoring the MAXIMUM competition risk, same as a
    genuine 50/50 committee situation). What this term should measure is
    UNCERTAINTY in who gets volume, not "not being the team's best
    player." A LARGE gap in either direction means a stable, predictable
    pecking order -- Higgins isn't fighting Chase for targets week to
    week, he just has a smaller, well-defined role, which his OWN
    target_share_percentile already prices separately. A SMALL gap
    (whether nominally ahead or behind) is what real week-to-week
    uncertainty looks like -- two backs genuinely splitting a backfield,
    two WRs in a real timeshare -- and that's what should score high here.

    QB is excluded from the gap calculation entirely and gets a flat
    QB_DEFAULT_COMPETITION instead -- tried snap_share as the QB-specific
    comparison metric first (targets/carries are near-zero for backups
    regardless of how entrenched the starter is, so gap-by-workload is
    pure noise at QB), but snap_share is a season-TOTAL rate with no raw
    counts to games-adjust (same documented limitation as nflverse's own
    target_share column). Confirmed broken on real data: Jayden Daniels
    missed real time to injury in 2025, so his backup (Marcus Mariota)
    accumulated MORE season-long snap share filling in (48.4% vs Daniels'
    38.4%) -- scoring the injured starter as if he'd lost a real
    competition. Genuine QB competitions are rare enough, and this data
    unreliable enough, that a flat low default is more honest than a
    metric that actively inverts for the exact players (injured starters)
    it matters most to get right."""
    rows = []
    for (team, position), group in target_share_df.groupby(["team", "position"]):
        if position == "QB":
            for player_name in group["player_name"]:
                rows.append({"player_name": player_name, "depth_chart_competition": QB_DEFAULT_COMPETITION})
            continue
        shares = group.set_index("player_name")["opportunity_share_rate"]
        for player_name, own_share in shares.items():
            if pd.isna(own_share):
                continue
            teammates = shares.drop(index=player_name).dropna()
            if teammates.empty:
                rows.append({"player_name": player_name, "depth_chart_competition": 0.0})
                continue
            gap = abs(own_share - teammates.max())
            if gap >= 0.10:
                competition = 0.5
            elif gap >= 0.05:
                competition = 1.5
            else:
                competition = 2.5
            rows.append({"player_name": player_name, "depth_chart_competition": competition})
    return pd.DataFrame(rows)


def build_position_percentiles(board: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank target_share_rate, opportunity_share_rate, and
    snap_share within each position group of the CURRENT draft pool (0-1,
    NaN stays NaN) -- this is what lets an elite real share (e.g. ~28-30%
    target share, or a bellcow RB's opportunity share) register near its
    true ~95th percentile instead of being judged against an unreachable
    100% ceiling. target_share_percentile and opportunity_share_percentile
    are identical for non-RBs (see load_target_share_rate)."""
    out = board[["target_share_rate", "opportunity_share_rate", "snap_share"]].copy()
    out["target_share_percentile"] = board.groupby("position")["target_share_rate"].rank(pct=True)
    out["opportunity_share_percentile"] = board.groupby("position")["opportunity_share_rate"].rank(pct=True)
    out["snap_share_percentile"] = board.groupby("position")["snap_share"].rank(pct=True)
    return out[["target_share_percentile", "opportunity_share_percentile", "snap_share_percentile"]]


def role_score_with_confidence(raw_score: float, games_played_recent) -> float:
    """Dampens toward a neutral 2.5 when the player's own most-recent-
    season real games played is small -- substitutes for the spec's live
    'games_played_recent' in-season window (see module docstring)."""
    games = 0 if pd.isna(games_played_recent) else float(games_played_recent)
    if games < 4:
        confidence = games / 4
        return round(raw_score * confidence + NEUTRAL_SCORE * (1 - confidence), 1)
    return raw_score


def load_redzone(abbrev_crosswalk: pd.DataFrame) -> pd.DataFrame:
    df = _latest_season(pd.read_csv(DATA_DIR / "redzone_airyards_player_seasons.csv"))
    df = df.rename(columns={"player_name": "abbrev_name"})
    df = df.merge(abbrev_crosswalk, on=["abbrev_name", "team"], how="left")
    return df[["player_name", "prior_redzone_target_share", "prior_redzone_carry_share"]].rename(columns={
        "prior_redzone_target_share": "redzone_target_share",
        "prior_redzone_carry_share": "redzone_carry_share",
    }).dropna(subset=["player_name"])


def load_td_rate(crosswalk: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / "stats_player_reg_by_season" / f"{RECENT_SEASON}.csv"
    df = pd.read_csv(path, usecols=[
        "player_id", "position", "passing_tds", "rushing_tds", "receiving_tds",
        "passing_yards", "rushing_yards", "receiving_yards", "carries", "receptions", "attempts",
    ])
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()
    df["total_tds"] = df[["passing_tds", "rushing_tds", "receiving_tds"]].fillna(0).sum(axis=1)
    # Real pass attempts, not a passing_yards/250 proxy -- that proxy
    # massively undercounted a QB's real volume (Joe Burrow's 2025: 259
    # real attempts vs. a 1809/250=7.2 "touches" estimate, a ~36x gap),
    # mechanically inflating any efficient QB's TD rate. attempts is
    # already a real column in this file (used elsewhere for
    # build_team_pass_attempts_per_game) -- no reason to approximate it.
    df["total_touches"] = df[["carries", "receptions", "attempts"]].fillna(0).sum(axis=1)
    df["td_rate"] = np.where(df["total_touches"] > 0, df["total_tds"] / df["total_touches"], np.nan)
    # Ratio, not an absolute difference -- the v5 formula's
    # max(0, td_rate_vs_baseline - 1.0) only makes sense against a ratio
    # baseline of 1.0 (exactly average). This replaces v1/v2's ad-hoc *5
    # scaling constant, which existed only because this field used to hold
    # an absolute difference the spec's own formula wasn't written for.
    baseline = df.groupby("position")["td_rate"].transform("mean")
    df["td_rate_vs_baseline"] = np.where(baseline > 0, df["td_rate"] / baseline, np.nan)
    df = df.merge(crosswalk[["player_id", "player_name"]], on="player_id", how="left")
    return df[["player_name", "td_rate_vs_baseline"]].dropna(subset=["player_name"])


def role_usage_td_score(row) -> float:
    """v5 percentile-based formula (spec Section 4). Percentile scoring
    naturally handles QBs without a special-case default (unlike the old
    absolute-value formula, which needed one) -- QBs rank against other
    QBs' own near-zero target_share_rate, so a QB's usage_risk correctly
    ends up driven by snap_share_percentile (their real starting-job-
    security signal) without needing an explicit carve-out.

    Missing percentile (no real most-recent-season row at all) defaults to
    a neutral 0.5, not 0.0 or 1.0 -- "no data" shouldn't imply either
    maximum security or maximum risk. redzone shares are computed/stored
    but intentionally unused here, matching the spec's own formula body."""
    target_pct = row.get("target_share_percentile")  # pure receiving involvement
    target_pct = 0.5 if pd.isna(target_pct) else float(target_pct)

    opportunity_pct = row.get("opportunity_share_percentile")  # total workload (RB: rush+rec; else == target_pct)
    opportunity_pct = 0.5 if pd.isna(opportunity_pct) else float(opportunity_pct)

    snap_pct = row.get("snap_share_percentile")
    snap_pct = 0.5 if pd.isna(snap_pct) else float(snap_pct)

    competition = row.get("depth_chart_competition")
    competition = 0.0 if pd.isna(competition) else float(competition)

    td_vs_baseline = row.get("td_rate_vs_baseline")
    td_vs_baseline = 1.0 if pd.isna(td_vs_baseline) else float(td_vs_baseline)  # 1.0 = exactly baseline, no signal

    # Split evenly between total workload security and pure receiving
    # involvement (post-v5 fix, round 3) -- identical to the old single
    # (1-target_pct)*2.0 for non-RBs, where opportunity_pct==target_pct.
    # For RBs the two dimensions can diverge sharply: opportunity_share
    # (round 2's fix) correctly recognizes a pure between-the-tackles back
    # as the workload leader, but that alone hid a real, distinct risk --
    # Derrick Henry's total-touch dominance (328, nearly all carries)
    # floored his score even though his receiving involvement (21 targets
    # in 17 games) is next to nothing, which matters on its own (limited
    # on passing downs, exposed to negative game script). Splitting the
    # term prices both instead of one hiding the other.
    usage_risk = (1 - opportunity_pct) * 1.0 + (1 - target_pct) * 1.0
    snap_risk = (1 - snap_pct) * 1.5
    competition_risk = competition * 0.8
    # Discounted by usage_security (post-v5 fix, round 2): a raw cap of 2.0
    # still let this term dominate the composite for a real elite bellcow
    # RB (Gibbs 2.16x baseline, Taylor 2.08x) with excellent usage/snap
    # percentiles -- because it was punishing an elevated TD rate as risk
    # regardless of WHY it's elevated. A bellcow with elite real volume
    # (huge total workload, no clear backup) scoring more TDs isn't
    # TD-DEPENDENT (lucky, prone to regression) -- his TD rate is explained
    # by legitimate opportunity, which usage_risk/snap_risk already price
    # in. A committee back scoring on limited touches IS TD-dependent --
    # his elevated rate isn't explained by volume, so it should still bite.
    # usage_security uses opportunity_pct (total workload), not receiving
    # share -- TD rate combines rush+receiving TDs, so what explains it is
    # overall volume, not receiving involvement specifically.
    usage_security = (opportunity_pct + snap_pct) / 2
    td_reliance_risk = min(2.0, max(0.0, td_vs_baseline - 1.0) * 3) * (1 - usage_security)

    raw = usage_risk + snap_risk + competition_risk + td_reliance_risk
    raw_score = min(5.0, max(1.0, round(raw, 1)))
    return role_score_with_confidence(raw_score, row.get("games_recent"))


def apply_usage_risk_overrides(board):
    """Manual usage-risk overrides, sourced from the same
    PROJECTION_MANUAL_ADJUSTMENTS table draft_analysis.py uses for
    projection_points -- one table, not a second parallel mechanism. Unlike
    that table's pct field (read-time, live on every Streamlit rerun), this
    is consulted here at CSV-build time, same as INJURY_MANUAL_OVERRIDES --
    a role_usage_td_score change only takes effect after this script is
    re-run. Direction is never auto-derived from pct (a bigger role doesn't
    always mean lower risk): usage_risk_score is an explicit, separately
    judged absolute override (1.0-5.0), defaulting to no change (key absent)
    for every entry that hasn't had that specific call made."""
    board["role_usage_td_override_note"] = None
    for name, override in PROJECTION_MANUAL_ADJUSTMENTS.items():
        usage_score = override.get("usage_risk_score")
        if usage_score is None:
            continue
        mask = board["player_name"] == name
        if not mask.any():
            continue
        board.loc[mask, "role_usage_td_score"] = usage_score
        board.loc[mask, "role_usage_td_override_note"] = (
            f"{override['note']} ({override['source']}, {override['date']})"
        )
    return board


# ---------------------------------------------------------------------------
# Offense Environment (team-level; substitutes real EPA/pace/scoring for
# the unavailable qb_tier/oline_rank/oc_continuity, per user's decision)
# ---------------------------------------------------------------------------

def load_offense_environment(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """season defaults to CURRENT_SEASON (the normal case everywhere this
    is called). build_wr_archetypes.py's QB-context starter-injury override
    passes season=CURRENT_SEASON-1 for specific teams -- see
    load_qb_context_by_team()'s docstring for why."""
    df = pd.read_csv(OFFENSE_CSV)
    df = df[df["season"] == season].copy()
    df["epa_rank"] = df["prior_team_epa_pg"].rank(ascending=False, method="first")   # 1 = best offense
    df["pace_rank"] = df["prior_team_plays_pg"].rank(ascending=False, method="first")  # 1 = fastest
    df["scoring_rank"] = df["prior_team_tds_pg"].rank(ascending=False, method="first")  # 1 = most TDs
    return df[["team", "epa_rank", "pace_rank", "scoring_rank"]]


def offense_environment_score(row) -> float:
    """Redesigned formula, not the spec's literal one -- qb_tier/oline_rank/
    oc_continuity don't exist anywhere in this repo (see module docstring).
    epa_rank carries the dominant weight since team EPA/play is the
    standard single metric for offense quality, implicitly capturing QB
    and O-line quality jointly."""
    epa_risk = (row["epa_rank"] / 32) * 3.0
    pace_risk = (row["pace_rank"] / 32) * 1.0
    scoring_risk = (row["scoring_rank"] / 32) * 1.0
    raw = epa_risk + pace_risk + scoring_risk
    return min(5.0, max(1.0, round(raw, 1)))


# ---------------------------------------------------------------------------
# Schedule/Weather/Venue (team-level; from build_schedule_data.py's output)
# ---------------------------------------------------------------------------

def load_schedule_risk() -> pd.DataFrame:
    return pd.read_csv(SCHEDULE_CSV)


def schedule_weather_venue_score(row) -> float:
    """v5 formula -- home_climate/road_cold_games (see build_schedule_data.py)
    replace dome_games/outdoor_cold_games, which incorrectly counted a
    team's OWN home dome games as risk-reducing (a stadium type is a
    constant, not a risk factor). Primetime/short-week games are capped at
    4 before the fatigue weight, not linear -- a team with 6 primetime
    games shouldn't scale 6x higher than a team with 1.

    season_sos_risk (post-v5 fix) uses a WHOLE-SEASON opponent-strength
    proxy, not weeks-15-17-only -- see build_schedule_data.py's docstring
    for the confirmed real mismatch this fixes (Cincinnati's real 2026
    schedule is the league's easiest by this proxy, but the narrow
    playoff-weeks-only version ranked them middling)."""
    weather_risk = row["road_cold_games"] * (0.5 if row["home_climate"] == "outdoor" else 0.3)
    season_sos_risk = (1 - row["season_sos_rank"] / 32) * 2
    bye_risk = 0.5 if row["bye_week"] <= 6 else 0.0
    primetime_capped = min(row["short_week_or_primetime_games"], 4)
    fatigue_risk = primetime_capped * 0.2
    raw = weather_risk + season_sos_risk + bye_risk + fatigue_risk
    return min(5.0, max(1.0, round(raw, 1)))


# ---------------------------------------------------------------------------
# Value signal (market) + volatility diagnostic -- excluded from risk_index
# ---------------------------------------------------------------------------

def load_value_and_volatility(master: pd.DataFrame) -> pd.DataFrame:
    if not SPORTSBOOK_CSV.exists():
        return pd.DataFrame(columns=["player_name", "value_signal_market", "volatility_diagnostic"])
    book = pd.read_csv(SPORTSBOOK_CSV)
    merged = master[["player_name", "projection_points"]].merge(
        book[["player_name", "sportsbook_half_ppr_points", "position_rank_gap"]],
        on="player_name", how="inner",
    )
    denom = merged[["projection_points", "sportsbook_half_ppr_points"]].mean(axis=1).replace(0, np.nan)
    merged["volatility_diagnostic"] = (
        (merged["projection_points"] - merged["sportsbook_half_ppr_points"]).abs() / denom
    )
    merged["value_signal_market"] = merged["position_rank_gap"]  # signed: + = market likes him more than ADP
    return merged[["player_name", "value_signal_market", "volatility_diagnostic"]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_risk_variables() -> pd.DataFrame:
    master = pd.read_csv(MASTER_CSV)
    master = master[master["position"].isin(SKILL_POSITIONS)].copy()
    has_adp = pd.to_numeric(master.get("adp"), errors="coerce").notna()
    has_real_proj = master.get("projection_source", pd.Series(dtype=object)) == "real"
    master = master[has_adp | has_real_proj].copy()
    master["_key"] = _key(master["player_name"])

    crosswalk = _build_id_name_crosswalk()
    abbrev_crosswalk = _build_abbrev_name_crosswalk()
    snap_share_df = load_snap_share()
    target_share_df = load_target_share_rate(crosswalk)

    player_sources = {
        "career_touches": load_career_touches(crosswalk),
        "snap": snap_share_df[["player_name", "snap_share"]],
        "depth_chart": compute_depth_chart_competition(target_share_df),
        "target_share": target_share_df[
            ["player_name", "target_share_rate", "opportunity_share_rate", "games_recent",
             "target_share_data_quality_flag"]
        ],
        "redzone": load_redzone(abbrev_crosswalk),
        "td_rate": load_td_rate(crosswalk),
        "value_vol": load_value_and_volatility(master),
    }

    board = master[
        ["player_name", "position", "team", "adp", "adp_rank", "age", "injury_status", "injury_risk", "_key"]
    ].copy()
    # Normalize team codes BEFORE the team-level joins below (not after) --
    # otherwise LAR players silently get NaN offense/schedule data instead
    # of matching the Rams' real "LA" rows. Preserves the original team
    # code as displayed elsewhere in the app (Home.py etc. all use "LAR").
    board["_join_team"] = board["team"].replace(TEAM_CODE_ALIASES)

    # player_id resolution -- needed by the v4 injury computation below,
    # which must join draft_capital_player_seasons.csv by id, not name (see
    # module docstring). Same crosswalk already used by every other
    # player-level join here, so this can't drift out of sync with them.
    id_lookup = crosswalk[["player_id", "player_name"]].copy()
    id_lookup["_key"] = _key(id_lookup["player_name"])
    id_lookup = id_lookup.drop_duplicates("_key", keep="first")
    board = board.merge(id_lookup[["_key", "player_id"]], on="_key", how="left")

    for name, df in player_sources.items():
        if df.empty or "player_name" not in df.columns:
            continue
        df = df.copy()
        df["_key"] = _key(df["player_name"])
        df = df.drop(columns=["player_name"]).drop_duplicates("_key")
        board = board.merge(df, on="_key", how="left", suffixes=("", f"_{name}"))

    # Position-relative percentiles for target_share_rate/snap_share -- must
    # run against the CURRENT draft pool (board), after the merges above,
    # not against the raw source files (which include irrelevant players).
    board = pd.concat([board, build_position_percentiles(board)], axis=1)

    # Team-level joins -- one row per team, broadcast to every player on it.
    # Joined on _join_team (normalized), not team (as-displayed) -- see
    # TEAM_CODE_ALIASES above.
    offense = load_offense_environment().rename(columns={"team": "_join_team"})
    schedule = load_schedule_risk().rename(columns={"team": "_join_team"})
    board = board.merge(offense, on="_join_team", how="left")
    board = board.merge(schedule, on="_join_team", how="left")
    board = board.drop(columns=["_join_team"])

    # --- Four category scores, one formula call per player/row ---
    game_log_by_id = load_game_log_by_season(crosswalk)
    draft_capital_by_id = load_draft_capital_by_id()
    injury_cols = board.apply(
        lambda r: compute_injury_v4(r, game_log_by_id, draft_capital_by_id), axis=1
    )
    board = pd.concat([board, injury_cols], axis=1)
    board["role_usage_td_score"] = board.apply(role_usage_td_score, axis=1)
    board = apply_usage_risk_overrides(board)
    board["offense_environment_score"] = board.apply(
        lambda r: offense_environment_score(r) if pd.notna(r.get("epa_rank")) else np.nan, axis=1
    )
    board["schedule_weather_venue_score"] = board.apply(
        lambda r: schedule_weather_venue_score(r) if pd.notna(r.get("bye_week")) else np.nan, axis=1
    )

    # --- Composite risk_index, via the same player_from_variable_row() /
    # risk_index() risk_cli.py uses, so the two can never silently drift ---
    weights = load_weights()
    risk_index_values = []
    for _, row in board.iterrows():
        player = player_from_variable_row(row)
        risk_index_values.append(risk_index(player, weights) if player is not None else np.nan)
    board["risk_index"] = risk_index_values

    return board.drop(columns=["_key"])


def main() -> int:
    board = build_risk_variables()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUTPUT_CSV, index=False)

    print(f"[write] {OUTPUT_CSV}: {len(board)} row(s)")
    for category in RiskCategory:
        column = f"{category.value}_score"
        coverage = board[column].notna().sum() if column in board.columns else 0
        print(f"  {category.value:24s} coverage={coverage}/{len(board)}")
    for extra in ("value_signal_market", "volatility_diagnostic"):
        coverage = board[extra].notna().sum() if extra in board.columns else 0
        print(f"  {extra:24s} coverage={coverage}/{len(board)} (not in composite)")
    print(f"  {'risk_index':24s} coverage={board['risk_index'].notna().sum()}/{len(board)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
