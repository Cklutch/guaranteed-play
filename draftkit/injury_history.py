"""Injury risk scoring -- v4 (season-range fix).

Ports the user's "Injury Risk Scoring -- v4 Consolidated Fix Spec" close to
verbatim. Kept as its own module (not folded into build_risk_variables.py)
because it's substantial and independently testable -- matching how
risk_scoring.py/risk_constraints.py are already split by concern.

Why v4 exists (both confirmed against real data before this was written,
not taken on faith from the spec's illustrative examples):

  - Erick All case: a fully missed season (zero games, no game-log row at
    all) was invisible to v1/v2's scoring because it never became a dict
    key -- v1/v2 built games-missed by counting weekly "Out" injury-report
    rows, and players on season-ending IR drop out of that report
    entirely. Confirmed: his 2025 has no stats_player_reg_by_season row at
    all. Fix: explicitly construct the season range (rookie year through
    last season) rather than trusting whatever rows exist in the source.

  - McCaffrey case: the same root cause, manifesting as undercounting
    instead of total invisibility -- his 2024 real games played was 4 (13
    missed), but v1/v2's Out-report-counting only found 1. A single/dual
    -season lookback also let his 2024 injury vanish once 2025 came back
    healthy. Fix: decay-weighted lookback across the player's FULL season
    history, not just the last 1-2 seasons.

No real per-season injury-severity feed exists anywhere in this repo
(confirmed in the v2 build) -- injury_report_source is always None here,
so severity is always inferred from games-missed magnitude via
infer_severity_for_missing_season(), consistently with the honesty already
established for this whole risk scorecard feature.

v5 constant/threshold updates (risk_engine_v5_master_fix_spec.pdf), same
season-range architecture as v4, just recalibrated against three more real
cases found in live testing:
  - severity thresholds lowered (season_ending >=12, moderate >=3, was
    15/6) -- Brock Bowers' 5 missed games was landing as "minor" under the
    old >=6 threshold, understating a real moderate absence.
  - DECAY_RATE raised 0.55 -> 0.70 (slower decay, older seasons retain
    more relative weight) -- Jonathan Taylor's real multi-year injury
    pattern was getting erased too fast by one recent healthy season.
  - single-season confidence dampening added -- a player with exactly one
    season of history (rookie_season == current_season - 1) had a single
    rookie-year injury (Omarion Hampton) mechanically maxing out the
    score; one data point isn't a confirmed pattern, so the raw score is
    blended toward a neutral 2.5 in that case. Initially a 50/50 blend
    (confidence=0.5), confirmed firing exactly as designed, but a true
    50/50 of an undamped ceiling (5.0) and neutral (2.5) still lands at
    3.75 -- too high to read as "moderate" for a single data point. Lowered
    to confidence=0.3 (post-v5 fix) to pull harder toward neutral.

Phase 2 (plan_risk_index_reweight_dynamic.pdf): acute/chronic split +
continuous calendar decay, replacing v5's single blended score and its
per-SEASON stepped decay (years_back = current_season - season - 1, an
integer -- meant a real, recent, healing injury got ZERO decay until the
NEXT season's data landed, months after real recovery evidence like
practice participation or a player's own "feeling fantastic" comments
would have accumulated). Confirmed no real per-season injury-severity feed
exists anywhere in this repo (injury_report_source is always None, see
above) -- so decay is computed from real, already-known data (today's real
date, each season's real approximate midpoint), not a fabricated
recovery-evidence stream.

Two intermediate scores, not one:
  - _acute_score(): the same v4/v5 formula, decay now continuous
    (DECAY_RATE=0.70 per real elapsed year from each season's real
    midpoint, not a season-boundary step) -- fast-fading, meant to track
    CURRENT/recent status.
  - _chronic_score(): the career-level recurring-injury pattern (the same
    "chronic_seasons > 2" multiplier v5 already had), computed decay-free
    against the full history (a real multi-year pattern is evidence about
    the PLAYER, not about any one season's age), then decayed SLOWLY
    (CHRONIC_DECAY_RATE=0.95/year, ~13.5-year half-life) from the most
    recent qualifying season -- real, but not eternal. Stress-tested
    against a real contrasting case (injury-heavy seasons 6+ years back,
    fully healthy every season since): frozen-forever chronic scoring was
    the first draft, rejected once this case showed it never lets a
    genuinely long-resolved pattern soften.
Combined via max(acute, chronic) -- same "either signal independently
justifies concern" pattern _blend_current_injury_status() already uses one
layer up in build_risk_variables.py.
"""

from __future__ import annotations

import datetime

SEVERITY_WEIGHT = {"minor": 1.0, "moderate": 1.6, "season_ending": 2.8}
DECAY_RATE = 0.70
FULL_SEASON_GAMES = 17

# Real injury dates aren't tracked anywhere in this repo -- approximates
# each season's real injury activity as centered on that season's rough
# real midpoint. A reasoned default, not asserted as precise; the whole
# point is real elapsed calendar time from SOME real reference point,
# rather than the old integer season-boundary step.
SEASON_MIDPOINT_MONTH = 10
SEASON_MIDPOINT_DAY = 15

# Chronic-recurrence decay -- much slower than DECAY_RATE (real ~13.5-year
# half-life vs. acute's real ~2-year half-life), reasoned default pending
# validation (see draftkit/tests/test_injury_history.py's long-resolved
# chronic case). A real multi-injury pattern from someone's early-20s
# shouldn't cap their score the same way a decade later.
CHRONIC_DECAY_RATE = 0.95
# A demonstrated pattern needs MORE than 2 real qualifying seasons (same
# threshold v5's original inline chronic multiplier used) -- 2 isn't yet a
# pattern, could be unrelated one-off injuries.
CHRONIC_PATTERN_MIN_SEASONS = 2
# Lowered from 0.5 (post-v5 fix): confirmed the 50/50 blend was firing
# exactly as designed (len(games_missed_by_season)<=1 check correct,
# confidence really applied at 0.5), but a true 50/50 blend of an undamped
# ceiling (5.0) and neutral (2.5) still lands at 3.75 -- too high for a
# single rookie-year injury (real case: Omarion Hampton) to read as merely
# "moderate" risk rather than "still quite risky." 0.3 pulls harder toward
# neutral while still letting a genuinely severe rookie injury register as
# above-neutral, not floor.
SINGLE_SEASON_CONFIDENCE = 0.3
NEUTRAL_SCORE = 2.5


def build_games_missed_by_season(
    rookie_season: int,
    current_season: int,
    game_log_source: dict,   # {season: games_played} -- only seasons WITH a stat row
    full_season_games: int = FULL_SEASON_GAMES,
) -> dict[int, int]:
    games_missed = {}
    for season in range(rookie_season, current_season):
        if season in game_log_source:
            played = game_log_source[season]
            games_missed[season] = max(0, full_season_games - played)
        else:
            # No row = zero games that season = fully missed, not "no data."
            games_missed[season] = full_season_games
    return games_missed


def infer_severity_for_missing_season(games_missed: int) -> str:
    if games_missed >= 12:
        return "season_ending"
    if games_missed >= 3:
        return "moderate"
    return "minor"


def assemble_injury_inputs(
    rookie_season: int,
    current_season: int,
    game_log_source: dict,
    injury_report_source: dict | None = None,
) -> tuple[dict[int, int], dict[int, str]]:
    """Single entry point feeding the scorer -- no other code path should
    populate games_missed_by_season directly from raw source data."""
    games_missed = build_games_missed_by_season(rookie_season, current_season, game_log_source)
    severity = {}
    for season, missed in games_missed.items():
        reported = (injury_report_source or {}).get(season)
        severity[season] = reported or infer_severity_for_missing_season(missed)
    return games_missed, severity


def _season_reference_date(season: int) -> datetime.date:
    """A season's approximate real midpoint -- see SEASON_MIDPOINT_MONTH's
    docstring for why this is a reasoned default, not a precise date."""
    return datetime.date(season, SEASON_MIDPOINT_MONTH, SEASON_MIDPOINT_DAY)


def _elapsed_years(season: int, today: datetime.date) -> float:
    """Real elapsed calendar time from a season's real approximate midpoint
    to `today`, in years. Floored at 0.0 -- a season that hasn't happened
    yet (or is still in progress relative to `today`) gets zero decay, not
    negative."""
    delta_days = (today - _season_reference_date(season)).days
    return max(0.0, delta_days / 365.25)


def _decayed_severity_average(
    games_missed_by_season: dict[int, int],
    injury_severity_by_season: dict[int, str],
    current_season: int,
    decay_rate: float,
    today: datetime.date,
) -> float | None:
    """Real, continuous-decay-weighted average games-missed*severity --
    shared by _acute_score() (DECAY_RATE) and available for any future
    caller needing the same shape at a different rate. Returns None (not
    1.0) when there's no season history at all, so callers can distinguish
    "no data" from "computed a real floor value"."""
    weighted_total = 0.0
    total_weight = 0.0
    for season, games_missed in games_missed_by_season.items():
        if current_season - season - 1 < 0:
            continue  # a season that hasn't happened relative to current_season
        decay = decay_rate ** _elapsed_years(season, today)
        severity = SEVERITY_WEIGHT.get(injury_severity_by_season.get(season, "minor"), 1.0)
        weighted_total += games_missed * severity * decay
        total_weight += decay
    if total_weight == 0:
        return None
    return weighted_total / total_weight


def _acute_score(
    games_missed_by_season: dict[int, int],
    injury_severity_by_season: dict[int, str],
    age: int,
    current_season: int,
    today: datetime.date,
) -> float:
    """CURRENT/recent injury status -- real, continuous per-year decay
    (DECAY_RATE), no chronic-recurrence multiplier (that's _chronic_score()'s
    job now, combined via max() in injury_risk_score_v4_detailed()). Same
    single-season dampening and recency-floor logic v4/v5 already had,
    unchanged in shape, just continuous instead of stepped."""
    normalized = _decayed_severity_average(
        games_missed_by_season, injury_severity_by_season, current_season, DECAY_RATE, today,
    )
    if normalized is None:
        return 1.0  # genuinely no season history -- correct floor for true rookies

    age_factor = 1.2 if age >= 28 else 1.0
    raw = normalized * age_factor
    raw_score = min(5.0, max(1.0, round(raw / 2.5, 1)))

    # Single-season dampening -- one data point isn't a confirmed pattern.
    # len() <= 1 means the constructed season range only ever had one
    # season in it (rookie_season == current_season - 1), not "one injury"
    # specifically -- a veteran with 5 healthy-or-hurt seasons on record
    # never hits this branch regardless of how many of those seasons had
    # an injury in them.
    #
    # Only fires when raw_score is already ABOVE floor (post-v5 fix, real
    # case: Ashton Jeanty -- 0 games missed in his only NFL season,
    # correctly floors at 1.0, but was getting blended toward a "neutral"
    # 2.5 anyway just for having one season of history). The dampening
    # exists to hedge an INFLATED score built from a single data point --
    # a clean record isn't inflated, it's just clean, and needs no hedge
    # regardless of sample size.
    if len(games_missed_by_season) <= 1 and raw_score > 1.0:
        confidence = SINGLE_SEASON_CONFIDENCE
        return round(raw_score * confidence + NEUTRAL_SCORE * (1 - confidence), 1)

    # Recency floor (post-v5 fix, real case: Drake London -- missed 5 real
    # games last season, moderate severity, but 3 prior mostly-clean
    # seasons diluted the decay-weighted average down to a floor-adjacent
    # 7.5/100). A real, recent absence shouldn't be allowed to score BELOW
    # what it would on its own merits just because a longer clean track
    # record adds more decay-weight to the denominator -- more history is
    # real evidence of durability and should keep tempering things, but
    # not to the point of erasing a genuine, current injury. Computed the
    # same way a rookie's sole season would be (single-season dampening
    # included), so the "one recent data point" caution still applies --
    # this only raises the FLOOR, never lowers what the full career
    # average already produced.
    #
    # Guarded to len > 1 -- for a true single-season player, the branch
    # above already handles their "recent-only" case directly, and calling
    # back into this same function with that same single-season dict here
    # would recurse forever whenever raw_score is exactly 1.0 (the dampening
    # branch's `> 1.0` guard doesn't trigger, so execution falls through
    # to here and re-derives the identical single-key dict every time).
    most_recent_season = current_season - 1
    if len(games_missed_by_season) > 1 and most_recent_season in games_missed_by_season:
        recent_only_score = _acute_score(
            {most_recent_season: games_missed_by_season[most_recent_season]},
            {most_recent_season: injury_severity_by_season.get(most_recent_season, "minor")},
            age=age, current_season=current_season, today=today,
        )
        raw_score = max(raw_score, recent_only_score)
    return raw_score


def _chronic_score(
    games_missed_by_season: dict[int, int],
    injury_severity_by_season: dict[int, str],
    age: int,
    current_season: int,
    today: datetime.date,
) -> float:
    """Career-level recurring-injury pattern -- real case this exists for:
    Tee Higgins (missed 2/3/3/5/5/2 games across all 6 of his seasons).
    Decay-weighted AVERAGING alone (what _acute_score() does) doesn't
    distinguish "moderate every single year" from "one moderate year" with
    the same average severity -- a demonstrated multi-year pattern is a
    stronger predictor of future missed time than an isolated incident.

    Threshold is games_missed >= 2 per qualifying season (CHRONIC_PATTERN_
    MIN_SEASONS), NOT the moderate-severity cutoff (>=3) -- confirmed too
    narrow on a second real case (Nico Collins: 3/7/2/5/2 missed across 5
    seasons, missed SOME time in literally every season, "never played a
    full season"). Counting only moderate-or-worse seasons missed 2 of his
    5 real recurring seasons (the "2"s) entirely -- this is about
    RECURRENCE frequency, a different question from decay-weighting's own
    severity threshold, so it gets its own, looser bar.

    Computed DECAY-FREE against the full history first (a real multi-year
    pattern is evidence about the PLAYER, not about any one season's age),
    then the whole result is decayed SLOWLY (CHRONIC_DECAY_RATE) from the
    most recent qualifying season -- see that constant's own docstring for
    why this isn't frozen forever."""
    chronic_seasons = [s for s, m in games_missed_by_season.items() if m >= CHRONIC_PATTERN_MIN_SEASONS]
    if len(chronic_seasons) <= CHRONIC_PATTERN_MIN_SEASONS:
        return 1.0  # not yet a real demonstrated pattern

    weighted_total = sum(
        m * SEVERITY_WEIGHT.get(injury_severity_by_season.get(s, "minor"), 1.0)
        for s, m in games_missed_by_season.items()
    )
    normalized = weighted_total / len(games_missed_by_season)
    age_factor = 1.2 if age >= 28 else 1.0
    raw = normalized * age_factor
    raw_score = min(5.0, max(1.0, round(raw / 2.5, 1)))

    # +10% per recurring season beyond the first two -- multiplicative, not
    # additive, so it scales with how elevated the base score already is
    # rather than a flat bump.
    chronic_multiplier = 1.0 + 0.1 * (len(chronic_seasons) - CHRONIC_PATTERN_MIN_SEASONS)
    base_score = min(5.0, round(raw_score * chronic_multiplier, 1))

    most_recent_chronic_season = max(chronic_seasons)
    chronic_decay = CHRONIC_DECAY_RATE ** _elapsed_years(most_recent_chronic_season, today)
    return round(1.0 + (base_score - 1.0) * chronic_decay, 1)


def injury_risk_score_v4_detailed(
    games_missed_by_season: dict[int, int],
    injury_severity_by_season: dict[int, str],
    age: int,
    current_season: int,
    today: datetime.date | None = None,
) -> tuple[float, float, float]:
    """Returns (acute_score, chronic_score, combined_score) -- both
    intermediate numbers are real and independently meaningful (see module
    docstring's Phase 2 section), not just an implementation detail of the
    combined score. `today` is injectable for deterministic testing;
    defaults to the real current date."""
    if today is None:
        today = datetime.date.today()
    acute = _acute_score(games_missed_by_season, injury_severity_by_season, age, current_season, today)
    chronic = _chronic_score(games_missed_by_season, injury_severity_by_season, age, current_season, today)
    return acute, chronic, max(acute, chronic)


def injury_risk_score_v4(
    games_missed_by_season: dict[int, int],
    injury_severity_by_season: dict[int, str],
    age: int,
    current_season: int,
    today: datetime.date | None = None,
) -> float:
    return injury_risk_score_v4_detailed(
        games_missed_by_season, injury_severity_by_season, age, current_season, today,
    )[2]


__all__ = [
    "SEVERITY_WEIGHT",
    "DECAY_RATE",
    "CHRONIC_DECAY_RATE",
    "CHRONIC_PATTERN_MIN_SEASONS",
    "FULL_SEASON_GAMES",
    "SEASON_MIDPOINT_MONTH",
    "SEASON_MIDPOINT_DAY",
    "build_games_missed_by_season",
    "infer_severity_for_missing_season",
    "assemble_injury_inputs",
    "injury_risk_score_v4",
    "injury_risk_score_v4_detailed",
]
