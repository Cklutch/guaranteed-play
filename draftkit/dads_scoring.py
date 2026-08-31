"""Dad's-league scoring -- an alternate fantasy-point projection used to
rework the board's model score (see Home.py's "Scoring model" selector).

Source: "Dad's League Settings.pdf". How dad's format differs from the
board's default (12-team Half-PPR):

  * NO PPR (receptions score nothing) and NO per-yard points. ALL yardage
    is scored through position-specific PER-GAME tiered buckets -- e.g. an
    RB/WR/TE game of 100-125 yards is worth 6, but 76-99 is only 3 (a
    deliberate cliff at 100). QBs have their own, higher table.
  * Touchdowns: passing 4, rushing 5, receiving 4, plus length bonuses.
  * Interception -1, fumble lost -2.
  * Different roster/scarcity: dad's league starts 1QB/2RB/2WR/1TE with NO
    flex (vs the user's Half-PPR 1QB/2RB/3WR/1TE/2FLEX). Applied to dad's
    VOR baselines ONLY (see DADS_STARTERS / _dads_replacement_baselines);
    the standard board keeps the user's own roster.

The board's scoring engine (draftkit/draft_analysis.py) is projection-
driven: final_score = VOR + within-position projection percentile + market
- risk. So swapping in a dad's-scoring projection (and its VOR) and reusing
the SAME engine function (calculate_base_value_score) reworks the model
score exactly as the user asked, leaving the market and risk terms -- which
don't depend on the projection -- untouched.

MODELING NOTES / assumptions (all display-only, like the board's other
re-scores; never touches the real Half-PPR pipeline):

  * We only carry SEASON stat totals (data/raw/winwithodds_season_
    projections.csv), but the yardage buckets and TD-length bonuses are
    per-game / per-play. Per the user's decisions:
      - Yardage (GAME-BY-GAME model, 2026-08-24): rather than bucketing the
        season AVERAGE, we treat each game's COMBINED pass+rush+rec yards
        (user chose combined) as a draw from a Gamma distribution with mean
        = season yards / GAMES_PER_SEASON and a position-specific
        coefficient of variation (POSITION_YARD_CV). Expected season
        yardage points = GAMES_PER_SEASON x the expectation of the tier
        step-function under that distribution. This correctly rewards the
        100-yd cliff: a 95-avg RB who sometimes clears 100 earns some 6-pt
        games, which averaging would miss. The CVs are documented estimates
        of NFL week-to-week variance (no local per-game data to calibrate;
        tune POSITION_YARD_CV or recalibrate from weekly logs later).
      - TD length bonus: a small EXPECTED value per TD (we have no per-TD
        length). Constants below; set to 0 to disable.
  * Fumbles: winwithodds 'Fumbles' is treated as total fumbles;
    FUMBLE_LOST_RATE approximates the share actually lost.
"""

import re

import pandas as pd

WINWITHODDS_PATH = "data/raw/winwithodds_season_projections.csv"
# JuiceBoxOne's 2026 ADP, blended across FantasyPros/ESPN/Sleeper/Yahoo --
# the STANDARD-scoring cut specifically (2026-08-30, user-supplied). Dad's
# league has no PPR (see module docstring), so this is the correct market
# read for it: a half-PPR ADP source systematically over-ranks high-target
# possession receivers relative to how a no-PPR room actually drafts them.
# There is a SEPARATE half-PPR cut of the same source
# (juicebox_2026_adp_sources.csv) already feeding the main board's own
# `adp` column -- the two must stay separate, not merged, or the main
# board's market signal silently drifts toward a scoring format it isn't.
DADS_ADP_PATH = "data/raw/juicebox_2026_adp_sources_standard.csv"

GAMES_PER_SEASON = 17

# Flat TD point values (dad's PDF).
PASS_TD_PTS = 4.0
RUSH_TD_PTS = 5.0
REC_TD_PTS = 4.0
INT_PTS = -1.0
FUMBLE_LOST_PTS = -2.0
# Share of projected fumbles assumed lost (recovery is ~50/50 in the NFL).
FUMBLE_LOST_RATE = 0.5

# Expected per-TD length bonus -- an APPROXIMATION (no per-TD length data).
# Rough expected values given dad's brackets and typical NFL TD-length
# distributions: rushing TDs skew short (many goal-line, no bonus until
# 6+ yds), receiving/passing TDs average ~12 yds. Small relative to the
# flat TD points above. Set to 0.0 to disable length bonuses entirely.
#
# These are FLAT position-wide constants -- every RB gets the same 0.6,
# whether he's a 60-yard-breakaway back or a 1-yard-plunge specialist. The
# per-player multiplier below (2026-08-30, user-directed: "players that
# consistently score... high numbers of long yardage touchdowns are
# premium") tilts each player's bonus around these baselines using a real,
# data-grounded proxy for TD length, rather than replacing them outright.
PASS_TD_LEN_BONUS_EV = 0.9
RUSH_TD_LEN_BONUS_EV = 0.6
REC_TD_LEN_BONUS_EV = 1.5

# Per-player TD-length tilt. We don't have real per-play TD-distance data
# locally (no nflverse play-by-play pull exists in this repo yet -- see
# module docstring), so this uses an honest PROXY instead of a measurement:
# projected yards-per-touchdown in a category. A player who needs fewer
# total yards per score, on average, is scoring more of his TDs short/
# goal-line; a player who needs more yards per score is getting more of his
# production from longer plays. Checked against 2026 winwithodds
# projections: real, ~2x spread within a position (RB rush 88-183 yds/TD,
# WR rec 82-222, TE rec 77-184, QB pass and QB rush also checked) -- not
# noise, and not just a volume artifact once the shrinkage below controls
# for sample size.
#
# Below TD_LENGTH_MIN_SAMPLE projected TDs in a category, a player's own
# ratio is pure noise (one 60-yard TD on an otherwise 1-TD season is not
# "explosive," it's a small sample) -- multiplier is 1.0, identical to the
# old flat-EV behavior. From there up to TD_LENGTH_FULL_CREDIT_TDS, the
# ratio is shrunk toward the position average by linear interpolation; at
# or above it, the player's own ratio is used at full weight. MULT_MIN/MAX
# bound the result so this stays a tilt, not a new dominant term -- same
# "small relative to the flat TD points" design intent as the EVs above.
TD_LENGTH_MIN_SAMPLE = 3
TD_LENGTH_FULL_CREDIT_TDS = 6
TD_LENGTH_MULT_MIN = 0.55
TD_LENGTH_MULT_MAX = 1.70

# (position, td_type) combos with enough real players at TD_LENGTH_MIN_SAMPLE+
# projected TDs to trust a position-average baseline (checked against 2026
# projections -- RB/WR/TE receiving-by-RB and other combos came back with too
# few qualifying players and are deliberately left out, falling back to the
# flat EV untouched rather than averaging over a handful of players).
TD_LENGTH_COMBOS = [("RB", "rush"), ("WR", "rec"), ("TE", "rec"), ("QB", "pass"), ("QB", "rush")]
# Minimum number of qualifying PLAYERS (not TDs) before a position/type
# combo's average is trusted as a baseline at all.
TD_LENGTH_MIN_PLAYERS = 5

# Per-game yardage tiers as half-open bands (lower, upper, points) for a
# single game's COMBINED pass+rush+rec yards. Continuous form of dad's
# inclusive ranges (e.g. skill "100-125" -> [100, 126); QB "226-249" ->
# [226, 250)). Yards below the first band score 0; the top band is open.
INF = float("inf")
SKILL_YARD_BANDS = [
    (40, 60, 1), (60, 76, 2), (76, 100, 3), (100, 126, 6),
    (126, 151, 7), (151, 176, 8), (176, 201, 9), (201, INF, 10),
]
QB_YARD_BANDS = [
    (151, 176, 1), (176, 201, 2), (201, 226, 3), (226, 250, 4),
    (250, 301, 7), (301, 351, 8), (351, INF, 9),
]
POSITION_YARD_BANDS = {
    "QB": QB_YARD_BANDS,
    "RB": SKILL_YARD_BANDS,
    "WR": SKILL_YARD_BANDS,
    "TE": SKILL_YARD_BANDS,
}

# Coefficient of variation (std / mean) of a player's per-GAME combined
# yards, by position -- documented estimates of NFL week-to-week variance
# (no local per-game data to calibrate against). QBs are steady week to
# week; WR/TE are the most boom-or-bust. These drive the game-by-game model
# (see module docstring); raise a position's CV to spread its per-game
# outcomes wider, which -- given the cliffs -- generally lifts its yardage
# points. Tune here or recalibrate from real weekly logs later.
POSITION_YARD_CV = {
    "QB": 0.30,
    "RB": 0.52,
    "WR": 0.66,
    "TE": 0.70,
}

# Dad's league roster -- drives the replacement ranks (and thus VOR /
# positional scarcity) for the Dad's League scoring model ONLY. The
# standard board keeps its own roster (the user's real Half-PPR league:
# 12-team, QB1/RB2/WR3/TE1/FLEX2). Dad's league (from the settings PDF):
# 12-team, QB1/RB2/WR2/TE1, K1/DST1, and NO flex. With no flex the
# replacement rank is simply teams x starters -> QB12 / RB24 / WR24 / TE12
# (shallower at RB/WR/TE than the user's league, which compresses their
# cross-position VOR on top of the no-PPR scoring change).
DADS_LEAGUE_SIZE = 12
DADS_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
DADS_FLEX = 0  # no flex spot in dad's lineup


def _dads_replacement_rank(position):
    """Replacement rank for a position under dad's roster. Mirrors
    draft_analysis._position_replacement_rank's formula (teams x (starters +
    flex_share)) but with dad's fixed roster rather than session state, so
    it never disturbs the standard board's own replacement ranks."""
    pos = str(position).upper()
    share = 0.0
    if DADS_FLEX:
        from draftkit.draft_analysis import POSITION_FLEX_SHARES
        share = DADS_FLEX * POSITION_FLEX_SHARES.get(pos, 0.0)
    return max(int(round(DADS_LEAGUE_SIZE * (DADS_STARTERS.get(pos, 0) + share))), 1)


def _dads_replacement_baselines(df, proj_col="dads_projection_points"):
    """Per-position dad's-projection value at dad's replacement rank -- the
    baseline VOR is measured from. Same shape as
    draft_analysis.build_position_replacement_baselines, but keyed on dad's
    ranks. Players without a dad's projection (NaN) sort last, so the
    baseline is drawn from real-projection players."""
    baselines = {}
    for position in DADS_STARTERS:
        pos_df = df[df["position"].astype(str).str.upper() == position].sort_values(
            proj_col, ascending=False, na_position="last"
        )
        if pos_df.empty:
            baselines[position] = 0.0
            continue
        idx = min(_dads_replacement_rank(position) - 1, len(pos_df) - 1)
        value = pos_df.iloc[idx][proj_col]
        baselines[position] = float(value) if pd.notna(value) else 0.0
    return baselines


# winwithodds spells a handful of players differently from the board.
# Keyed by normalized board name -> normalized winwithodds name.
_NAME_ALIASES = {
    "hollywood brown": "marquise brown",
}


def _norm(name):
    """Normalize a player name for cross-source matching: lowercase, drop
    generational suffixes and punctuation, collapse whitespace."""
    n = str(name).lower().strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    n = re.sub(r"[.'’-]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _f(value):
    """Coerce to float, treating missing/blank as 0.0."""
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _band_points_at(bands, yards):
    """Tier points for a single game's yards under a band table (step fn)."""
    for lo, hi, pts in bands:
        if lo <= yards < hi:
            return float(pts)
    return 0.0


def _expected_season_yardage_points(position, total_yds):
    """Game-by-game expected season yardage points.

    Models each game's combined yards as Gamma(mean = total/GAMES, CV =
    POSITION_YARD_CV[pos]) and returns GAMES x the expectation of the tier
    step-function under that distribution -- i.e. it integrates the buckets
    over the per-game distribution instead of bucketing the season average.
    Falls back to bucketing the per-game mean if the position is unknown,
    the mean is non-positive, or scipy is unavailable.
    """
    pos = str(position).upper()
    bands = POSITION_YARD_BANDS.get(pos)
    if not bands or total_yds is None or total_yds <= 0:
        return 0.0

    mean_pg = float(total_yds) / GAMES_PER_SEASON
    cv = POSITION_YARD_CV.get(pos)

    if cv and cv > 0:
        try:
            from scipy.stats import gamma

            # Gamma with mean m, CV c: shape k = 1/c^2, scale = m*c^2.
            shape = 1.0 / (cv * cv)
            scale = mean_pg * cv * cv
            expected_pg = 0.0
            for lo, hi, pts in bands:
                if pts == 0:
                    continue
                prob = gamma.cdf(hi, a=shape, scale=scale) - gamma.cdf(lo, a=shape, scale=scale)
                expected_pg += pts * prob
            return round(expected_pg * GAMES_PER_SEASON, 4)
        except Exception:
            pass  # fall through to average-based

    # Fallback: bucket the per-game average (no variance).
    return round(_band_points_at(bands, mean_pg) * GAMES_PER_SEASON, 4)


def _td_length_baselines(raw, col):
    """{(position, td_type): avg yards-per-TD} for TD_LENGTH_COMBOS -- a
    population statistic, computed ONCE per build_dads_projections_df() call
    (not per player). Combos without TD_LENGTH_MIN_PLAYERS qualifying players
    are simply absent from the returned dict, and _td_length_multiplier()
    treats a missing baseline as "no adjustment" (multiplier 1.0)."""
    yard_col = {"pass": col["pass_yds"], "rush": col["rush_yds"], "rec": col["rec_yds"]}
    td_col = {"pass": col["pass_td"], "rush": col["rush_td"], "rec": col["rec_td"]}
    baselines = {}
    for pos, kind in TD_LENGTH_COMBOS:
        sub = raw[raw[col["pos"]] == pos]
        tds = pd.to_numeric(sub[td_col[kind]], errors="coerce").fillna(0)
        yds = pd.to_numeric(sub[yard_col[kind]], errors="coerce").fillna(0)
        qualifies = tds >= TD_LENGTH_MIN_SAMPLE
        if qualifies.sum() < TD_LENGTH_MIN_PLAYERS:
            continue
        baselines[(pos, kind)] = float((yds[qualifies] / tds[qualifies]).mean())
    return baselines


def _td_length_multiplier(player_yds, player_td, baseline_ratio):
    """Shrinkage-adjusted length-bonus multiplier for one TD type. See the
    TD_LENGTH_* constants' comments for the reasoning; this is just the
    linear-interpolation implementation of it."""
    if baseline_ratio is None or baseline_ratio <= 0 or player_td < TD_LENGTH_MIN_SAMPLE:
        return 1.0
    weight = min(1.0, player_td / TD_LENGTH_FULL_CREDIT_TDS)
    effective_ratio = weight * (player_yds / player_td) + (1 - weight) * baseline_ratio
    mult = effective_ratio / baseline_ratio
    return max(TD_LENGTH_MULT_MIN, min(TD_LENGTH_MULT_MAX, mult))


def dads_points_from_stats(position, pass_yds=0, rush_yds=0, rec_yds=0,
                           pass_td=0, rush_td=0, rec_td=0, ints=0, fumbles=0,
                           length_baselines=None):
    """Full-season dad's-league fantasy points from season stat totals.

    See module docstring for the per-game / per-play approximations.
    `length_baselines` (from _td_length_baselines(), or None) makes the TD-
    length bonus per-player instead of a flat position constant -- omitting
    it reproduces the exact old flat-EV behavior, so any other caller of
    this function is unaffected.
    """
    pass_yds, rush_yds, rec_yds = _f(pass_yds), _f(rush_yds), _f(rec_yds)
    pass_td, rush_td, rec_td = _f(pass_td), _f(rush_td), _f(rec_td)

    pos = str(position).upper()
    lb = length_baselines or {}
    pass_mult = _td_length_multiplier(pass_yds, pass_td, lb.get((pos, "pass")))
    rush_mult = _td_length_multiplier(rush_yds, rush_td, lb.get((pos, "rush")))
    rec_mult = _td_length_multiplier(rec_yds, rec_td, lb.get((pos, "rec")))

    td_pts = pass_td * PASS_TD_PTS + rush_td * RUSH_TD_PTS + rec_td * REC_TD_PTS
    td_len_bonus = (
        pass_td * PASS_TD_LEN_BONUS_EV * pass_mult
        + rush_td * RUSH_TD_LEN_BONUS_EV * rush_mult
        + rec_td * REC_TD_LEN_BONUS_EV * rec_mult
    )
    turnover_pts = _f(ints) * INT_PTS + _f(fumbles) * FUMBLE_LOST_RATE * FUMBLE_LOST_PTS

    total_yds = pass_yds + rush_yds + rec_yds
    yard_pts = _expected_season_yardage_points(position, total_yds)

    return round(td_pts + td_len_bonus + turnover_pts + yard_pts, 2)


# Manual news-driven corrections for Dad's League specifically (2026-08-30).
#
# Real gap found live: the standard board has a whole correction pipeline
# (model_projections_v1.csv, MODEL_PROJECTION_CORRECTIONS, the News Queue,
# the Rankings-page popover) -- winwithodds_season_projections.csv, which
# THIS module reads, has none of that. A player's news-driven correction on
# the standard board has zero effect here; Dad's League silently kept
# serving his full, un-adjusted raw projection. First two real instances:
# Josh Jacobs' Commissioner Exempt List placement and Jakobi Meyers' camp-
# depth-chart role compression were both live on the standard board and
# invisible here until this dict existed.
#
# Deliberately a SEPARATE dict from the standard board's, not a shared one:
# the two boards' raw baselines differ (dads uses winwithodds season
# totals; standard uses model_projections_v1.csv / master_players.csv), so
# a single shared pct would apply to two different bases and silently drift.
# Percent always applies to the RAW stat-derived projection (post TD-length
# tilt), matching the standard board's own "never stacks on the current
# adjusted number" rule. Keys are normalized names (see _norm()).
MANUAL_DADS_CORRECTIONS = {
    "josh jacobs": {
        "pct": -58.8,
        "note": (
            "Formally charged 2026-08-27 with two misdemeanors (battery, criminal "
            "damage to property), May 2026 domestic dispute. Placed on NFL "
            "Commissioner Exempt List 2026-08-30 -- barred from ALL team activity, "
            "practice and games, while paid (CBS Sports / NBC Sports / WTMJ). Will "
            "miss the start of the season (GB opens 9/13 at MIN). Initial court "
            "appearance not until 2026-11-17; exempt-list duration is at the "
            "Commissioner's discretion, not automatically tied to the case timeline "
            "-- no official return date exists to source from. Disclosed judgment "
            "call: assume 7 of 17 games played at his healthy per-game rate "
            "(missing through roughly his court date, real risk of missing more). "
            "Mirrors the standard board's correction (model_projections_v1.csv)."
        ),
    },
    "jakobi meyers": {
        "pct": -30.0,
        "note": (
            "Jaguars 2026 camp depth chart has Parker Washington at WR1, Meyers "
            "WR2 behind him in a room with Brian Thomas Jr. and rookie Travis "
            "Hunter; PFN's outlook reports Meyers is expected to 'come off the "
            "field for Brian Thomas Jr. and Travis Hunter far more often than "
            "Washington does,' despite his new 3yr/$60M extension -- real target-"
            "competition compression the raw stat line (extrapolated from his "
            "2025 post-trade pace, 42/483/3 over 9 games) doesn't capture. "
            "Disclosed judgment call, not season-ending, just a smaller role. "
            "Mirrors the standard board's correction (model_projections_v1.csv)."
        ),
    },
}


def _apply_manual_dads_corrections(df, proj_col="dads_projection_points"):
    """Apply MANUAL_DADS_CORRECTIONS on top of the raw stat-derived
    projection, in place of df[proj_col]. A player id appearing twice in
    the dict silently lets the later entry win (same footgun as the
    standard board's MODEL_PROJECTION_CORRECTIONS -- sequencing matters)."""
    for norm_name, correction in MANUAL_DADS_CORRECTIONS.items():
        idx = df.index[df["norm_name"] == norm_name]
        if len(idx) == 0:
            continue
        i = idx[0]
        raw_pts = df.at[i, proj_col]
        if pd.isna(raw_pts):
            continue
        df.at[i, proj_col] = round(float(raw_pts) * (1 + correction["pct"] / 100), 2)
    return df


def build_dads_projections_df(path=WINWITHODDS_PATH):
    """Load the stat-level season projections and return a frame of
    [norm_name, dads_projection_points] -- one row per projected player.
    Returns an empty frame if the source file is unavailable."""
    try:
        raw = pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return pd.DataFrame(columns=["norm_name", "dads_projection_points"])

    col = {
        "name": "Name", "pos": "Pos", "pass_yds": "Pass Yards", "pass_td": "Pass TDs",
        "ints": "Ints", "rush_yds": "Rush Yards", "rush_td": "Rush TDs",
        "rec_yds": "Rec Yards", "rec_td": "Rec TDs", "fumbles": "Fumbles",
    }
    if not {col["name"], col["pos"]}.issubset(raw.columns):
        return pd.DataFrame(columns=["norm_name", "dads_projection_points"])

    # Population statistic (yards-per-TD by position), computed ONCE against
    # the full projection set -- not per player. See TD_LENGTH_* comments.
    length_baselines = _td_length_baselines(raw, col)

    raw["dads_projection_points"] = raw.apply(
        lambda r: dads_points_from_stats(
            r.get(col["pos"]),
            pass_yds=r.get(col["pass_yds"]), rush_yds=r.get(col["rush_yds"]),
            rec_yds=r.get(col["rec_yds"]), pass_td=r.get(col["pass_td"]),
            rush_td=r.get(col["rush_td"]), rec_td=r.get(col["rec_td"]),
            ints=r.get(col["ints"]), fumbles=r.get(col["fumbles"]),
            length_baselines=length_baselines,
        ),
        axis=1,
    )
    raw["norm_name"] = raw[col["name"]].map(_norm)
    # Keep the highest projection if a name appears twice (dup guard).
    out = (
        raw[["norm_name", "dads_projection_points"]]
        .sort_values("dads_projection_points", ascending=False)
        .drop_duplicates("norm_name", keep="first")
        .reset_index(drop=True)
    )
    # Applied AFTER the dup guard (one row per player at this point) so a
    # correction can never land on the wrong duplicate row or get
    # double-applied to two rows for the same normalized name.
    out = _apply_manual_dads_corrections(out)
    return out


def load_dads_adp(path=DADS_ADP_PATH):
    """[norm_name, dads_adp, dads_adp_avg] from the Standard-scoring
    JuiceBoxOne source -- see DADS_ADP_PATH. `dads_adp` is that sheet's own
    blended ADP number (its "ADP" column); `dads_adp_avg` is the average of
    the four individual-site Standard ranks it also carries (its "Average"
    column) -- a useful cross-check when the two disagree by a lot on a
    specific player. Returns an empty frame if the source is unavailable."""
    try:
        raw = pd.read_csv(path)
    except (FileNotFoundError, OSError):
        return pd.DataFrame(columns=["norm_name", "dads_adp", "dads_adp_avg"])
    if not {"Name", "ADP"}.issubset(raw.columns):
        return pd.DataFrame(columns=["norm_name", "dads_adp", "dads_adp_avg"])

    out = pd.DataFrame({
        "norm_name": raw["Name"].map(_norm),
        "dads_adp": pd.to_numeric(raw["ADP"], errors="coerce"),
        "dads_adp_avg": pd.to_numeric(raw.get("Average"), errors="coerce"),
    })
    return out.sort_values("dads_adp").drop_duplicates("norm_name", keep="first").reset_index(drop=True)


def add_dads_scores(board_df):
    """Attach dad's-league columns to a board frame that already carries
    player_name, position, adp, injury_risk, and value_over_replacement_
    points (i.e. the output of build_recommendation_rankings_df).

    Adds:
      * dads_projection_points   -- dad's-scoring season fantasy points
                                    (NaN for players with no stat projection)
      * dads_vor                 -- dad's value over replacement (NaN likewise)
      * dads_final_score         -- the reworked model score (same engine
                                    formula as final_score, now market- AND
                                    risk-aware for dad's actual format --
                                    see dads_adp below; risk_penalty still
                                    comes from the board's own injury_risk,
                                    which isn't scoring-format-dependent)
      * dads_adp                 -- Standard-scoring ADP (DADS_ADP_PATH),
                                    falling back to the board's half-PPR adp
                                    for anyone missing from that sheet.
                                    Populated even for players with no dad's
                                    projection, so "what's he going in a
                                    standard room" stays answerable either way.
      * dads_adp_avg              -- that sheet's average of 4 individual-site
                                    Standard ranks, a cross-check against
                                    dads_adp when they disagree a lot.

    The board's own final_score/adp are left untouched; Home.py swaps these
    in only when the user selects the Dad's League scoring model.
    """
    # Local import avoids a circular dependency at module import time
    # (draft_analysis is a heavy module; dads_scoring is imported from it
    # nowhere, but keep the edge one-directional and lazy regardless).
    from draftkit.draft_analysis import calculate_base_value_score

    df = board_df.copy()
    if "player_name" not in df.columns or "position" not in df.columns:
        return df

    projections = build_dads_projections_df()
    if projections.empty:
        return df

    df["norm_name"] = df["player_name"].map(_norm)
    # Apply board-side aliases before the join.
    df["norm_name"] = df["norm_name"].replace(_NAME_ALIASES)
    df = df.merge(projections, on="norm_name", how="left")

    # Standard-scoring ADP (see DADS_ADP_PATH) -- the market signal dad's
    # scoring should actually use, not the board's own half-PPR `adp`. Falls
    # back to the half-PPR number for anyone missing from the Standard sheet
    # (deep bench names, etc.) rather than leaving the market term blank.
    dads_adp = load_dads_adp()
    if not dads_adp.empty:
        df = df.merge(dads_adp, on="norm_name", how="left")
        df["dads_adp"] = df["dads_adp"].fillna(pd.to_numeric(df.get("adp"), errors="coerce"))
    else:
        df["dads_adp"] = pd.to_numeric(df.get("adp"), errors="coerce")

    # Dad's replacement baselines / VOR -- keyed on DAD'S roster (see
    # _dads_replacement_baselines), so positional scarcity reflects his
    # 1QB/2RB/2WR/1TE no-flex league rather than the user's Half-PPR roster.
    baselines = _dads_replacement_baselines(df)
    positions = df["position"].astype(str).str.upper()
    baseline_series = positions.map(lambda p: baselines.get(p, 0.0))
    df["dads_vor"] = (
        pd.to_numeric(df["dads_projection_points"], errors="coerce") - baseline_series
    ).round(3)

    # Rework the model score by feeding dad's projection + VOR + Standard-
    # scoring ADP through the exact engine function. Only risk falls out of
    # the board unchanged (injury_risk isn't scoring-format-dependent); the
    # projection AND the market term now both reflect dad's actual format.
    scoring_input = pd.DataFrame(
        {
            "position": df["position"],
            "projection_points": pd.to_numeric(df["dads_projection_points"], errors="coerce"),
            "value_over_replacement_points": df["dads_vor"],
            "adp": df["dads_adp"],
            "injury_risk": pd.to_numeric(df.get("injury_risk"), errors="coerce"),
        }
    )
    scored = calculate_base_value_score(scoring_input)
    df["dads_final_score"] = scored["base_value_score"].values

    # A player with no dad's projection has no dad's score (don't fabricate).
    no_projection = pd.to_numeric(df["dads_projection_points"], errors="coerce").isna()
    df.loc[no_projection, ["dads_vor", "dads_final_score"]] = pd.NA

    return df.drop(columns=["norm_name"])
