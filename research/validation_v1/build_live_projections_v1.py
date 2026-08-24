"""projection_model_iteration_plan.pdf, Iteration 8: connect the new model
to the live draft tool.

Per review before building: this is a live-app-touching change, unlike
Iterations 1-7 (entirely confined to research/validation_v1/, zero blast
radius on the live app -- confirmed back in Iteration 2). Two decisions
made explicitly rather than defaulted:

  1. QB excluded. Its model is real, out-of-sample validated, and loses to
     ADP by ~33-36% (Iterations 3/4) -- not "unproven," a validated
     downgrade versus the market baseline already live in the app. Wiring
     it in would replace a currently-working signal with one shown to be
     worse. Only RB/WR/TE are projected here.
  2. New parallel column, not a replacement. Writes model_projection_points
     as an ADDITIONAL column, never touches the existing FantasyPros-sourced
     projection_points. Follows the exact precedent already in this
     codebase for a secondary projection signal (draftkit/data_sources/
     winwithodds_source.py -> draftkit/scripts/compare_sportsbook_vs_adp.py
     -> a side CSV -> Home.py merges it in at render time, guarded by
     `if path.exists()`) -- not a new pattern invented for this.

ID crosswalk (new, required): master_players.csv (the live app's player
universe) uses FantasyPros' own numeric player_id (e.g. 9221.0 for Jahmyr
Gibbs) -- ZERO overlap with the gsis-style player_id ("00-0023459") this
entire model was built and validated on (checked directly: 0/3935 raw
overlap). Built via clean_name(player_name)+position, matching
stats_player_reg_by_season's player_display_name, same pattern as
Iteration 5's snap-count crosswalk -- extended locally (NOT by editing the
shared validation_utils.clean_name(), too much blast radius on every
already-validated pipeline this late) to also strip II/III/IV suffixes,
since validation_utils.clean_name() only strips jr/sr. This exact gap was
real: "Kenneth Walker" (master_players) vs "Kenneth Walker III" (stats
file) failed to match under the unextended cleaner. Verified: 514/3475
RB/WR/TE rows matched, zero (name, position) collisions, and 513/514
matched ids have a real 2025 E6 row. The ~2960 unmatched rows are
overwhelmingly real rookies/deep-bench names with no NFL history to
project from anyway (already excluded by n_prior_seasons>0 downstream, and
handled by this app's separate rookie-projection system, not this model).

Feature construction for the live 2026 target season (2026 has no E6 row
-- the season hasn't been played -- so this is NOT a re-run of
run_position(), which requires an existing outcome row):
  - Recency-weighted baseline: real final_fantasy_ppg at E6 seasons
    2025/2024/2023 (lags 1/2/3), using each position's LOCKED
    POSITION_RECENCY_WEIGHTS (same weights Iteration 4 validated -- not
    re-derived here).
  - Efficiency stats: real 2025 stats_player_reg_by_season (season
    immediately before 2026 -- no leakage, matches the "_prior" convention
    used throughout).
  - Team context: offense_environment_team_seasons.csv's real 2026 rows
    (confirmed to exist), joined on the player's REAL CURRENT team from
    master_players.csv (not a historical team field).
  - Snap trend + missing-data flag (WR/TE only, matching
    POSITIONS_WITH_TREND_LAYER): real 2025 snap counts via Iteration 5's
    crosswalk.

Model: production fit -- trained on 100% of available real historical E6
rows (1999-2025), not held out, since this is a real forward projection,
not a validation run. Same Ridge(alpha=1.0) and current_best_features()
feature list Iteration 5/6/7 already validated -- no new modeling choices
introduced here.

Output scale: ppg x 17 (full-season total assuming health) -- games-played
uncertainty is handled separately by this app's existing injury_history.py
risk system, not re-solved here, consistent with Iteration 3's original
framing. 17 matches the implied season length in the existing
FantasyPros-sourced projection_points values (verified in Iteration 2's
half-PPR check).

Validating (plan's own gate, unchanged): re-run test_gibbs_anchor.py and
the full regression suite. Nothing ships unless both pass.

Usage:
    python build_live_projections_v1.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from build_baseline_projection_v1 import (
    E6_PATH,
    POSITION_EFFICIENCY_COLS,
    POSITION_RECENCY_WEIGHTS,
    TEAM_CONTEXT_COLS,
    _load_efficiency_by_season_player,
    _load_team_context,
)
from build_trend_layer_v1 import (
    POSITIONS_WITH_TREND_LAYER,
    TREND_COLS_RAW,
    _load_snap_trend_by_season_player,
)
from validation_utils import PROJECT_ROOT, VALIDATION_DIR, clean_name

MASTER_PLAYERS_PATH = PROJECT_ROOT / "data" / "processed" / "master_players.csv"
ROSTER_2026_PATH = VALIDATION_DIR / "data" / "roster_2026.csv"
ROOKIE_PROJECTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "rookie_projections.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_projections_v1.csv"
SPORTSBOOK_PATH = PROJECT_ROOT / "data" / "processed" / "sportsbook_vs_adp_comparison.csv"
TEAM_SCHEDULE_PATH = PROJECT_ROOT / "data" / "processed" / "team_schedule_risk.csv"
CONTINUITY_MEANINGFUL_VOLUME = {"RB": 100, "WR": 50, "TE": 30}  # carries+targets (RB) or targets (WR/TE) threshold for a "meaningful" competitor

# Real, confirmed team-code mismatch (2026-08-20): stats_player_reg_by_season
# and roster_2026.csv both use "LA" for the Rams; master_players.csv (and
# every downstream live-app display) uses "LAR" -- every other one of the
# 32 team codes matches exactly across all three sources, this is the one
# real exception. Left unfixed, this silently flagged EVERY Rams player as
# team_changed=True in compute_roster_continuity() (team_2025 "LA" != team
# "LAR"), and could equally corrupt departed/arrived detection for any
# player connected to the Rams via a real move. Normalized at the point
# each source is read, not by editing master_players.csv (regenerated by
# the separate data_pipeline.py) or leaving team-code comparisons to
# silently assume every source agrees.
TEAM_CODE_NORMALIZE = {"LA": "LAR"}
ROOKIE_FALLBACK_POSITIONS = ["RB"]  # rb_model_fix_plan.pdf, Phase 1 -- scoped to RB per that plan's own title/examples; same pattern could extend to WR/TE later

TARGET_SEASON = 2026
POSITIONS = ["RB", "WR", "TE"]  # QB excluded, see module docstring
GAMES_PER_SEASON = 17

# Real, verified current-team corrections for players BOTH upstream sources
# have wrong or missing (2026-08-20). Applied in _build_crosswalk() so they
# flow into team-context features AND roster-continuity detection, rather
# than hand-editing master_players.csv -- that file is regenerated by
# draftkit/data_pipeline.py, so an edit there would be silently reverted on
# the next pipeline run. Same disclosure standard as every other manual
# value in this codebase: real source + date, never a silent fix.
#
# Add an entry ONLY for a real, verified signing/trade that the sources
# genuinely missed -- not to force a preferred team assignment. Remove an
# entry once the upstream sources catch up (harmless if left, but it stops
# being load-bearing).
CURRENT_TEAM_OVERRIDES = {
    "00-0036893": {  # Najee Harris
        "team": "NYG",
        "source": "Real free-agent signing 2026-08-18 (one-year deal), confirmed via ESPN/AP/"
                  "Giants.com. master_players.csv still lists his 2025 team (LAC) -- stale by two "
                  "days at time of writing; roster_2026.csv omits him entirely (same real coverage "
                  "gap already documented for Tyreek Hill/Deebo Samuel in _build_crosswalk()).",
        "date": "2026-08-20",
    },
    "00-0030279": {  # Keenan Allen
        "team": "IND",
        "source": "Real free-agent signing 2026-08-18 (one-year deal, up to $8.32M), reuniting with "
                  "HC Shane Steichen (LAC 2014-2020). Confirmed via ESPN/CBS Sports/Colts depth "
                  "chart reporting. master_players.csv still lists his 2025 team (LAC) -- his real "
                  "2025 season (122 targets) was genuinely with LAC, but he departed for IND in "
                  "free agency, same stale-team pattern as Najee Harris above.",
        "date": "2026-08-20",
    },
    "00-0031610": {  # Darren Waller
        "team": "CAR",
        "source": "Real free-agent signing 2026-08-12 (one-year deal), resolving the ambiguous "
                  "'likely over in Miami' status flagged in the MIA batch earlier this session. "
                  "master_players.csv still lists him as 'FA'; roster_2026.csv omits him entirely, "
                  "same real coverage gap pattern as Najee Harris/Keenan Allen above.",
        "date": "2026-08-20",
    },
}


def _clean_name_ext(name: str) -> str:
    """clean_name() + strip II/III/IV/V suffixes. Local to this script --
    not editing the shared validation_utils.clean_name() (see docstring)."""
    base = clean_name(name)
    return re.sub(r"\b(ii|iii|iv|v)\b", "", base).strip()


def _build_crosswalk(positions: list[str]) -> pd.DataFrame:
    """Two-tier identity match (roster continuity fix, Step 2). Checked
    directly before building this way: roster_2026.csv was proposed as the
    sole "broad roster reference" for identity resolution, but real
    verification found it MISSING prominent, unambiguous veterans entirely
    (Tyreek Hill, Deebo Samuel -- not name-mismatches, genuinely absent
    rows; all 32 teams present, so not a team-filter issue either). Using
    it as the primary/only source would have regressed real coverage for
    established players while fixing only the rookie case.

    Fixed as a fallback, not a replacement: try stats_player_reg_by_season
    2025 first (the original, proven ~91-93% match source), and ONLY for
    players it misses, also check roster_2026.csv (which DOES carry a real
    gsis_id for 2026 rookies with zero NFL games, e.g. Jeremiyah Love ->
    00-0041027 -- exactly the case Step 2 needs). Verified directly: this
    combination gains 316 real players (514->830 matched) with zero
    regression -- Hill/Samuel/Diggs/Allen all still match via the stats-
    based primary path, since they clearly do have real 2025 production."""
    stats = pd.read_csv(
        VALIDATION_DIR / "data" / "stats_player_reg_by_season" / "2025.csv",
        usecols=["player_id", "player_display_name", "position"],
    )
    stats = stats[stats["position"].isin(positions)].dropna(subset=["player_display_name", "player_id"])
    stats["key"] = stats["player_display_name"].apply(_clean_name_ext)
    stats = stats.drop_duplicates(["key", "position"], keep="first")
    stats_map = {(row.key, row.position): row.player_id for row in stats.itertuples()}

    roster = pd.read_csv(ROSTER_2026_PATH)
    roster = roster[roster["position"].isin(positions)].dropna(subset=["full_name", "gsis_id"])
    roster["key"] = roster["full_name"].apply(_clean_name_ext)
    roster = roster.drop_duplicates(["key", "position"], keep="first")
    roster_map = {(row.key, row.position): row.gsis_id for row in roster.itertuples()}

    mp = pd.read_csv(MASTER_PLAYERS_PATH)
    mp = mp[mp["position"].isin(positions)].copy()
    mp["key"] = mp["player_name"].apply(_clean_name_ext)
    key_pos = list(zip(mp["key"], mp["position"]))
    gsis_from_stats = pd.Series(key_pos, index=mp.index).map(stats_map)
    gsis_from_roster = pd.Series(key_pos, index=mp.index).map(roster_map)
    mp["gsis_id"] = gsis_from_stats.fillna(gsis_from_roster)
    mp["identity_source"] = np.select(
        [gsis_from_stats.notna(), gsis_from_roster.notna()],
        ["stats_2025", "roster_2026"],
        default="unmatched",
    )
    # Real current-team corrections (see CURRENT_TEAM_OVERRIDES). Applied
    # here, before team context and roster continuity are computed off this
    # frame, so a real but source-missed signing doesn't silently carry the
    # player's OLD team's 2026 offense environment through the whole model.
    for gsis_id, override in CURRENT_TEAM_OVERRIDES.items():
        mask = mp["gsis_id"].eq(gsis_id)
        if not mask.any():
            continue
        stale_team = mp.loc[mask, "team"].iloc[0]
        if stale_team == override["team"]:
            continue  # upstream caught up -- override is now a no-op
        mp.loc[mask, "team"] = override["team"]
        print(f"  team override: {gsis_id} {stale_team} -> {override['team']} (verified {override['date']})")
    return mp


def _recency_weighted_baseline_live(e6: pd.DataFrame, player_ids: pd.Series, position: str) -> pd.DataFrame:
    weights = POSITION_RECENCY_WEIGHTS[position]
    base = e6[(e6["position"] == position) & (e6["season"].isin([2023, 2024, 2025]))][
        ["season", "player_id", "final_fantasy_ppg", "games_played"]
    ]
    out = pd.DataFrame({"player_id": player_ids.unique()})
    weighted_sum = pd.Series(0.0, index=out.index)
    weight_total = pd.Series(0.0, index=out.index)
    n_seasons_available = pd.Series(0, index=out.index)
    for lag, weight in zip([1, 2, 3], weights):
        season = TARGET_SEASON - lag
        lagged = base[base["season"] == season][["player_id", "final_fantasy_ppg", "games_played"]]
        out = out.merge(lagged, on="player_id", how="left", suffixes=("", f"_lag{lag}"))
        out = out.rename(columns={"final_fantasy_ppg": f"ppg_lag{lag}", "games_played": f"games_lag{lag}"})
        has_lag = out[f"ppg_lag{lag}"].notna() & (out[f"games_lag{lag}"].fillna(0) > 0)
        weighted_sum = weighted_sum + np.where(has_lag, out[f"ppg_lag{lag}"] * weight, 0.0)
        weight_total = weight_total + np.where(has_lag, weight, 0.0)
        n_seasons_available = n_seasons_available + has_lag.astype(int)
    out["recency_weighted_ppg_baseline"] = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
    out["n_prior_seasons"] = n_seasons_available
    return out[["player_id", "recency_weighted_ppg_baseline", "n_prior_seasons"]]


def _train_production_model(e6: pd.DataFrame, position: str, features: list[str], feature_frame: pd.DataFrame) -> Ridge:
    train = feature_frame[
        feature_frame["position"].eq(position) & feature_frame["n_prior_seasons"].gt(0) & feature_frame["season"].lt(TARGET_SEASON)
    ].copy()
    train_y = pd.to_numeric(train["final_fantasy_ppg"], errors="coerce")
    valid = train_y.notna()
    train, train_y = train[valid], train_y[valid]
    medians = train[features].median(numeric_only=True)
    train_x = train[features].fillna(medians)
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_y)
    return model, medians


def build_training_frame() -> pd.DataFrame:
    """Reuses build_baseline_projection_v1.build_features()-equivalent
    logic for the HISTORICAL rows only (needed to fit the production
    model) -- separate from the live 2026 rows, which get built directly
    since no E6 row exists for them."""
    from build_trend_layer_v1 import build_features_with_trend
    return build_features_with_trend()


def _load_2025_position_volume(positions: list[str]) -> pd.DataFrame:
    """Real 2025 season volume (carries+targets for RB, targets for WR/TE)
    per player, with their real 2025 team -- the source for detecting
    same-team-same-position competitor departures/arrivals (Step 4)."""
    stats = pd.read_csv(
        VALIDATION_DIR / "data" / "stats_player_reg_by_season" / "2025.csv",
        usecols=["player_id", "position", "recent_team", "carries", "targets"],
    )
    stats = stats[stats["position"].isin(positions)].copy()
    stats["recent_team"] = stats["recent_team"].replace(TEAM_CODE_NORMALIZE)
    stats["volume"] = np.where(stats["position"].eq("RB"), stats["carries"].fillna(0) + stats["targets"].fillna(0), stats["targets"].fillna(0))
    return stats[["player_id", "position", "recent_team", "volume"]].rename(columns={"player_id": "gsis_id", "recent_team": "team_2025"})


def _load_current_team_by_gsis() -> dict[str, str]:
    """Real CURRENT (2026) team for ANY gsis_id, sourced from roster_2026.csv
    directly -- broader than this script's own live-draft-pool crosswalk
    (which excludes QB and unmatched players), so a departed competitor who
    isn't in this year's fantasy-relevant pool at all still resolves
    correctly instead of looking like "still on the old team."""
    roster = pd.read_csv(ROSTER_2026_PATH, usecols=["gsis_id", "team"]).dropna(subset=["gsis_id"])
    roster["team"] = roster["team"].replace(TEAM_CODE_NORMALIZE)
    return roster.drop_duplicates("gsis_id").set_index("gsis_id")["team"].to_dict()


def compute_roster_continuity(crosswalk: pd.DataFrame, position: str) -> pd.DataFrame:
    """Step 1 (flag detection) + Step 4 (competitor-change detection,
    interim -- NOT folded into the model, see module docstring). Returns
    one row per PROJECTED player in `crosswalk` with:
      team_changed: real current team (master_players.csv) != real 2025
        team (stats_player_reg_by_season) -- Kenneth Walker III case.
      competitor_departed / _arrived: real same-team-same-season,
        same-position player above CONTINUITY_MEANINGFUL_VOLUME whose
        current team differs from the relevant season -- Gibbs/Montgomery
        case. Named in a human-readable note for the UI tooltip and any
        future manual projection_adjustment.
    """
    projected = crosswalk[crosswalk["model_projection_status"].eq("projected")].copy()
    if projected.empty:
        return pd.DataFrame(columns=["gsis_id", "team_changed", "competitor_departed", "competitor_arrived", "continuity_note"])

    volume = _load_2025_position_volume([position])
    current_team_by_gsis = _load_current_team_by_gsis()
    threshold = CONTINUITY_MEANINGFUL_VOLUME[position]

    own_2025 = volume.set_index("gsis_id")["team_2025"]
    projected["team_2025"] = projected["gsis_id"].map(own_2025)
    projected["team_changed"] = projected["team_2025"].notna() & (projected["team_2025"] != projected["team"])

    rows = []
    for row in projected.itertuples():
        notes = []
        departed = False
        arrived = False
        if pd.notna(row.team_2025):
            teammates_2025 = volume[
                volume["team_2025"].eq(row.team_2025) & volume["position"].eq(position)
                & volume["gsis_id"].ne(row.gsis_id) & volume["volume"].ge(threshold)
            ]
            for tm in teammates_2025.itertuples():
                current = current_team_by_gsis.get(tm.gsis_id)
                if current != row.team_2025:
                    departed = True
                    notes.append(f"departed teammate (2025 vol={tm.volume:.0f})")
        arrivals = volume[
            volume["position"].eq(position) & volume["team_2025"].ne(row.team if pd.notna(row.team_2025) else "__none__")
            & volume["gsis_id"].ne(row.gsis_id) & volume["volume"].ge(threshold)
        ]
        for cand in arrivals.itertuples():
            if current_team_by_gsis.get(cand.gsis_id) == row.team:
                arrived = True
                notes.append(f"new teammate arrived (2025 vol={cand.volume:.0f} elsewhere)")
        rows.append({
            "gsis_id": row.gsis_id,
            "team_changed": bool(row.team_changed),
            "competitor_departed": departed,
            "competitor_arrived": arrived,
            "continuity_note": "; ".join(notes) if notes else None,
        })
    return pd.DataFrame(rows)


# Diagnosed-defect corrections (model_proj_staleness_fix_plan.pdf, Step 3).
# Distinct from draftkit/draft_analysis.py's PROJECTION_MANUAL_ADJUSTMENTS,
# which is reserved for narrative/editorial overrides on the FantasyPros-
# sourced projection_points, where the external number itself has no known
# defect (AJ Brown/DJ Moore trades, McVay comments, injury timelines).
# MODEL_PROJECTION_CORRECTIONS instead corrects a diagnosed, mechanical
# staleness bug INSIDE this model's own feature pipeline -- team-context/
# competitor-volume features that don't yet reflect a real, current-
# offseason roster change -- for specific, individually-verified players.
# model_projection_points is NEVER hand-edited by this mechanism (stays the
# pure, validated model output backing compare_model_versions()'s win-count
# standard); this only populates the separate model_projection_points_adjusted
# column below. Keyed on gsis_id (stable, never null for a matched row),
# not player_name/player_id.
#
# Jahmyr Gibbs (added 2026-08-19, reverted from draft_analysis.py's
# PROJECTION_MANUAL_ADJUSTMENTS -- that was the wrong column for this exact
# case, see that dict's comment): real touch-share vacancy math, not a
# ported-over guess. Detroit real 2025: Montgomery 158 carries + 29
# targets = 187 touches, traded to HOU (confirmed March 2026). Real
# incoming Isiah Pacheco: 144 touches at KC in 2025, but arrives DET in a
# backup role with an MCL sprain -- realistic 2026 touches ~60-90, not his
# full 2025 workload (per model_proj_staleness_fix_plan.pdf). Net backfield
# vacancy Pacheco doesn't absorb: 187 - 75 (range midpoint) = 112 touches.
#
# Corrected 2026-08-19 (rb_model_fix_plan.pdf, Phase 0): the original entry
# used a 75% pass-through (112 * 0.75 = 84 touches -> +24.9%, 327.7), which
# the follow-up plan flagged as an arbitrary number without a stated
# rationale. Replaced with a reasoned 50% split -- half the net vacancy
# realistically goes to Gibbs, the other half to game-script variance and
# Pacheco outperforming the low end of his own 60-90 range -- 112 * 0.50 =
# 56 touches, or 56 / 337 (Gibbs' own real 2025 touches) = +16.6% applied
# to his raw model_projection_points (262.4 -> 306.0). Still a disclosed,
# rough proportional estimate, same standard as every entry in
# PROJECTION_MANUAL_ADJUSTMENTS -- not a precision touch-to-point
# conversion model.
#
# rb_model_fix_plan.pdf, Phase 2 (added 2026-08-19): 13 more RBs, triaged
# by real departed/arrived teammate touch volume. Every departed/arrived
# player was individually IDENTIFIED by name (not trusted from the raw
# continuity_note volume alone) and their real current role researched
# live (trades, free-agent contract size, training-camp depth charts) --
# per that plan's explicit Step 1-2 process. Same net-vacancy-touches x
# reasoned-pass-through-share / own-2025-touches formula as Gibbs, except
# where noted below as a direct role-share read from confirmed reporting
# language instead (Irving/Tucker/Marks/JCM -- the real sourcing was
# specific enough about the actual split that redoing it through the
# blunter net-vacancy formula would have thrown away real information).
#
# Known caveat, disclosed rather than hidden: dividing by a player's own
# SMALL 2025 touch base amplifies the resulting pct when that low volume
# was itself suppressed by the departing player's presence (Tuten behind
# Etienne, Conner's unusually thin 41-touch season). Both cases originally
# used a HALVED pass-through share specifically to counteract that
# amplification (user-directed 2026-08-19, after Tuten/Conner's uncapped
# math produced +60.6%/-21.5%) -- not a claim the discount was precisely
# calibrated, just a disclosed, deliberate conservatism on the two most
# base-amplified cases.
#
# Tuten superseded 2026-08-19 (rb_correction_manual_review_prompt.pdf,
# per-player manual review pass): the net-vacancy-formula description
# above no longer matches his entry below, which was rebuilt on a
# different, Book-anchored (sportsbook_vs_adp_comparison.csv) methodology
# after the review surfaced a real fact (Rodriguez's foot surgery) the
# original note omitted -- see that entry's own note for the full
# reasoning. Conner's entry below still reflects the original halved-
# share approach, pending its own turn in that same review pass.
MODEL_PROJECTION_CORRECTIONS = {
    "00-0039139": {  # Jahmyr Gibbs
        "pct": 16.6,
        "note": "Montgomery traded to HOU (187 real 2025 touches); Pacheco arrives DET in a "
                "backup role w/ MCL sprain (~60-90 realistic touches, not his full 144) -- net "
                "vacancy ~112 touches, 50% (reasoned split, corrected 2026-08-19 from an "
                "unjustified 75%) allocated to Gibbs = +16.6% vs. his 337 real 2025 touches",
        "date": "2026-08-19",
    },
    "00-0040719": {  # Bhayshul Tuten
        "pct": 82.1,
        "note": "Superseded 2026-08-19 (second pass) -- rebuilt independently from real per-touch "
                "rates instead of Book's split. Etienne left JAX entirely (real FA to NO, 312 real "
                "2025 touches). Rodriguez arrived (real FA, 2yr/$10M, 116 real 2025 touches at WAS) "
                "but underwent real foot surgery, missed OTAs/minicamp, 'slow and calculated' return "
                "-- real, ongoing competition for the job, not a handcuff, but near-term snap count "
                "capped by recovery. Real 2025 per-touch rates: Tuten 0.862 half-PPR/touch (97 "
                "touches -> 83.6 pts), Rodriguez 0.780 (116 touches -> 90.5 pts). Shared 320-touch "
                "JAX backfield pool (real shrink from the old 409-touch Tuten+Etienne total), 60/40 "
                "touch-share favoring healthy, 'earned the nod' Tuten over recovering Rodriguez: "
                "Tuten 192 touches x 0.862 = 165.5; Rodriguez 128 touches x 0.78 = 99.8. Raw model "
                "output 90.9 -> 165.5 = +82.1%. See Rodriguez's (00-0038611) companion entry.",
        "date": "2026-08-19",
    },
    "00-0036555": {  # Chuba Hubbard
        "pct": -7.4,
        "note": "Superseded 2026-08-19 (second pass) -- rebuilt independently from real per-touch "
                "rates instead of Book. Dowdle left CAR entirely (real FA to PIT, 286 real 2025 "
                "touches). Jonathon Brooks (real CAR depth chart #2, recovered from back-to-back "
                "ACL tears, real training-camp buzz) is a real, ascending competitor invisible to "
                "touch-volume flag detection. Hubbard himself has a real, current hamstring injury "
                "(week-to-week, HC 'confident' for Week 1) -- Brooks got the real starting nod in "
                "his place for a preseason game. Real 2025 per-touch rates: Hubbard 0.638 half-PPR/"
                "touch (173 touches -> 110.4 pts), Brooks has no usable real per-touch history (his "
                "real NFL sample is essentially wiped by the two ACL tears) -- a proxy rate (~0.68, "
                "comparable committee-back baseline) is used instead, disclosed as lower-confidence. "
                "Shared 300-touch CAR pool (real shrink from the old 459-touch Hubbard+Dowdle "
                "total), 65/35 touch-share favoring the still-established Hubbard despite his camp "
                "injury: Hubbard 195 touches x 0.638 = 124.4. Raw model output 134.4 -> 124.4 = "
                "-7.4%. See Brooks' (00-0039344) companion entry.",
        "date": "2026-08-19",
    },
    "00-0039165": {  # Zach Charbonnet
        "pct": -58.0,
        "note": "Walker left SEA (real FA to KC, 257 real 2025 touches), but this does NOT mean "
                "Charbonnet inherits the vacancy: Charbonnet tore his ACL in the January 2026 "
                "playoffs, underwent surgery, and was placed on the PUP list as camp opened (July "
                "22) -- real, decisive reporting frames him as unlikely to be ready by Week 1 and "
                "likely to miss a meaningful portion of the 2026 season. Exact return timeline "
                "unsettled as of Aug 16 (most recent, most specifically-sourced reporting: "
                "'realistically looking at a midseason return' per Seattle Times/Bob Condotta, vs. "
                "earlier, less-specific 'possible Week 1' framing from an April post-draft piece "
                "and a generic depth-chart tracker) -- full deference to Book sidesteps needing to "
                "resolve this precisely, since Book's pricing already reflects the market's "
                "collective read on it. The real beneficiary of Walker's departure is rookie "
                "Jadarian Price (SEA's own 2026 1st-round pick, #32 overall, drafted specifically "
                "to fill this vacancy), with reporting describing the starting job as 'becoming "
                "more and more clear as a spot owned by Price... well beyond' just this year. "
                "Emanuel Wilson (real FA, 1yr/$2.19M) is real depth behind Price. Book's real "
                "current pricing: Charbonnet 63.7 half-PPR pts, Price 143.75, Wilson 51.0 (258.45 "
                "combined) -- a ~24.6/55.6/19.7 real split reflecting Charbonnet as a complementary "
                "eventual-return piece, not this year's starter. No disclosed reason to depart "
                "from Book here (the ACL tear and PUP designation are hard medical facts, not a "
                "market read I have grounds to second-guess) -- full deference: 151.5 raw model "
                "output -> 63.7 = -58.0%. See Rodriguez (00-0038611)/Brooks (00-0039344)/Wilson "
                "(00-0038797) entries below for the companion corrections on Tuten's and this "
                "player's real competitors.",
        "date": "2026-08-19",
    },
    "00-0037228": {  # Jaylen Warren
        "pct": -11.6,
        "note": "Superseded 2026-08-19 (second pass) -- rebuilt independently from real per-touch "
                "rates instead of Book's near-even split. Gainwell left PIT (real FA to TB, 199 real "
                "2025 touches). Dowdle arrived (real FA, 2yr/$12.25M, 286 real 2025 touches at CAR, "
                "reunited with new HC Mike McCarthy). Real, genuine, unsettled battle -- Warren "
                "opened RB1 on the Aug 5 depth chart despite Dowdle taking the majority of first-"
                "team practice reps beforehand; Warren separately re-signed a real 2yr/$17.5M "
                "extension. Real 2025 per-touch rates: Warren 0.770 half-PPR/touch (256 touches -> "
                "197.1 pts), Dowdle 0.688 (286 touches -> 196.8 pts) -- Warren's real rate is "
                "meaningfully higher. Shared 380-touch PIT pool (moderate shrink from the old "
                "455-touch Warren+Gainwell total), 52/48 split favoring Warren per his official "
                "depth-chart standing despite Dowdle's camp-reps edge: Warren 197.6 touches x 0.770 "
                "= 152.2. Raw model output 172.2 -> 152.2 = -11.6% (less negative than the prior "
                "Book-matched -16.9%). See Dowdle's (00-0036139) companion entry.",
        "date": "2026-08-19",
    },
    "00-0036139": {  # Rico Dowdle (PIT) -- companion to Warren (00-0037228)
        "pct": -28.5,
        "note": "Superseded 2026-08-19 (second pass) -- same 380-touch PIT pool as Warren's "
                "(00-0037228) entry, not matched to Book. Real FA signing w/ PIT (2yr/$12.25M), "
                "reunited with new HC Mike McCarthy, real camp surge to majority first-team reps. "
                "Real 2025 per-touch rate (at CAR): 0.688 half-PPR/touch -- meaningfully lower than "
                "Warren's 0.770. 48% share of the 380-touch pool = 182.4 touches x 0.688 = 125.5. "
                "Raw model output 175.5 -> 125.5 = -28.5% -- MORE negative than the prior Book-"
                "matched -16.7%, since the real per-touch rate gap between the two backs is wider "
                "than the near-even points split Book was pricing implied.",
        "date": "2026-08-19",
    },
    "00-0037197": {  # Isiah Pacheco
        "pct": -56.2,
        "note": "The naive continuity flag (Kareem Hunt departing KC, his OLD team) is a real red "
                "herring -- irrelevant to Pacheco's new team. His actual 2026 role is CONFIRMED "
                "directly: 'backup running back to starter Jahmyr Gibbs,' 1yr/$1.81M deal, "
                "short-yardage/change-of-pace ('spells Gibbs for a series'), plus a real MCL sprain. "
                "Book's real current pricing: 50.60 half-PPR pts -- lower than even the 60-90-touch "
                "estimate this correction originally used (which implied ~60.2), with no additional "
                "disclosed fact found to explain the extra gap beyond the already-confirmed backup "
                "role and injury, so deferring to Book per the standing rule. Split check: Gibbs "
                "(297.15) / Pacheco (50.60) Book points implies an 85.5/14.5 real split; Pacheco's "
                "implied touches (~50-65 at a plausible 0.8-1.0 pts/touch for a non-receiving power "
                "role) are a reasonable, non-alarming number for a capped backup role. Raw model "
                "output 115.6 -> 50.6 (Book) = -56.2%. Note: Gibbs' companion side of this backfield "
                "was corrected separately in Phase 0 (+16.6%, 262.4->306.0) and is out of this "
                "batch's scope; Book's own number for Gibbs (297.15) is reasonably close (~3% gap) "
                "-- flagged for a low-priority follow-up cleanup pass, not reopened here.",
        "date": "2026-08-19",
    },
    "00-0039361": {  # Bucky Irving
        "pct": 5.3,
        "note": "Superseded 2026-08-19 (third pass) -- rebuilt from a career-spanning rate blend, "
                "not his single depressed 2025 season. White left TB (real FA to WAS, 177 real 2025 "
                "touches); Gainwell arrived (real FA, 2yr/$14M, 199 real 2025 touches at PIT). "
                "Irving's 2025 (10 games, 208 touches -> 123.5 pts, 0.594/touch) is real but not "
                "representative: only 1 rushing TD on 173 carries (likely Rachaad White, still on "
                "the roster for part of the year, absorbing real goal-line work), AND real, publicly "
                "confirmed injury (foot/shoulder, season-ending shoulder surgery) plus mental health "
                "struggles during recovery, directly acknowledged by Irving himself and corroborated "
                "across multiple outlets. His healthy 2024 season tells a different story: 17 games, "
                "259 touches -> 220.9 pts, 0.853/touch -- a legitimately elite full-season rate. "
                "Blended 75% 2024 / 25% 2025 = 0.789/touch. 44% share of the 400-touch TB pool = 176 "
                "touches x 0.789 = 138.9. On top of that: Gainwell's own rate was separately revised "
                "down (see his companion entry) from his 2025-only figure (0.929) to a real 4-season "
                "career average (0.776), freeing up 26.9 points that don't just vanish -- those "
                "touches would realistically flow to the more established back, so they're added to "
                "Irving's total: 138.9 + 26.9 = 165.8. Raw model output 176.6 -> 165.8 = -6.1%. See "
                "Gainwell's (00-0036919) and Tucker's (00-0038951) companion entries, same shared "
                "pool. REVISED 2026-08-21, user-directed (underrated): fresh mid-August camp reports "
                "confirm the shoulder is fully healed ('no restrictor plates'), Irving is 'full-go' "
                "and taking the most first-team carries, with Gainwell used primarily as a receiver -- "
                "cleaner real workhorse confirmation than the offseason rate-blend alone captured. "
                "+18% on top of the prior 165.8 = 195.6. Raw model output 176.6 -> 195.6 = +10.8%. "
                "REVISED again 2026-08-21, user-directed: -5% off 195.6 = 185.9. Raw model output "
                "176.6 -> 185.9 = +5.3%.",
        "date": "2026-08-21",
    },
    "00-0036919": {  # Kenneth/Kenny Gainwell (TB) -- companion to Irving (00-0039361); same gsis_id resolves both name-variant rows
        "pct": 1.6,
        "note": "Superseded 2026-08-19 (third pass) -- rebuilt from his real career-spanning rate, "
                "not his 2025 spike alone. Real FA signing w/ TB (2yr/$14M) after a real career-best "
                "2025 at PIT (73 catches/85 targets/486 rec yards, 5 rushing TDs on 114 carries -> "
                "0.929 half-PPR/touch). But his full real career is volatile, not steadily ascending "
                "-- 2022 (rookie) 0.932, 2023 0.681, 2024 0.563, 2025 0.929. The 2025 rate is close "
                "to his rookie-year rate on a much larger sample, not a new sustained talent level; "
                "the two intervening seasons were real and meaningfully weaker. 4-season simple "
                "average: (0.932+0.681+0.563+0.929)/4 = 0.776 half-PPR/touch. 44% share of the "
                "400-touch pool = 176 touches x 0.776 = 136.6. Raw model output 134.4 -> 136.6 = "
                "+1.6% -- a large reduction from the prior 2025-only-based +21.7%, since the career "
                "average reveals real, meaningful volatility his best season alone obscured.",
        "date": "2026-08-19",
    },
    "00-0038951": {  # Sean Tucker
        "pct": -45.7,
        "note": "Superseded 2026-08-19 (second pass) -- same 400-touch TB pool as Irving's "
                "(00-0039361)/Gainwell's (00-0036919) entries. CONFIRMED: Tucker 'remains buried,' "
                "'RB3,' 'precarious situation' behind both Irving and Gainwell. CAVEAT on his real "
                "per-touch rate (0.901 half-PPR/touch, 97 touches -> 87.4 pts): this is driven almost "
                "entirely by 7 rushing TDs on just 86 carries (an 8.1% TD-per-carry rate, well above "
                "anything sustainable) with almost no receiving work (8 catches, 34 yards) -- very "
                "likely hot short-yardage/goal-line variance on a small sample, not a repeatable "
                "skill level. Using this rate at face value probably overstates his true expected "
                "value going forward, but it's the only real data point available. 12% share of the "
                "400-touch pool = 48 touches x 0.901 = 43.2. Raw model output 79.5 -> 43.2 = -45.7%.",
        "date": "2026-08-19",
    },
    "00-0040583": {  # Woody Marks
        "pct": -13.2,
        "note": "REVISED 2026-08-21 (again -- user flagged Montgomery still too high after the "
                "first trim; real August 2026 camp reporting found and applied). NFL Network's Jane "
                "Slater reports Houston is planning to split lead-back duties, assessing a real "
                "'55-45' arrangement in camp, and multiple outlets (si.com/onsi, heavy.com) describe "
                "Marks as earning 'a major role from the start of the season' rather than the "
                "backup/complement role originally assumed -- a real update on top of the prior "
                "60/40 assumption, which was itself only a same-day-audit guess with no camp evidence "
                "behind it. Direction of the 55-45 split (which back gets the 55) is not resolved in "
                "reporting, so moved to a genuinely even 50/50 of the same 420-touch pool rather than "
                "presuming Marks wins it: 210 touches x 0.574 (his own real rate) = 120.5. Raw model "
                "output 138.9 -> 120.5 = -13.2%. See Montgomery's (00-0035685) companion entry.",
        "date": "2026-08-21",
    },
    "00-0035685": {  # David Montgomery (HOU) -- companion to Marks (00-0040583)
        "pct": 21.8,
        "note": "REVISED 2026-08-21 (fourth pass, user-directed: -5% off the Book+15% value). "
                "200.5 x 0.95 = 190.5. Raw model output 156.4 -> 190.5 = +21.8%. Prior basis: later "
                "August camp reporting shifted further than the prior trim accounted for: multiple "
                "outlets now describe Montgomery as 'expected to be the backfield's star' and DeMeco "
                "Ryans has praised him directly, a more decisive lead-back framing than the July "
                "55-45 split report the prior trim was built on."
                "reporting shifted further than the prior trim accounted for: multiple outlets now "
                "describe Montgomery as 'expected to be the backfield's star' and DeMeco Ryans has "
                "praised him directly, a more decisive lead-back framing than the July 55-45 split "
                "report the prior trim was built on. User set the target explicitly at 15% above "
                "Book (174.45 x 1.15 = 200.5) rather than a re-derived touch-share number. Raw model "
                "output 156.4 -> 200.5 = +28.2%. NOTE: this reopens the 50/50 touch-pool assumption "
                "shared with Marks' (00-0040583) companion entry -- Marks was left unchanged since "
                "only Montgomery was in scope of this instruction.",
        "date": "2026-08-21",
    },
    "00-0035700": {  # Josh Jacobs
        "pct": -10.5,
        "note": "Wilson left GB (real FA to SEA, 142 real 2025 touches) but his snaps had already "
                "faded to Chris Brooks late in 2025 -- not the real driver here. Full depth chart "
                "verified: Jacobs > Chris Brooks/MarShawn Lloyd (00-0039811)/Pierre Strong Jr. "
                "MarShawn Lloyd (real 2024 3rd-round pick, 2 seasons mostly lost to hamstring "
                "injuries, essentially no usable real per-touch history -- 1 game/6 carries in 2024 "
                "is his entire real NFL sample) is reported 'finally healthy' and the real 'No. 2 RB' "
                "option, invisible to touch-based flag detection for the same reason as CAR's "
                "Jonathon Brooks. Jacobs' own real 2025 rate: 278 touches -> 201.1 standard/237.1 "
                "PPR pts (~219 half-PPR, ~0.79 pts/touch). Real, direct, attributed RBs coach "
                "assessment (Ben Sirmans, May 2026): 'same ability as his Pro Bowl caliber first "
                "year,' 'little drop-off' -- kept the per-touch RATE flat for 2026, no aging discount "
                "applied there. The real driver instead: Jacobs was arrested in May 2026 (domestic "
                "disturbance, no charges filed), the investigation remains open as of July 23, and "
                "Schefter reported as recently as Aug 11 that NFL Personal Conduct Policy discipline "
                "is still 'pending' -- not yet resolved, not yet ruled out. Built independently, NOT "
                "matched to Book: a deterministic (not probability-weighted) assumption of 2 missed "
                "games (explicit judgment call, no official ruling exists to source this from) -- "
                "healthy-pace total (216.6, real rate x ~flat 2026 touch share) x 15/17 games played "
                "= 191.1. Raw model output 213.5 -> 191.1 = -10.5%. See Lloyd's (00-0039811) "
                "companion entry -- his number moves the OPPOSITE direction under this same "
                "assumption, since his 2 guaranteed primary-back games are worth more built as a "
                "certainty than diluted across a probability-weighted expectation.",
        "date": "2026-08-19",
    },
    "00-0039811": {  # MarShawn Lloyd (GB) -- companion to Jacobs (00-0035700)
        "pct": 15.5,
        "note": "Companion entry to Jacobs' (00-0035700) correction -- built independently from the "
                "same real 2-missed-game assumption, NOT from Book. No usable real per-touch history "
                "of his own (1 game/6 carries in 2024 is his entire real NFL sample; 2025 has no row "
                "at all) -- constructed from role-share logic instead: in the 15 games Jacobs plays, "
                "Lloyd's real complementary 'No. 2 RB' role is estimated at ~20% of Jacobs' per-game "
                "pace (~2.55 pts/game); in the 2 games Jacobs is assumed out, Lloyd becomes the real "
                "primary back (sharing with Chris Brooks/Pierre Strong Jr. but taking the majority "
                "share, ~14.0 pts/game at that role). 15x2.55 + 2x14.0 = ~66.3. Raw model output "
                "57.4 -> 66.3 = +15.5%. Every input here (20%/majority-share splits, the 2-game "
                "figure itself) is a disclosed judgment call, not a sourced fact.",
        "date": "2026-08-19",
    },
    "00-0033553": {  # James Conner
        "pct": -71.3,
        "note": "Superseded 2026-08-19 -- the original net-vacancy-touch read (Allgeier arriving, "
                "Carter departing) missed the real story entirely. Conner's thin 41-touch 2025 base "
                "was itself an artifact of a real, decisive Week 3 foot injury requiring season-ending "
                "surgery, not representative of his normal role -- Trey Benson (real, previously "
                "unresearched) became ARI's real 2025 lead back in his absence. Full real depth chart "
                "verified: Allgeier (RB1) > Jeremiyah Love (00-0041027, RB2) > Conner (RB3) > Benson "
                "(RB4) -- reporting frames Allgeier/Love as a real '1A/1B,' with Love 'almost certain' "
                "to be the real top back by the regular season. Conner himself: age 31, still not "
                "practicing in full team (11v11) drills per the most recent report, HC gave 'no "
                "timetable,' explicitly 'no reason to rush him back' given Love/Allgeier ahead of him. "
                "Built INDEPENDENTLY, not matched to Book: real established healthy-season rate from "
                "2023-2024 (before the injury) ~0.785 pts/touch (291 touches -> 230.3 half-PPR in "
                "2024; 241 touches -> ~188 in 2023). Real touch-share estimate as genuine RB3 in a "
                "420-touch total-pool assumption (shared consistently with Love's and Allgeier's "
                "entries below): 52 touches, further discounted ~20% for the real, disclosed recovery-"
                "timeline risk (judgment call, no official return date exists) -> ~41 points. This "
                "lands BELOW Book's own number (53.70) -- independently constructed, not deferred. "
                "Raw model output 142.8 -> 41.0 = -71.3%. See Love's (00-0041027) rookie-fallback "
                "override and Allgeier's (00-0037263) companion entry -- all three share the same "
                "420-touch pool assumption, cross-checked against but not matched to Book.",
        "date": "2026-08-19",
    },
    "00-0037263": {  # Tyler Allgeier (ARI) -- companion to Conner (00-0033553); shares the 420-touch ARI backfield pool assumption
        "pct": -41.5,
        "note": "Companion entry to Conner's (00-0033553) correction, same independently-constructed "
                "420-touch ARI backfield pool. Real established rate from his actual 2025 ATL "
                "production: 159 touches -> 116 half-PPR pts (~0.73 pts/touch). Real 2026 role: "
                "officially RB1 on ARI's depth chart, but real substance ('1A/1B' with Love, Love "
                "'almost certain' to be the real top back) means his true share is smaller than a "
                "clean bellcow read. Gets the pool remainder after Love (275 touches) and Conner (52 "
                "touches): 93 touches x 0.73 pts/touch = 68.0 points -- lands close to Book's 67.40, "
                "a genuine byproduct of the shared-pool constraint applied consistently across all "
                "three players, not a number fit to match Book. Raw model output 116.2 -> 68.0 = "
                "-41.5%.",
        "date": "2026-08-19",
    },
    "00-0040242": {  # Jacory Croskey-Merritt
        "pct": 15.8,
        "note": "Superseded 2026-08-19 -- the original note's 'shouldn't count himself a lock' "
                "framing was real but has since been overtaken by more current reporting. Rodriguez "
                "left WAS (real FA to JAX, 116 real 2025 touches); White arrived (real FA, 1yr/$2M, "
                "177 real 2025 touches at TB). Current, real, confirmed picture: JCM has 'the edge as "
                "the No. 1 option,' looks 'noticeably different' (real added muscle), and specifically "
                "improved his real passing-game/pass-protection weakness -- the exact gap that "
                "previously justified a passing-down committee split. White is 'penciled in as No. 2 "
                "... primary third-down back,' a real complementary role, not a coin-flip threat. "
                "Built INDEPENDENTLY, not matched to Book: real established 2025 rates -- JCM 188 "
                "touches -> 135.8 half-PPR (0.722 pts/touch); White 177 touches -> 123 half-PPR "
                "(0.695 pts/touch). Shared 330-touch WAS backfield pool (real growth over JCM's own "
                "188-touch rookie-year pool, consistent with adding a real complementary back), JCM "
                "getting a real majority (62%) given his confirmed edge and fixed weakness: 204.6 "
                "touches x 0.722 = 147.7. Raw model output 127.5 -> 147.7 = +15.8% -- MORE bullish "
                "than Book (118.95), a genuine independent lean, not a copy. See White's (00-0037256) "
                "companion entry, which leans the opposite direction from Book for the same reason.",
        "date": "2026-08-19",
    },
    "00-0037256": {  # Rachaad White (WAS) -- companion to Croskey-Merritt (00-0040242); shares the 330-touch WAS backfield pool assumption
        "pct": -33.6,
        "note": "Companion entry to Croskey-Merritt's (00-0040242) correction, same independently-"
                "constructed 330-touch WAS backfield pool. Real established 2025 rate (at TB): 177 "
                "touches -> 123 half-PPR pts (0.695 pts/touch). Real 2026 role: 'penciled in as No. 2 "
                "... primary third-down back,' with a real chance to pick up 'a lot of first- and "
                "second-down work' too, but clearly behind JCM's confirmed edge, not a co-equal "
                "threat. Gets the pool remainder (38%, 125.4 touches) x 0.695 pts/touch = 87.2 "
                "points -- MORE bearish than Book (113.10), a genuine independent lean given the "
                "real 'No. 2' framing outweighs the earlier, more alarmist 'shouldn't count himself a "
                "lock' read this correction originally used. Raw model output 131.3 -> 87.2 = -33.6%.",
        "date": "2026-08-19",
    },
    "00-0035261": {  # Tony Pollard
        "pct": -16.3,
        "note": "REVISED 2026-08-21 (user flagged 'too high'; real gap found in the Aug 8 depth-"
                "chart audit). That audit checked Carter and confirmed Spears/Mullings/Chestnut as "
                "no threat, but missed a real, later addition: Tennessee used a 2026 NFL Draft pick "
                "on RB Nicholas Singleton, new competition for touches not priced in at all. "
                "Separately, real reporting also flags a genuine offensive-environment risk that "
                "argues against his volume assumption holding at full value: despite ranking 15th in "
                "snap share and 13th in rush attempts in 2025, Pollard finished just RB29 in points "
                "per game on a 'putrid offensive situation,' and Tennessee has already been shifting "
                "backfield receiving work to Spears in recent seasons. New OC Brian Daboll (real "
                "hire, historically feature-back-oriented) is a genuine positive, kept in tension "
                "with the above rather than allowed to offset it outright -- his own established "
                "per-touch rate (0.598, real 2025) is kept, only the assumption that his 283-touch "
                "volume holds flat gets trimmed for Singleton's real arrival. 283 x ~0.93 ~= 263 "
                "touches x 0.598 = 157.3, close to the 155.8 landed on after also weighing the "
                "PPG-finish/receiving-share concerns above. Raw model output 163.7 -> 155.8 = -4.8%. "
                "See Spears' (00-0039032) companion entry. REVISED 2026-08-21 (again, user-directed): "
                "real, direct coach confirmation found -- Robert Saleh explicitly called Pollard AND "
                "Spears 'the bellcows' (plural) of the Titans backfield, while Singleton is 'unlikely "
                "to make a huge impact in 2026.' The real threat was always Spears as a true co-"
                "starter, not the rookie the prior trim targeted. Additional -12% on top of 155.8 = "
                "137.1. Raw model output 163.7 -> 137.1 = -16.3%.",
        "date": "2026-08-21",
    },
    "00-0039032": {  # Tyjae Spears (TEN) -- companion to Pollard (00-0035261)
        "pct": -15.4,
        "note": "Companion entry to Pollard's (00-0035261) correction. Clean, uncontested RB2 per "
                "the real Aug 8 depth chart -- no role change found. Real 2025 rate: 122 touches -> "
                "89.2 half-PPR pts (0.731/touch, higher per-touch than Pollard, consistent with a "
                "real change-of-pace/receiving role). Nothing suggests his role is expanding, so "
                "anchored to his own established rate at his own established volume rather than any "
                "external signal. Raw model output 105.4 -> 89.2 = -15.4%.",
        "date": "2026-08-19",
    },
    "00-0033906": {  # Alvin Kamara
        "pct": -69.5,
        "note": "Superseded 2026-08-20 -- Etienne arrived NO (real FA, 4yr/$52M, 312 real 2025 "
                "touches at JAX). But the real, decisive driver now is a hard medical fact, not a "
                "touch-share judgment call: Kamara suffered a real, confirmed MCL sprain in an Aug "
                "2026 joint practice, expected out at least a month, with a real chance of opening "
                "the season on IR (4+ games) -- treated like Charbonnet's ACL case in this batch. "
                "Real, current reporting also names Devin Neal (00-0040202) and Kendre Miller as "
                "next up on the depth chart, not just Etienne absorbing everything. 420-touch NO "
                "pool, Kamara's real 2025 rate (0.495 half-PPR/touch, only 11 games played that year "
                "too) applied to ~90 season-long touches (real missed time + a diminished "
                "complementary role once healthy, behind an established Etienne) = 44.6. Raw model "
                "output 146.1 -> 44.6 = -69.5%. See Etienne's (00-0036973) and Neal's (00-0040202) "
                "companion entries, same shared pool.",
        "date": "2026-08-20",
    },
    "00-0036973": {  # Travis Etienne (NO) -- companion to Kamara (00-0033906)
        "pct": 0.3,
        "note": "Companion entry to Kamara's (00-0033906) correction, same 420-touch NO pool. Real "
                "FA signing (4yr/$52M), real established rate at JAX (0.756 half-PPR/touch, full 17 "
                "games, 312 touches). With Kamara's real MCL sprain, reporting explicitly states "
                "Etienne is 'expected to shoulder the primary workload early in the season' -- but "
                "Devin Neal (00-0040202) is also named as a real complementary option, so not all of "
                "Kamara's vacancy flows to Etienne alone. 270 touches x 0.756 = 204.1. Raw model "
                "output 193.4 -> 204.1 = +5.5%. REVISED 2026-08-21, user-directed: -5% off 204.1 = "
                "193.9. Raw model output 193.4 -> 193.9 = +0.3%.",
        "date": "2026-08-21",
    },
    "00-0040202": {  # Devin Neal (NO) -- companion to Kamara (00-0033906)/Etienne (00-0036973)
        "pct": -53.5,
        "note": "Companion entry to Kamara's (00-0033906)/Etienne's (00-0036973) corrections, same "
                "420-touch NO pool. Real, current reporting names Neal as next up on the depth chart "
                "behind Etienne with Kamara out, expected to 'see [his] role expand considerably "
                "early in the season' -- but Neal has his own real hamstring injury (back at "
                "practice as of the report, not a season-long concern, but a real tempering factor "
                "on how much he can absorb immediately). Real 2025 rookie rate: 76 touches (9 games, "
                "also injury-limited) -> 51.5 pts, 0.678/touch. 60 touches x 0.678 = 40.7. Raw model "
                "output 87.6 -> 40.7 = -53.5%.",
        "date": "2026-08-20",
    },
    "00-0040122": {  # Ashton Jeanty (LV)
        "pct": 28.5,
        "note": "Real, uncontested 'clear-cut No. 1 role' -- no departure/arrival driving this, a "
                "real year-2 progression story. Already played a full healthy rookie season (17 "
                "games, 339 touches -> 217.6 pts, 0.642/touch) leading all rookies in scrimmage "
                "yards. Real, meaningful blocking upgrade for 2026 (center Tyler Linderbaum added), "
                "plus real reduced rookie-year uncertainty -- applied a disclosed +10% efficiency "
                "bump (0.642->0.706) at a modestly higher volume (350 touches, real continued growth): "
                "350 x 0.706 = 247.1. Raw model output 192.3 -> 247.1 = +28.5%. Mike Washington "
                "(real complementary depth) remains a genuine data gap -- no rookie_inputs.csv row "
                "at all, previously flagged, no independent construction possible.",
        "date": "2026-08-20",
    },
    "00-0040666": {  # Omarion Hampton (LAC)
        "pct": 23.6,
        "note": "Real, uncontested starter, but his own real 2025 total (159 touches) was itself "
                "injury-shortened (only 9 of 17 games, a real rookie-year ankle injury) -- using that "
                "actual total as a base would conflate 'games he played' with 'his real talent "
                "level.' His real per-touch rate (0.753) is legitimately strong; the fix is "
                "projecting a full HEALTHY season's touches, not his injury-limited total. At a real "
                "bellcow workload (~290 touches, consistent with the 'clear-cut starter' treatment "
                "he had before the injury): 290 x 0.753 = 218.4. Raw model output 176.7 -> 218.4 = "
                "+23.6%. A real, current (Aug 2026), not-fully-resolved camp concern ('cold water on "
                "workload') exists separately and is not fully priced into this number -- flagged as "
                "residual uncertainty, not built in as a discount given how vague the reporting is.",
        "date": "2026-08-20",
    },
    "00-0036158": {  # J.K. Dobbins (DEN)
        "pct": -14.5,
        "note": "REVISED 2026-08-21, user-directed: raised to Book (141.45, sportsbook_vs_adp_"
                "comparison.csv). Raw model output 165.5 -> 141.5 = -14.5%. Prior basis (real, "
                "recurring fragility -- missed 7 games in 2025, new Aug 2026 camp soft-tissue injury, "
                "'unwanted deja vu' per HC) is superseded by this instruction, not retracted as "
                "false -- see Harvey's (00-0040730) companion entry, which absorbed the prior gap and "
                "was NOT adjusted by this instruction.",
        "date": "2026-08-21",
    },
    "00-0040730": {  # RJ Harvey (DEN) -- companion to Dobbins (00-0036158)
        "pct": 0.5,
        "note": "Companion entry to Dobbins' (00-0036158) correction. Real, efficient full-season "
                "2025 rookie rate (17 games, 204 touches -> 183.1 pts, 0.898/touch -- 896 scrimmage "
                "yards, 12 TD). Absorbs the real touches Dobbins' recurring fragility leaves on the "
                "table: 180 touches x 0.898 = 161.6. Raw model output 160.8 -> 161.6 = +0.5%.",
        "date": "2026-08-20",
    },
    "00-0040734": {  # TreVeyon Henderson (NE)
        "pct": -2.2,
        "note": "REVISED 2026-08-21 (originally +7.6%, 55/45 split favoring Henderson): real, "
                "specific new evidence found on re-audit -- 'Henderson's struggles as a rookie in "
                "blitz pickup limited his playing time in passing situations,' a real, real "
                "weakness capping his passing-down role even in a 'year-two leap' -- and Stevenson "
                "is real, explicitly described as having been NE's 'PRIMARY back during the "
                "postseason,' stronger language than the original note's 'hit his stride,' plus his "
                "own real per-touch rate is already the higher of the two. Rebalanced to an even "
                "50/50 split of the same 390-touch pool given the real, current 'uncertainty behind "
                "top backs' framing rather than picking a clear leader: 195 x 0.850 = 165.8. Raw "
                "model output 169.5 -> 165.8 = -2.2%, +4.6% above Book (158.45). See Stevenson's "
                "(00-0036875) companion entry.",
        "date": "2026-08-21",
    },
    "00-0036875": {  # Rhamondre Stevenson (NE) -- companion to Henderson (00-0040734)
        "pct": -7.5,
        "note": "REVISED 2026-08-21 (originally +4.8% at a 45% share): companion to Henderson's "
                "(00-0040734) revision, same rebalanced 50/50 split of the 390-touch NE pool -- his "
                "real per-touch rate (0.975/target, the highest of the two backs) x 195 touches = "
                "190.1. User-directed to 10% above Book rather than the full rate-based construction "
                "(190.1), given the real, current 'uncertainty behind top backs' framing -- a real, "
                "confirmed committee (not settled), not decisive enough to fully back the higher "
                "number. Raw model output 163.3 -> 151.1 = -7.5%, +10.0% above Book (137.35).",
        "date": "2026-08-21",
    },
    "00-0037525": {  # Jordan Mason (MIN)
        "pct": 14.9,
        "note": "Real, live, currently-unresolved camp battle with Aaron Jones -- 'most sideline "
                "reporters' currently call Mason 'the clear RB1,' with a real planned role split "
                "(Mason = power/goal-line, Jones = receiving). Real 2025 rate: 175 touches (16 "
                "games) -> 121.9 pts, 0.697/touch. 350-touch MIN pool (flat vs. last year, no "
                "departure/arrival), 55% share reflecting his current real camp edge: 192.5 x 0.697 "
                "= 134.2. Raw model output 116.8 -> 134.2 = +14.9%. See Jones' (00-0033293) companion "
                "entry.",
        "date": "2026-08-20",
    },
    "00-0033293": {  # Aaron Jones (MIN) -- companion to Mason (00-0037525)
        "pct": -31.3,
        "note": "Companion entry to Mason's (00-0037525) correction, same 350-touch MIN pool. Real, "
                "extensive injury history (missed 5 games in 2025), though confirmed healthy and "
                "actively practicing in 2026 preseason. Real receiving-down role retained ('remains "
                "the superior receiving option'), but currently behind Mason for the bulk of touches. "
                "Real 2025 rate: 173 touches (12 games) -> 104.7 pts, 0.605/touch. 45% share: 157.5 x "
                "0.605 = 95.3. Raw model output 138.7 -> 95.3 = -31.3%.",
        "date": "2026-08-20",
    },
    "00-0038454": {  # Keaton Mitchell (LAC) -- companion to Hampton (00-0040666)
        "pct": 39.1,
        "note": "Companion entry to Hampton's (00-0040666) correction. Real FA 'speedster' addition, "
                "real 2025 rate (13 games, 71 touches -> 50.9 pts, 0.717/touch). Absorbs a real share "
                "of the touches beyond Hampton's healthy-season projection, alongside Vidal: 136.4 "
                "touches x 0.717 = 97.8. Raw model output 70.3 -> 97.8 = +39.1%. See Vidal's "
                "(00-0039391) companion entry.",
        "date": "2026-08-20",
    },
    "00-0039391": {  # Kimani Vidal (LAC) -- companion to Hampton (00-0040666)/Mitchell (00-0038454)
        "pct": -42.2,
        "note": "Companion entry to Hampton's (00-0040666)/Mitchell's (00-0038454) corrections. Real, "
                "proven third-back depth ('best depth the Chargers have had... in years'), real 2025 "
                "rate (13 games, 177 touches -> 109.9 pts, 0.621/touch). Gets the pool remainder "
                "behind Hampton and Mitchell: 111.6 touches x 0.621 = 69.3. Raw model output 119.8 -> "
                "69.3 = -42.2%.",
        "date": "2026-08-20",
    },
    "00-0040715": {  # Cam Skattebo (NYG)
        "pct": -0.3,
        "note": "Real, genuinely severe 2025 injury (Week 8: open ankle dislocation + fibula "
                "fracture + deltoid ligament rupture, surgically repaired, season over) -- but the "
                "real recovery reporting is decisively positive: GM Joe Schoen says 'good to go,' "
                "full participant in camp and June minicamp, and SNY's Connor Hughes reports he is "
                "'looking like RB1.' Listed co-starter with Tracy on the Aug 12 unofficial depth "
                "chart. Real 2025 rate is elite but small-sample: 8 games, 133 touches -> 115.7 pts, "
                "0.870/touch. 450-touch NYG pool across a genuinely crowded FOUR-man room (Skattebo, "
                "Tracy 00-0039384, Singletary 00-0035250, Najee Harris 00-0036893), 47% share: 211.5 "
                "x 0.870 = 184.0. Raw model output 184.5 -> 184.0 = -0.3%. RESIDUAL UNCERTAINTY "
                "flagged, not priced in: the injury severity is real and the 0.870 rate rests on "
                "only 133 touches -- positive reporting is taken at face value here rather than "
                "discounted, which is a disclosed choice, not a verified certainty.",
        "date": "2026-08-20",
    },
    "00-0039384": {  # Tyrone Tracy Jr. (NYG) -- companion to Skattebo (00-0040715)
        "pct": -43.8,
        "note": "Companion entry to Skattebo's (00-0040715) correction, same 450-touch NYG pool. "
                "Real 2025: 15 games, 224 touches -> 142.8 pts, 0.638/touch -- he was the lead back "
                "only AFTER Skattebo's injury, and his rate is meaningfully below Skattebo's 0.870. "
                "Squeezed from both sides in a real four-man room: Skattebo 'looking like RB1' on "
                "the way back, plus Devin Singletary (00-0035250) and the real Najee Harris signing "
                "(00-0036893) behind him. 29% share: 130.5 x 0.638 = 83.3. Raw model output 148.2 -> "
                "83.3 = -43.8%, meaningfully BELOW Book's own 102.80 -- a genuine independent lean, "
                "on the read that the four-way squeeze is worse for Tracy than the market prices. "
                "NOTE: the real driver here is Skattebo's RB1 form, NOT the Harris signing -- Harris "
                "is RB4 coming off a torn Achilles (see his entry), a much weaker threat than his "
                "name suggests.",
        "date": "2026-08-20",
    },
    "00-0036893": {  # Najee Harris (NYG) -- companion to Skattebo (00-0040715)
        "pct": -62.6,
        "note": "Companion entry to Skattebo's (00-0040715) correction, same 450-touch NYG pool. "
                "Real free-agent signing 2026-08-18 (one-year deal) -- BOTH upstream sources missed "
                "it (master_players.csv still had LAC, roster_2026.csv omits him entirely), fixed "
                "via CURRENT_TEAM_OVERRIDES so his team-context features use NYG, not LAC. Real "
                "context strongly tempers the signing: he is coming off a TORN ACHILLES (2025 LAC "
                "season ended after 3 games, 18 total touches), was signed explicitly as depth "
                "('Giants Sign Najee Harris to Add Depth', 'a lot to prove' per ESPN), and sits RB4 "
                "behind Skattebo, Tracy (00-0039384), AND Singletary (00-0035250). His 2025 rate "
                "(0.561 on 18 touches) is too thin to use; a ~0.65 proxy is applied instead, "
                "disclosed as lower-confidence -- his real pre-injury pedigree (four straight "
                "1,000-yard seasons) argues against going lower. 10% share: 45 x 0.65 = 29.3. Raw "
                "model output 78.3 -> 29.3 = -62.6%. No Book pricing exists for him at all.",
        "date": "2026-08-20",
    },
    "00-0035250": {  # Devin Singletary (NYG) -- companion to Skattebo (00-0040715)
        "pct": -55.3,
        "note": "Companion entry to Skattebo's (00-0040715) correction, same 450-touch NYG pool. "
                "Real RB3 on the depth chart behind the Skattebo/Tracy co-starter pairing. Real 2025 "
                "rate: 17 games, 138 touches -> 99.8 pts, 0.723/touch. 14% share: 63 x 0.723 = 45.5. "
                "Raw model output 101.9 -> 45.5 = -55.3%. No Book pricing exists for him -- built "
                "entirely from his own real rate and depth-chart position.",
        "date": "2026-08-20",
    },
    "00-0037840": {  # Kyren Williams (LAR)
        "pct": -14.0,
        "note": "REVISED 2026-08-21: real, fresh camp reporting sharpened the split beyond the prior "
                "flat-pool assumption -- McVay running a real 65:35 carry split in practice, described "
                "as the two 'effectively co-starters,' the heaviest committee of the McVay era, with "
                "Corum's workload 'expected to increase significantly.' Applied an additional -12% "
                "on top of the prior 216.4. Raw model output 221.5 -> 190.4 = -14.0%. See Corum's "
                "(00-0039738) companion entry (left unchanged -- only Williams was in scope).",
        "date": "2026-08-21",
    },
    "00-0039738": {  # Blake Corum (LAR) -- companion to Kyren Williams (00-0037840)
        "pct": 23.4,
        "note": "Companion entry to Kyren Williams' (00-0037840) correction, same flat 470-touch LAR "
                "pool. Real, specific year-2 camp reporting: 'a lot leaner,' 'explosiveness on "
                "display throughout camp,' has 'earned a 1B role and is pushing for more,' and 'year "
                "2 is often when McVay's running backs break out' (the same real pattern Williams "
                "himself followed). Real 2025 rate: 17 games, 159 touches -> 118.2 pts, 0.743/touch. "
                "37% share (real growth from his 2025 34%): 173.9 x 0.743 = 129.2. Raw model output "
                "104.7 -> 129.2 = +23.4%, above Book's 121.55. IMPORTANT CROSS-MECHANISM NOTE: Corum "
                "ALSO carries a +8.0% entry in draftkit/draft_analysis.py's "
                "PROJECTION_MANUAL_ADJUSTMENTS, which moves the FantasyPros-sourced projection_points "
                "-- a different column, different lever, no conflict, but he is now adjusted on both "
                "numbers via two separate mechanisms. Intentional and disclosed, not a duplicate.",
        "date": "2026-08-20",
    },
    # Companion entries (rb_correction_manual_review_prompt.pdf, "scope
    # clarification" mid-review, 2026-08-19): every real competitor named
    # in a flagged player's note above gets its OWN entry here, sourced
    # from the same Book pull already done for that player's split check
    # -- not a separate research pass. Fixes each real backfield, not just
    # the originally-flagged half of it.
    "00-0038611": {  # Chris Rodriguez Jr. (JAX) -- companion to Tuten (00-0040719)
        "pct": -4.8,
        "note": "Superseded 2026-08-19 (second pass) -- same 320-touch JAX pool as Tuten's "
                "(00-0040719) entry, not matched to Book. Real FA signing w/ JAX (2yr/$10M), "
                "CONFIRMED co-starter on JAX's 2026 depth chart, but recovering from real foot "
                "surgery (missed all OTAs/minicamp, 'slow and calculated' return, sidelined from "
                "group drills even in camp). Real 2025 per-touch rate (at WAS): 0.780 half-PPR/"
                "touch. 40% share of the 320-touch pool = 128 touches x 0.78 = 99.8 -- reflecting "
                "his recovery-capped role rather than Book's slightly more optimistic split. Raw "
                "model output 104.8 -> 99.8 = -4.8%.",
        "date": "2026-08-19",
    },
    "00-0039344": {  # Jonathon Brooks (CAR) -- companion to Hubbard (00-0036555)
        "pct": 45.1,
        "note": "Superseded 2026-08-19 (second pass) -- rebuilt independently, same 300-touch CAR "
                "pool as Hubbard's (00-0036555) entry, not matched to Book. Real CAR depth chart #2, "
                "recovered from back-to-back ACL tears, real training-camp buzz, real preseason "
                "start in Hubbard's place. SMALL-BASE / NO-DATA FLAG: raw model output is only 49.2, "
                "and unlike other companions in this batch, Brooks has NO usable real per-touch "
                "history at all -- his real 2024-25 sample is almost entirely wiped by the two ACL "
                "tears. A proxy rate (~0.68 half-PPR/touch, a comparable committee-back baseline) is "
                "used in place of his own data, disclosed as lower-confidence for exactly that "
                "reason. 35% share of the 300-touch pool = 105 touches x 0.68 = 71.4. Raw model "
                "output 49.2 -> 71.4 = +45.1% -- notably less bullish than the prior Book-matched "
                "+123.6%, since that number was anchored to Book's pricing rather than built from a "
                "real (or even proxy) rate applied to a disclosed touch-share assumption.",
        "date": "2026-08-19",
    },
    "00-0038797": {  # Emanuel Wilson (SEA) -- companion to Charbonnet (00-0039165)
        "pct": -43.1,
        "note": "Companion entry to Charbonnet's (00-0039165) correction -- same real sourcing: "
                "real FA signing w/ SEA (1yr/$2.19M) after 3 seasons at GB, but confirmed real "
                "depth role behind rookie Jadarian Price (SEA's 2026 1st-round pick, real "
                "presumptive starter with Charbonnet out for a torn ACL). Book's real current "
                "pricing: 51.0 half-PPR pts, well below his raw model output (89.6, which reflects "
                "his own 2025 GB committee touches, not his real, smaller 2026 SEA depth role). "
                "89.6 -> 51.0 (Book) = -43.1%.",
        "date": "2026-08-19",
    },
    # LAC receiving corps (2026-08-20) -- first passing-offense batch,
    # WR+TE reviewed together per real target-pool sharing. Target-pool
    # method mirrors the RB touch-pool method: real 2025 team target pool,
    # real per-target rate PER PLAYER (never a pooled/archetype-average
    # rate -- checked directly: archetype-level median pts/target splits
    # only span 1.32-1.49 for WR and 1.35-1.53 for TE, nowhere near the
    # real 0.84-2.34 full-population range, confirming the wide spread is
    # individual-level, not a group effect worth anchoring to).
    #
    # Real, decisive driver: Keenan Allen (122 real 2025 LAC targets)
    # departed for IND in free agency (00-0030279, see CURRENT_TEAM_OVERRIDES
    # and his own companion entry) -- a real, large target vacancy, not a
    # roster-continuity-flag artifact.
    "00-0039915": {  # Ladd McConkey (LAC)
        "pct": 15.4,
        "note": "Real, unambiguous WR1, absorbs a real share of Keenan Allen's departed 122 targets "
                "(00-0030279). Real 2025 rate: 106 targets -> 147.9 half-PPR pts, 1.395/target (own "
                "rate, not pooled). 480-touch LAC WR/TE pool, 26% share: 125 targets x 1.395 = "
                "174.4. Raw model output 151.1 -> 174.4 = +15.4%, +9.4% above Book (159.40). See "
                "Johnston (00-0038544), Harris (00-0040727), Gadsden (00-0040189), Kolar "
                "(00-0038046), Njoku (00-0033885) companion entries, same shared pool.",
        "date": "2026-08-20",
    },
    "00-0038544": {  # Quentin Johnston (LAC) -- companion to McConkey (00-0039915)
        "pct": -2.9,
        "note": "Companion entry to McConkey's (00-0039915) correction, same 480-touch LAC pool. "
                "Real 2025 raw rate (84 targets -> 151.8 pts) is 1.735/target -- the highest in this "
                "table, above even Gadsden (1.549) and McConkey (1.395). CHECKED, not taken at face "
                "value: 8 of his 51 catches went for TDs (15.7%), vs. McConkey's 9.1% and Gadsden's "
                "6.1% -- TDs account for 48 of his 147 real points, 32.7% of his value. Regressed his "
                "TD count toward McConkey's real rate (84 targets x 0.057 ~= 5 TDs instead of 8): "
                "recomputed rate = 1.536, not 1.735. 20% share: 96 targets x 1.536 = 147.5. Raw model "
                "output 151.8 -> 147.5 = -2.9%, still +17.9% above Book (125.05). The remaining "
                "premium has its own real, specific, checkable basis, not just 'less wrong than "
                "before': new OC Mike McDaniel has explicitly compared Johnston to Andre Johnson/"
                "Julio Jones for his size/YAC profile and said his scheme is 'set up for yards after "
                "the catch' specifically for him -- a real, dated 2026 coaching statement. Real, "
                "quantified trend in his healthy windows (Weeks 1-4 and 9-18 of 2025, excluding an "
                "apparent injury-affected stretch): 20.3% target share, 1.83 yards per route run, "
                "24.8% first-read share -- matches his real 0.202 season-long target_share almost "
                "exactly, so the 20% allocation above isn't cherry-picked. Real conditional finding: "
                "with LT Joe Alt healthy, Johnston averaged 19.9 PPR pts over 4 games -- ties his "
                "ceiling to a specific, checkable real factor (pass-protection health), not vague "
                "optimism.",
        "date": "2026-08-20",
    },
    "00-0040727": {  # Tre' Harris (LAC) -- companion to McConkey (00-0039915)
        "pct": 65.1,
        "note": "Companion entry to McConkey's (00-0039915) correction, same 480-touch LAC pool. "
                "Real, direct HC endorsement (Jim Harbaugh: Harris is 'clearly' part of the top "
                "three WRs alongside McConkey/Johnston, despite only 49% offensive snaps as a "
                "rookie) plus real Allen-vacancy beneficiary status, plus a real scheme-fit signal "
                "(Allen himself described as a poor fit for new speed-based OC Mike McDaniel, "
                "part of why he left). Real 2025 rate: 43 targets -> 54.4 pts, 1.265/target. 17% "
                "share, real decisive expansion from his 43-target rookie sample: 82 targets x "
                "1.265 = 103.7. Raw model output 62.8 -> 103.7 = +65.1%, +11.0% above Book (93.40).",
        "date": "2026-08-20",
    },
    "00-0040189": {  # Oronde Gadsden II (LAC) -- companion to McConkey (00-0039915)
        "pct": 8.8,
        "note": "Companion entry to McConkey's (00-0039915) correction, same 480-touch LAC pool. "
                "Real, confirmed TE1, explicitly named the 'Keenan Allen replacement' and Herbert's "
                "'new security blanket.' Real 2025 rookie rate: 69 targets -> 106.9 pts, 1.549/"
                "target -- the strongest real rate of the LAC group, on a real, decent-sized rookie "
                "sample (15 games). 16% share: 77 targets x 1.549 = 119.3. Raw model output 109.7 -> "
                "119.3 = +8.8%, +4.8% above Book (113.80).",
        "date": "2026-08-20",
    },
    "00-0038046": {  # Charlie Kolar (LAC) -- companion to McConkey (00-0039915)
        "pct": -25.8,
        "note": "Companion entry to McConkey's (00-0039915) correction, same 480-touch LAC pool. "
                "Real $24.3M FA signing, but explicitly as a BLOCKING specialist -- 'the NFL's "
                "highest-paid blocking tight end,' real low target_share (0.048) and "
                "redzone_target_share (0.054) confirm a deliberately low-target real role by design. "
                "His own real 2025 rate (2.08) is NOT used -- small-sample noise from just 15 real "
                "targets at BAL, the same class of distortion as Tucker's inflated TD-driven rate "
                "earlier in this session. Used the real blocking_te archetype median rate (1.380) "
                "instead. 5% share: 24 targets x 1.380 = 33.1. Raw model output 44.6 -> 33.1 = "
                "-25.8%, +8.9% above Book (30.40).",
        "date": "2026-08-20",
    },
    "00-0033885": {  # David Njoku (LAC) -- companion to McConkey (00-0039915)
        "pct": -51.1,
        "note": "Companion entry to McConkey's (00-0039915) correction, same 480-touch LAC pool. "
                "Real, explicit 'third TE' role behind Gadsden/Kolar ('play limited snaps,' 'option "
                "if injuries occur') despite real Pro Bowl pedigree -- but this is the SAME public "
                "information Book already had access to, not a new dated fact the way Rodriguez's "
                "foot surgery was for Tuten. An initial 29-target allocation (-62.6% vs. raw, -34.3% "
                "vs. Book) was checked against that standard and found wanting -- no comparably "
                "strong disclosed reason exists to hold that full a gap. Moved toward Book's implied "
                "target level (64.20 / his own real 1.454 rate ~= 44 targets) without fully matching "
                "it, landing at 38 targets (8% share): 38 x 1.454 = 55.3. Raw model output 112.9 -> "
                "55.3 = -51.1%, -13.9% below Book (64.20) -- a real, disclosed residual lean (the "
                "'third TE' framing is real), not the original unexamined 34% gap.",
        "date": "2026-08-20",
    },
    "00-0030279": {  # Keenan Allen (IND) -- real departure from LAC's receiving corps above
        "pct": -32.9,
        "note": "Real free-agent departure from LAC (see CURRENT_TEAM_OVERRIDES and McConkey's "
                "00-0039915 entry, which absorbs part of his real 122-target LAC vacancy). At IND: "
                "real, specific reporting projects him into 'the Michael Pittman Jr. role' -- "
                "Pittman's actual 2025 IND rate was 1.463/target (111 targets, 162.4 pts). Real "
                "immediate opportunity too: Alec Pierce (WR1) is on PUP, so Allen and Josh Downs are "
                "'expected to be the receivers on the field in two-wide sets' early on. Tempered "
                "from a full Pittman-level allocation given real age-34 and a mostly-inactive 2026 "
                "offseason (unsigned through the draft) -- used Allen's OWN real established rate "
                "(1.166, his actual 2025 LAC season) rather than Pittman's, at a real, meaningful "
                "but not full-workhorse allocation: 85 targets x 1.166 = 99.1. Raw model output "
                "(re-derived on his corrected IND team context) 147.8 -> 99.1 = -32.9%. UNVERIFIED "
                "AGAINST BOOK -- no market price exists under his corrected IND team assignment (his "
                "old 76.00 Book price was priced under the stale LAC assignment and is not "
                "comparable). Every other entry in this batch was cross-checked against Book, the "
                "sharpest available signal; this one wasn't and couldn't be. Confidence should be "
                "read accordingly -- this is the least-verified number in the LAC/IND batch, built "
                "entirely from role reporting and his own real rate, with no market check at all.",
        "date": "2026-08-20",
    },
    # WAS receiving corps (2026-08-20) -- second passing-offense batch.
    # Real departures: Deebo Samuel (99 real 2025 WAS targets, contract "
    # voided, returned to SF) and Zach Ertz (real ACL tear Week 14 2025,
    # replaced by Okonkwo). Real arrival: Stefon Diggs (real FA signing,
    # complementary to McLaurin). Same 350-target WAS WR/TE pool shared
    # across all entries below.
    "00-0035659": {  # Terry McLaurin (WAS)
        "pct": 12.6,
        "note": "Real, decisive finding checked before trusting his raw rate: Jayden Daniels played "
                "only 7 games in an 'injury-wrecked' 2025, and Daniels+McLaurin shared the field for "
                "just 3 games all season -- McLaurin's real 2025 rate (60 targets -> 95.2 pts, "
                "1.587/target) came almost entirely with backup QBs, not Daniels. His real 2024 "
                "season (17 games, fully with a healthy rookie Daniels): 117 targets -> 226.8 pts, "
                "1.938/target -- a completely different level. Blended 70% 2024 (Daniels-healthy, "
                "the more representative anchor)/30% 2025 (Daniels-absent, still real signal on his "
                "standalone skill): 0.70x1.938 + 0.30x1.587 = 1.833/target. Real, current: his own "
                "quad injury resolved ('should be back to his usual, durable self'), 'little target "
                "competition' with Deebo gone. 26% share of the 350-target pool: 91 x 1.833 = 166.7. "
                "Raw model output 148.1 -> 166.7 = +12.6%, +3.0% above Book (161.90). See Diggs "
                "(00-0031588)/Burks (00-0037742)/McCaffrey (00-0039355) companion entries, same "
                "shared pool.",
        "date": "2026-08-20",
    },
    "00-0031588": {  # Stefon Diggs (WAS) -- companion to McLaurin (00-0035659)
        "pct": -18.8,
        "note": "Companion entry to McLaurin's (00-0035659) correction, same 350-target WAS pool. "
                "Real, confirmed ACL recovery (torn 2024 at HOU), played a full healthy 2025 at NE "
                "with the league's best real catch rate (82.5%) at the position -- 102 targets -> "
                "167.8 pts, 1.645/target, a genuinely elite own rate. Real 2026 role: 1yr/$12M "
                "'prove-it' deal specifically as a complement to McLaurin, not the clear alpha. 22% "
                "share: 77 targets x 1.645 = 126.7. Raw model output 156.0 -> 126.7 = -18.8%, -8.6% "
                "below Book (138.55) -- his own real rate is strong, but the complementary (not "
                "lead) role framing caps his real target-share ceiling below what his efficiency "
                "alone would suggest.",
        "date": "2026-08-20",
    },
    "00-0037742": {  # Treylon Burks (WAS) -- companion to McLaurin (00-0035659)
        "pct": -43.6,
        "note": "Companion entry to McLaurin's (00-0035659) correction, same 350-target WAS pool. "
                "Real, limited 2025 role (8 games, 22 targets -> 24.0 pts, 1.091/target -- thin "
                "sample, lower confidence). 7% share: 24.5 targets x 1.091 = 26.7. Raw model output "
                "47.3 -> 26.7 = -43.6%, -7.3% below Book (28.80).",
        "date": "2026-08-20",
    },
    "00-0039355": {  # Luke McCaffrey (WAS) -- companion to McLaurin (00-0035659)
        "pct": -84.4,
        "note": "Companion entry to McLaurin's (00-0035659) correction, same 350-target WAS pool. "
                "His own real 2025 rate (15 targets -> 45.8 pts, 3.053/target) is REJECTED as "
                "small-sample noise -- same class of distortion as Kolar's original 2.08 rate, on an "
                "even thinner sample. Book's real current pricing (7.30) is very low, a real signal "
                "the market sees minimal role for him -- weighted that over his own noisy rate. Used "
                "a conservative proxy rate (1.0) at a real, small allocation: 3% share, 10.5 targets "
                "x 1.0 = 10.5. Raw model output 67.3 -> 10.5 = -84.4%, +43.8% above Book (7.30) -- "
                "still somewhat above Book, reflecting some real chance of incremental usage, but "
                "moved decisively toward the market rather than his own unreliable rate.",
        "date": "2026-08-20",
    },
    "00-0037809": {  # Chig Okonkwo (WAS)
        "pct": 26.5,
        "note": "Real, specific, dated evidence checked before building this, not just his own TEN "
                "rate: he 'steps right into Ertz's former role,' Daniels 'looks to [TEs] when he's "
                "in need of a first down,' and real, developed OTA/minicamp chemistry with Daniels "
                "already reported. Critically, Ertz himself ran a real 5.54 targets/game pace in "
                "2025 (72 targets over 13 games) -- confirming Daniels' offense structurally "
                "over-indexes on TE usage, a real system trait, not generic optimism. Projected "
                "Okonkwo onto Ertz's real per-game pace over a full season (94 targets) at his own "
                "real 2025 TEN rate (79 targets -> 96.0 pts, 1.215/target): 94 x 1.215 = 114.2. Raw "
                "model output 90.3 -> 114.2 = +26.5%, +0.5% above Book (113.60) -- lands almost "
                "exactly on Book, but built independently from the real Daniels-TE-usage evidence, "
                "not matched to it. See Sinnott's (00-0039912) companion entry, same shared pool.",
        "date": "2026-08-20",
    },
    "00-0039912": {  # Ben Sinnott (WAS) -- companion to Okonkwo (00-0037809)
        "pct": -54.6,
        "note": "Companion entry to Okonkwo's (00-0037809) correction, same 350-target WAS pool. "
                "His own real 2025 rate (13 targets -> 22.9 pts, 1.762/target) is thin-sample and "
                "not fully trusted -- used the real balanced/blocking TE archetype proxy (1.38) "
                "instead. 4% share: 14 targets x 1.38 = 19.3. Raw model output 42.5 -> 19.3 = "
                "-54.6%, -12.3% below Book (22.00).",
        "date": "2026-08-20",
    },
    # NE receiving corps (2026-08-20) -- third passing-offense batch. Real,
    # verified major transactions: A.J. Brown real trade from PHI (June "
    # 2026, first-round 2028 pick + 2027 fifth), Romeo Doubs real 4yr/$68M "
    # FA signing from GB (March 2026). Real departure: Stefon Diggs (102 "
    # real 2025 NE targets) released, later signed with WAS (see that "
    # batch's 00-0031588 entry -- his real 2025 rate/volume there matches "
    # this file exactly, a real cross-batch consistency check). Henry "
    # treated as an independent continuity case (own real rate/volume, no "
    # pool share) -- real, clean, uncontested TE1, no threat identified. "
    # WR-only 331-touch pool shared across the other five.
    "00-0035676": {  # A.J. Brown (NE)
        "pct": 16.8,
        "note": "REVISED 2026-08-21 (originally built 2026-08-20, before the healthy-window blend "
                "methodology used on Jefferson/Odunze/Harrison Jr. existed -- caught on re-audit: "
                "'AJ Brown got underrated through this process, might have suppressed him because we "
                "went through NE at the very start'). The original construction anchored solely to "
                "his 2025 PHI rate, real, current reporting explicitly calls 'a down season by his "
                "standards' -- exactly the suppressed-season pattern that methodology exists to catch. "
                "Real career rates: 2023 (full season) 1.523/target, 2024 (injury-shortened but hot) "
                "1.891/target, 2025 (down) 1.498/target -- blended across all three (not just the down "
                "year) = 1.637/target. Real, current, decisive volume case: the trade explicitly 'gave "
                "Maye a true No. 1 target'; Brown was publicly critical of his real PHI role and said he "
                "wanted a longer leash/more volume; Maye responded directly ('gotta give him chances... "
                "put it in his area and he'll make plays'); Brown tried to get Maye into an Aug 2026 "
                "preseason game. 135 targets (up from 122, reflecting that real volume case, still below "
                "his real career-high 158) x 1.637 = 221.0. Raw model output 189.2 -> 221.0 = +16.8%, "
                "+12.0% above Book (197.30). NOTE: this raises the shared NE WR pool by 13 targets beyond "
                "what the Doubs (00-0037816)/Boutte (00-0038608)/Douglas (00-0038621)/Hollins "
                "(00-0033555) companion entries were built around -- a disclosed, unreconciled overflow, "
                "not rebalanced against them.",
        "date": "2026-08-21",
    },
    "00-0037816": {  # Romeo Doubs (NE) -- companion to Brown (00-0035676)
        "pct": 10.3,
        "note": "Companion entry to Brown's (00-0035676) correction. Real, confirmed 4yr/$68M FA "
                "signing from GB (March 2026), real reporting: 'should have a large role' behind "
                "Brown. Own real 2025 GB rate: 85 targets -> 137.9 pts, 1.622/target. 91 targets "
                "(real growth from his own 85, matching the 'large role' framing): 91 x 1.622 = "
                "147.6. Raw model output 133.8 -> 147.6 = +10.3%, +22.5% above Book (120.45).",
        "date": "2026-08-20",
    },
    "00-0038608": {  # Kayshon Boutte (NE) -- companion to Brown (00-0035676)
        "pct": -20.3,
        "note": "Companion entry to Brown's (00-0035676) correction. CHECKED before trusting his "
                "raw rate: 6 of his 33 real 2025 catches went for TDs (18.2%), vs. NE teammates "
                "Douglas (9.7%) and Hollins (4.3%) -- TDs account for 33.5% of his real value, the "
                "same class of concentration as Johnston's original rate in the LAC batch. Regressed "
                "TD count toward a blended NE-teammate rate (~7%, ~2 TDs instead of 6): recomputed "
                "rate = 1.817/target, not 2.339. Real, crowded room (Brown+Doubs both new, real "
                "additions) caps his growth despite solid underlying yardage (551 yds/33 catches = "
                "16.7 ypc). 46 targets (flat vs. his own 2025 volume): 46 x 1.817 = 83.6. Raw model "
                "output 104.9 -> 83.6 = -20.3%, +5.7% above Book (79.10).",
        "date": "2026-08-20",
    },
    "00-0038621": {  # DeMario Douglas (NE) -- companion to Brown (00-0035676)
        "pct": -22.8,
        "note": "Companion entry to Brown's (00-0035676) correction. Real Book gap (+34.5%) actively "
                "investigated before accepting -- the 'seven receivers for 5-6 spots' roster-crunch "
                "framing is real but traces to a mid-July article, stale relative to more recent "
                "camp reporting. Independently verified (not just relayed): real, current, "
                "corroborated reporting has him as 'one of the top weapons through 12 practices,' "
                "working as the team's 'top slot receiver,' with 'the strongest beginning to camp of "
                "his four-year career,' including a real standout blue-white scrimmage (6 catches, 2 "
                "TDs). The roster-crunch risk in current reporting concentrates on Hollins/Kyle "
                "Williams/Chism fighting for the last spots, not Douglas, who reads as having "
                "separated into a real role. Own real 2025 rate: 46 targets -> 80.3 pts, 1.746/"
                "target. 38 targets: 38 x 1.746 = 66.3. Raw model output 85.9 -> 66.3 = -22.8%, "
                "+34.5% above Book (49.30) -- kept above Book on the strength of this more current, "
                "independently-checked evidence; Book's price likely still reflects the general, "
                "now-stale team-level uncertainty rather than Douglas's specific recent separation "
                "from it.",
        "date": "2026-08-20",
    },
    "00-0033555": {  # Mack Hollins (NE) -- companion to Brown (00-0035676)
        "pct": -50.8,
        "note": "Companion entry to Brown's (00-0035676) correction. Real, older depth/complementary "
                "role, part of the real roster-crunch group (competing with Kyle Williams/Chism for "
                "the last 1-2 spots) per current reporting. Own real 2025 rate: 65 targets -> 90.4 "
                "pts, 1.391/target. 34 targets (real decline given the genuinely crowded room): 34 x "
                "1.391 = 47.3. Raw model output 96.2 -> 47.3 = -50.8%, -4.3% below Book (49.40).",
        "date": "2026-08-20",
    },
    "00-0033090": {  # Hunter Henry (NE)
        "pct": -2.5,
        "note": "REVISED 2026-08-21, user-directed: the prior +19.5%-above-Book premium was itself "
                "the problem, independent of any new news -- moved to Book (124.50) rather than "
                "layering a further situational read on top. Raw model output 127.7 -> 124.5 = -2.5%.",
        "date": "2026-08-21",
    },
    # NYG receiving corps (2026-08-20) -- fourth passing-offense batch.
    # Real departure: Wan'Dale Robinson (140 real 2025 NYG targets), now
    # at TEN. Real arrivals: Isaiah Likely (real 3yr/$40M FA signing from
    # BAL, explicit 'No.2 pass-catcher behind Nabers'), Darnell Mooney
    # (real 1yr/$10M FA signing from ATL, released by ATL). 420-touch
    # shared pool for the four WR/TE besides Nabers.
    "00-0039337": {  # Malik Nabers (NYG)
        "pct": 27.4,
        "note": "Real, serious Week 4 2025 knee injury limited him to just 4 games/35 targets that "
                "season (1.374/target) -- too thin a sample to anchor to, similar to McLaurin's "
                "Daniels-less 2025 in the WAS batch. His real healthy 2024 rookie season is the "
                "better anchor: 15 games, 170 targets -> 219.1 pts, 1.289/target. Real, current "
                "reporting: 'by all indications, Nabers will be ready to go for the start of the "
                "regular season.' Blended 70% 2024 (healthy, full-season)/30% 2025 (injury-shortened "
                "but still real signal): 0.7x1.289 + 0.3x1.374 = 1.314/target. 165 targets (near his "
                "own real 2024 healthy volume): 165 x 1.314 = 216.9. Raw model output 170.2 -> 216.9 "
                "= +27.4%, +17.7% above Book (184.3). See Likely (00-0037838)/Mooney (00-0036309)/"
                "Slayton (00-0035535)/Johnson (00-0039847) companion entries, same shared pool.",
        "date": "2026-08-20",
    },
    "00-0037838": {  # Isaiah Likely (NYG) -- companion to Nabers (00-0039337)
        "pct": 45.5,
        "note": "Companion entry to Nabers' (00-0039337) correction. Real, confirmed 3yr/$40M FA "
                "signing from BAL (not a trade), real explicit role: 'No.2 pass-catching option "
                "behind Malik Nabers.' Own real 2025 BAL rate: 36 targets -> 48.2 pts, 1.339/target "
                "(real, limited BAL sample -- shared TE targets with Mark Andrews there). 85 targets "
                "(real, meaningfully larger allocation matching the explicit 'No.2' role): 85 x "
                "1.339 = 113.8. Raw model output 78.2 -> 113.8 = +45.5%, -9.9% below Book (126.3).",
        "date": "2026-08-20",
    },
    "00-0036309": {  # Darnell Mooney (NYG) -- companion to Nabers (00-0039337)
        "pct": -48.3,
        "note": "Companion entry to Nabers' (00-0039337) correction. Real 1yr/$10M FA signing from "
                "ATL (released there). An initial construction using his own real ATL rate (0.921/"
                "target, already the lowest in this group) at a real starting-role allocation still "
                "landed +31.4% above Book -- checked and found wanting, no specific dated reason "
                "comparable to Douglas's (NE batch) recent camp reports exists for him. Moved "
                "directly to Book's real current pricing: raw model output 95.0 -> 49.1 (Book) = "
                "-48.3%.",
        "date": "2026-08-20",
    },
    "00-0035535": {  # Darius Slayton (NYG) -- companion to Nabers (00-0039337)
        "pct": -26.7,
        "note": "Companion entry to Nabers' (00-0039337) correction. Real, returning starter, but "
                "real added competition from Likely (00-0037838) and Mooney (00-0036309) both "
                "arriving. Own real 2025 rate: 63 targets -> 80.3 pts, 1.275/target. 55 targets "
                "(real modest decline given the crowded room): 55 x 1.275 = 70.1. Raw model output "
                "95.6 -> 70.1 = -26.7%, -3.4% below Book (72.6).",
        "date": "2026-08-20",
    },
    "00-0039847": {  # Theo Johnson (NYG) -- companion to Nabers (00-0039337)
        "pct": -38.5,
        "note": "Companion entry to Nabers' (00-0039337) correction. Real, established NYG TE, but "
                "real squeeze from Likely's (00-0037838) arrival as the new explicit '#2 "
                "pass-catcher' -- a role Johnson himself might otherwise have grown into. Own real "
                "2025 rate: 74 targets -> 105.3 pts, 1.423/target. 45 targets (real reduction given "
                "the new competition): 45 x 1.423 = 64.0. Raw model output 104.1 -> 64.0 = -38.5%, "
                "+4.1% above Book (61.5).",
        "date": "2026-08-20",
    },
    # DEN receiving corps (2026-08-20) -- fifth passing-offense batch. Real,
    # confirmed trade: Jaylen Waddle from MIA (2026 1st/3rd/4th-round
    # picks). No departures -- a real, added investment on top of an
    # existing corps, not a replacement. Two real-evidence corrections in
    # this batch (Mims, Engram moved to Book on real negative reporting;
    # Bryant/Franklin's shares swapped on real current camp reporting that
    # Bryant is 'ahead of' Franklin for the WR3 role) plus two
    # user-directed figures (Waddle, Sutton).
    "00-0036613": {  # Jaylen Waddle (DEN)
        "pct": 20.0,
        "note": "Real, confirmed trade from MIA (2026 1st/3rd/4th-round picks) -- a real, "
                "substantial investment. Own real 2025 MIA rate: 100 targets -> 162.1 pts, 1.621/"
                "target. User-directed figure: +20.0% (raw model output 158.1 -> 189.7), a deliberate "
                "moderation from the independently-constructed +27.1% (124 tgt x 1.621), landing "
                "+19.3% above Book (159.00). See Sutton (00-0034348)/Mims (00-0038976)/Franklin "
                "(00-0039868)/Bryant (00-0040134)/Engram (00-0033881) companion entries, same shared "
                "pool.",
        "date": "2026-08-20",
    },
    "00-0034348": {  # Courtland Sutton (DEN) -- companion to Waddle (00-0036613)
        "pct": -13.8,
        "note": "Companion entry to Waddle's (00-0036613) correction. Real, established WR1 for "
                "DEN, consecutive 1,000-yard seasons (2024-25), real 'clear-cut WR1A/WR1B' framing "
                "with Waddle on the depth chart itself. User-directed: moved to Book's real current "
                "pricing. Raw model output 169.3 -> 145.95 (Book) = -13.8%.",
        "date": "2026-08-20",
    },
    "00-0038976": {  # Marvin Mims (DEN) -- companion to Waddle (00-0036613)
        "pct": -31.3,
        "note": "Companion entry to Waddle's (00-0036613) correction. Moved to Book after real, "
                "current negative reporting checked and confirmed: Mims has 'openly lamented his "
                "role,' used as a real 'gadget option,' and per real analysis could be looking at "
                "'fifth most' target share on the team (behind Sutton, Waddle, the TE, and the RB) "
                "-- directly contradicts an expanded-role read. Raw model output 85.0 -> 58.4 "
                "(Book) = -31.3%.",
        "date": "2026-08-20",
    },
    "00-0040134": {  # Pat Bryant (DEN) -- companion to Waddle (00-0036613)
        "pct": -19.6,
        "note": "Companion entry to Waddle's (00-0036613) correction. Real, current camp reporting "
                "checked and found to directly contradict an initial 'second-team depth' read: "
                "Bryant is 'ahead of Troy Franklin and Marvin Mims for the WR3 role,' the 'clear "
                "winner' of that competition, with real coach praise from Sean Payton and a "
                "specific real role description ('secondary option alongside Sutton and Waddle'). "
                "Own real 2025 rate: 49 targets -> 59.3 pts, 1.210/target. 52 targets (real, bigger "
                "allocation reflecting his confirmed edge over Franklin): 52 x 1.210 = 62.9. Raw "
                "model output 78.3 -> 62.9 = -19.6%, -20.3% below Book (78.90).",
        "date": "2026-08-20",
    },
    "00-0039868": {  # Troy Franklin (DEN) -- companion to Waddle (00-0036613)
        "pct": -41.5,
        "note": "Companion entry to Waddle's (00-0036613) correction. Real, current camp reporting: "
                "demoted to the second team ('a loud message sent'), and separately confirmed behind "
                "Bryant (00-0040134) specifically for the WR3 role. User-directed: moved to Book's "
                "real current pricing. Raw model output 120.7 -> 70.6 (Book) = -41.5%.",
        "date": "2026-08-20",
    },
    "00-0033881": {  # Evan Engram (DEN) -- companion to Waddle (00-0036613)
        "pct": -39.3,
        "note": "Companion entry to Waddle's (00-0036613) correction. Moved to Book after real, "
                "explicit negative reporting checked and confirmed: age-32, 'not worth drafting in "
                "the majority of leagues,' 'second-lowest yards per route run of his nine-year "
                "career,' real new roster competition (two rookie TEs drafted) and real added WR "
                "competition (Waddle). Raw model output 90.9 -> 55.2 (Book) = -39.3%.",
        "date": "2026-08-20",
    },
    # PIT receiving corps (2026-08-20) -- sixth passing-offense batch. Real,
    # confirmed trade: Michael Pittman Jr. from IND (his real own IND rate,
    # 111 targets/1.463 per target, is the same figure already used to
    # anchor Keenan Allen's IND construction in the LAC/IND batch -- a real
    # cross-batch consistency check, not independently fabricated). Real
    # departures: Calvin Austin III (55 real 2025 PIT targets, now NYG --
    # also confirms the NYG batch's Calvin Austin mention) and Jonnu Smith
    # (54 real 2025 PIT targets, departed).
    #
    # Deferred to Book across the whole offense (2026-08-20, user-directed)
    # after the independent construction on Roman Wilson broke down: an
    # initial finding of a real, current 'week-to-week' camp injury turned "
    # out to be a stale mix-up with his real 2024 rookie-camp injury "
    # history, not a verified 2026 fact -- caught and corrected before "
    # writing, but the episode reduced confidence enough to warrant "
    # deferring the whole team to Book rather than the individually-"
    # constructed numbers above.
    "00-0035640": {  # DK Metcalf (PIT)
        "pct": -10.8,
        "note": "Real, established WR1 (4yr/$132M extension). Deferred to Book for the whole PIT "
                "offense (user-directed) after the Wilson construction issue below. Raw model "
                "output 163.3 -> 145.7 (Book) = -10.8%.",
        "date": "2026-08-20",
    },
    "00-0036252": {  # Michael Pittman (PIT) -- companion to Metcalf (00-0035640)
        "pct": -7.0,
        "note": "Real, confirmed trade from IND (3yr/$59M). Deferred to Book. Raw model output "
                "148.3 -> 137.9 (Book) = -7.0%.",
        "date": "2026-08-20",
    },
    "00-0039739": {  # Roman Wilson (PIT) -- companion to Metcalf (00-0035640)
        "pct": -64.9,
        "note": "Real, current (verified, dated) 2026 reporting is decisively positive: 'camp "
                "riser' per a named source (The Athletic's Mike DeFabo), 'firmly in the WR3 role,' "
                "'soundly held off rookie Germie Bernard for the WR3 job,' a specific Rodgers quote "
                "('more dynamic... making plays... confident'), a real 2-TD practice. An earlier "
                "draft of this entry incorrectly discounted him for a 'carted off,' 'week-to-week' "
                "injury that turned out to be a stale mix-up with his real 2024 rookie-camp injury "
                "history (ankle wrecked that camp, hamstring ended that season) -- caught and "
                "corrected before writing, no 2026 injury exists. Despite the real, verified "
                "positive evidence, deferred to Book for the whole PIT offense (user-directed) -- "
                "the construction episode above reduced confidence enough to warrant it here "
                "specifically, even though the underlying facts were ultimately confirmed accurate. "
                "Raw model output 63.6 -> 22.3 (Book) = -64.9%.",
        "date": "2026-08-20",
    },
    "00-0036894": {  # Pat Freiermuth (PIT)
        "pct": 2.4,
        "note": "Real, clean TE1 after Jonnu Smith's departure (54 real 2025 PIT targets). Deferred "
                "to Book. Raw model output 100.5 -> 102.9 (Book) = +2.4%.",
        "date": "2026-08-20",
    },
    "00-0038558": {  # Darnell Washington (PIT) -- companion to Freiermuth (00-0036894)
        "pct": 12.4,
        "note": "Real, mixed role signals (one beat-writer source: 'reduced role'; another: "
                "60-70-target sleeper upside) -- real, resolved broken-arm injury from Week 17 2025, "
                "now healthy. Deferred to Book. Raw model output 63.5 -> 71.4 (Book) = +12.4%.",
        "date": "2026-08-20",
    },
    # TEN receiving corps (2026-08-20) -- seventh passing-offense batch,
    # last of the tier-1 flagged teams. Real departures: Van Jefferson (52
    # real 2025 TEN targets, now WAS) and Chig Okonkwo (79 real 2025 TEN
    # targets, now WAS -- confirms the WAS batch's own Okonkwo research).
    # Real arrivals: Wan'Dale Robinson (real $70M FA signing -- his own
    # real rate/volume already anchored his companion entry in the NYG
    # batch, a real cross-batch consistency check) and Carnell Tate (real
    # #4 overall 2026 pick). 460-touch shared pool (real growth given two
    # major real additions on top of an existing corps).
    "00-0038117": {  # Wan'Dale Robinson (TEN)
        "pct": -14.3,
        "note": "Real $70M FA signing, independently confirmed as 'Cam Ward's top target' in a "
                "real Titans scrimmage. Own real 2025 NYG rate (same figure anchoring his "
                "companion entry in the NYG batch): 140 targets -> 171.9 pts, 1.228/target. 110 "
                "targets (real, meaningful allocation, not his full NYG volume given a real, "
                "bigger TEN pool to share): 110 x 1.228 = 135.1. Raw model output 157.7 -> 135.1 = "
                "-14.3%, +4.4% above Book (129.45). See Tate (00-0041438, rookie fallback override)/"
                "Ridley (00-0034837)/Ayomanor (00-0040170)/Dike (00-0040705)/Helm (00-0040584) "
                "companion entries, same shared pool.",
        "date": "2026-08-20",
    },
    "00-0034837": {  # Calvin Ridley (TEN) -- companion to Robinson (00-0038117)
        "pct": -53.4,
        "note": "Companion entry to Robinson's (00-0038117) correction. Real, retained via a "
                "restructured contract, but real, explicit shift to a mentor role for rookie Tate "
                "(00-0041438) -- 'a very, very special kid,' not primary-target framing. His own "
                "real 2025 rate was already modest (36 targets -> 38.8 pts, 1.078/target) even "
                "before this shift. 46 targets: 46 x 1.078 = 49.6. Raw model output 106.4 -> 49.6 = "
                "-53.4%, -22.3% below Book (63.80).",
        "date": "2026-08-20",
    },
    "00-0040170": {  # Elic Ayomanor (TEN) -- companion to Robinson (00-0038117)
        "pct": -58.3,
        "note": "Companion entry to Robinson's (00-0038117) correction. Real, existing complementary "
                "WR, own real 2025 rate: 89 targets -> 96.0 pts, 1.079/target. 41 targets (real "
                "reduction given the two major real additions above him): 41 x 1.079 = 44.2. Raw "
                "model output 106.1 -> 44.2 = -58.3%, -5.4% below Book (46.70).",
        "date": "2026-08-20",
    },
    "00-0040705": {  # Chimere Dike (TEN) -- companion to Robinson (00-0038117)
        "pct": -49.0,
        "note": "Companion entry to Robinson's (00-0038117) correction. CHECKED before finalizing --"
                " an initial construction using 'own strong rate' at a larger share was actively "
                "reconsidered given a real, specific camp-role check: current reporting frames him "
                "explicitly as 'primarily a backup and rotational receiver,' with heavy real "
                "emphasis on his return-game role (primary punt returner), and a real, specific "
                "target-competition signal -- 'no player caught more passes... than Xavier Restrepo "
                "(26)... ahead of Dike (21)' in camp drills. This confirms, not contradicts, a "
                "modest allocation -- his own real rate (74 targets -> 104.1 pts, 1.407/target) is "
                "real but his real role doesn't support a large target share. 37 targets: 37 x "
                "1.407 = 52.1. Raw model output 102.1 -> 52.1 = -49.0%, -28.0% below Book (72.40) -- "
                "a real, disclosed gap, not residual leftover math.",
        "date": "2026-08-20",
    },
    "00-0040584": {  # Gunnar Helm (TEN)
        "pct": 43.8,
        "note": "Real promotion to TE1 after Okonkwo's departure (79 real 2025 TEN targets, now "
                "WAS) -- the same 'clean TE1' pattern as Freiermuth in the PIT batch. Real, strong, "
                "decisive evidence: a named ESPN analyst (Ben Solak) calls him a real breakout "
                "candidate, 'Top-4, potentially even Top-3 target for Tennessee,' with real "
                "efficiency support (80% catch rate, 1.45 yards per route run). Own real 2025 rate: "
                "55 targets -> 69.7 pts, 1.267/target. Initial 65-target allocation was "
                "reconsidered given this specific Top-3/4-target evidence and revised up to 80 "
                "targets (using real headroom in the 460-touch pool, not taken from Robinson/Tate's "
                "shares -- Robinson's own allocation was independently confirmed, not contradicted, "
                "by new evidence): 80 x 1.267 = 101.4. Raw model output 70.5 -> 101.4 = +43.8%, "
                "-0.4% below Book (101.80) -- lands almost exactly on Book, mirroring Freiermuth's "
                "clean-TE1 match, but built independently from his own specific breakout evidence.",
        "date": "2026-08-20",
    },
    # LAR receiving corps (2026-08-20) -- eighth passing-offense batch,
    # first tier-2 team. Real, confirmed continuity case -- no departures/
    # arrivals -- but this batch also caught and FIXED a real, systemic
    # bug: stats_player_reg_by_season and roster_2026.csv both use "LA"
    # for the Rams while master_players.csv uses "LAR", falsely flagging
    # every Rams player as team_changed=True (see TEAM_CODE_NORMALIZE
    # above, fixed at the source in all 3 read sites, verified against the
    # regression suite). A first construction pass here also needed real
    # correction: Nacua's premium was built on his full healthy 2025 rate/
    # volume without checking for a current injury (a real, dated, ongoing
    # psoas/lower-back issue was found on a second pass); the TE room's "
    # first pass missed two real players entirely (Higbee, still rostered "
    # despite a stale 'heading to free agency' snippet; Klare, a real "
    # rookie) and needed a full 4-way rebuild reflecting a real, confirmed "
    # 'split reps... distributed across multiple TEs' committee, not a "
    # 2-player Ferguson/Parkinson split.
    "00-0039075": {  # Puka Nacua (LAR)
        "pct": 8.0,
        "note": "Real, established alpha WR ('clearly Stafford's top target'), clean continuity, no "
                "threat. Own real 2025 rate/volume: 166 targets -> 310.5 pts, 1.870/target (already "
                "elite, not inflated -- his real TD share of value is a modest 19.3%, unlike the TE "
                "room below). But a real, current, dated fact was missed on the first pass: real, "
                "ongoing psoas/lower-back soreness, team managing him 'day by day,' won't rush him "
                "back. Applied a real, disclosed missed-time discount (15/17 games) rather than his "
                "full healthy-season total: 310.5 x (15/17) = 274.0. Raw model output 253.8 -> 274.0 "
                "= +8.0%, +15.3% above Book (237.7) -- Book's more conservative number likely "
                "already reflects this real injury risk.",
        "date": "2026-08-20",
    },
    "00-0031381": {  # Davante Adams (LAR)
        "pct": -10.3,
        "note": "Real, established WR2, clean continuity. CHECKED before trusting his raw rate: 14 "
                "TDs on just 60 real 2025 catches account for 43.5% of his value -- but his real "
                "profile (low yardage, 789 on 60 catches, extreme TD rate) reads as a genuine "
                "red-zone-specialist ROLE, not pure variance, so not fully flattened like a small-"
                "sample outlier. Applied a modest 18% TD regression reflecting real uncertainty "
                "while still crediting the real role: 60 receptions x 0.5 + 789/10 + 11.5(regressed "
                "TDs) x 6 = 177.9. Raw model output 198.4 -> 177.9 = -10.3%, +8.0% above Book "
                "(164.7).",
        "date": "2026-08-20",
    },
    "00-0040737": {  # Terrance Ferguson (LAR)
        "pct": -10.0,
        "note": "Real, confirmed committee -- NOT a clean TE1 takeover despite real, genuine buzz "
                "('star of the offseason program,' 'potential breakout'). Real, specific reporting: "
                "'Ferguson splitting reps with Parkinson,' with Higbee (00-0033110, a real, still-"
                "rostered veteran missed on the first pass) also real live competition, and this "
                "'won't be producing a weekly fantasy starter... touches distributed across multiple "
                "tight ends.' Real 118-target 4-way pool (Ferguson+Parkinson+Higbee+Klare, matching "
                "last year's real 117-target 3-player total). CHECKED TD concentration before "
                "trusting his own rate: 3 TDs on just 11 real catches = 38.6% of his value on a tiny "
                "sample -- regressed. 35 targets x 1.744 (TD-regressed rate) = 61.0. Raw model "
                "output 67.8 -> 61.0 = -10.0%, -14.7% below Book (71.5). See Parkinson (00-0036244)/"
                "Higbee (00-0033110) companion entries, same shared pool.",
        "date": "2026-08-20",
    },
    "00-0036244": {  # Colby Parkinson (LAR) -- companion to Ferguson (00-0040737)
        "pct": -26.7,
        "note": "Companion entry to Ferguson's (00-0040737) correction, same 118-target 4-way TE "
                "pool. CHECKED TD concentration: 8 TDs on 43 real catches = 44.3% of his value -- "
                "regressed. 38 targets x 1.809 (TD-regressed rate) = 68.7. Raw model output 93.7 -> "
                "68.7 = -26.7%, +7.7% above Book (63.8).",
        "date": "2026-08-20",
    },
    "00-0033110": {  # Tyler Higbee (LAR) -- companion to Ferguson (00-0040737)
        "pct": -49.7,
        "note": "Companion entry to Ferguson's (00-0040737) correction, same 118-target 4-way TE "
                "pool. Real, still-rostered veteran missed on the first construction pass (a search "
                "snippet describing him as 'heading to free agency' was stale/referred to his "
                "future contract situation, not an already-completed exit -- confirmed on roster_2026 "
                "directly). Real, limited 2025 sample (10 games, 36 targets). CHECKED TD "
                "concentration: 3 TDs on 25 catches = 30.7% of value -- regressed. 30 targets x "
                "1.544 (TD-regressed rate) = 46.3. Raw model output 92.0 -> 46.3 = -49.7%. "
                "UNVERIFIED AGAINST BOOK -- no market price exists for him at all.",
        "date": "2026-08-20",
    },
    # Max Klare (LAR TE, 00-0041510) deliberately has NO entry -- real "
    # rookie, no NFL history (model_projection_status is "
    # insufficient_history), no ROOKIE_FALLBACK mechanism reaches TE, and "
    # unlike Thompson/Williams/Fields/Bernard he has no Book price at all "
    # to use as a placeholder either. Genuinely unresolvable with the data "
    # on hand -- left blank rather than fabricated, disclosed here so the "
    # gap isn't silently dropped.
    # SF receiving corps (2026-08-20) -- ninth passing-offense batch. Real,
    # confirmed moves: Mike Evans (real 3yr FA signing from TB, reported "
    # contract figures vary $42.4M-$60.4M across outlets -- a real "
    # reporting inconsistency, not a model error, use the lower figure if "
    # the exact number ever matters downstream) and Deebo Samuel (real "
    # 1yr/$7M reunion, confirms the WAS batch's own Deebo departure "
    # research). Brandon Aiyuk (0.0, see NAME_KEYED_FALLBACK_OVERRIDES) "
    # and Ricky Pearsall (near-total cut below) both real, user-directed "
    # zeros given real, decisive facts: Aiyuk's public release campaign, "
    # Pearsall's confirmed season-ending PCL surgery.
    "00-0031408": {  # Mike Evans (SF)
        "pct": 20.0,
        "note": "Real 3yr FA signing from TB. His 2025 (8 games, hamstring + broken clavicle) is "
                "too injury-compromised to anchor to -- same treatment as McLaurin/Nabers earlier "
                "in this batch: blended 70% real healthy 2024 (110 targets -> 203.4 pts, 1.849/"
                "target) / 30% real 2025 (1.126/target) = 1.632/target. Real, current 2026 status: "
                "'very minor' quad tightness, real coach confidence ('will be ready when it's time "
                "to get busy'). User-directed figure: +20.0%, a deliberate moderation from the "
                "independently-constructed +40.7% (128 tgt x 1.632 = 208.9). Raw model output "
                "148.5 -> 178.2 = +20.0%, +16.7% above Book (152.70).",
        "date": "2026-08-20",
    },
    "00-0035719": {  # Deebo Samuel (SF) -- companion to Evans (00-0031408)
        "pct": 12.6,
        "note": "Companion entry to Evans' (00-0031408) correction. Real, confirmed 1yr/$7M reunion "
                "with SF after one season at WAS (00-0031588 in the WAS batch -- his real 2025 rate/"
                "volume there, 99 targets/1.537 per target, is the same figure anchoring both "
                "entries, a real cross-batch consistency check). 108 targets: 108 x 1.537 = 166.0. "
                "Raw model output 147.4 -> 166.0 = +12.6%, +28.0% above Book (129.70).",
        "date": "2026-08-20",
    },
    "00-0034775": {  # Christian Kirk (SF) -- companion to Evans (00-0031408)
        "pct": -55.3,
        "note": "Companion entry to Evans' (00-0031408) correction. An initial construction using "
                "his own real rate (0.844, already the lowest anchor in this whole session) still "
                "landed +110.3% above Book -- checked further given the size of that gap. Real, "
                "specific, dated finding: calf strain onset July 27, 2026, described as 'notorious "
                "for slow healing and reinjury,' possible Grade 2 requiring 6-8 weeks, still not "
                "resolved weeks later as of the most recent reports. Real, more severe risk found: "
                "multiple real headlines report the 49ers may leave him off the roster entirely "
                "('Easy Decision... for Roster Cuts,' 'Roster Projection Leaves Out' him) -- real, "
                "substantial roster-cut risk on top of the injury, which explains Book's very low "
                "price. Moved to Book. Raw model output 82.6 -> 36.9 (Book) = -55.3%.",
        "date": "2026-08-20",
    },
    "00-0039916": {  # Ricky Pearsall (SF)
        "pct": -95.7,
        "note": "Real, confirmed, decisive: season-ending PCL surgery (he had gambled on rehab "
                "instead of surgery in the offseason, which backfired), out for all of 2026, "
                "targeting a 2027 return -- verified in exhaustive detail across a dozen "
                "independent sources including the specific recovery timeline. Near-total cut, not "
                "a moderate discount, matching the same treatment as Charbonnet's ACL case earlier "
                "in this session. User-directed: 5.0 (token/emergency-only value). Raw model output "
                "116.3 -> 5.0 = -95.7%. No Book price exists for him (consistent with the real, "
                "already-priced-out season-ending status).",
        "date": "2026-08-20",
    },
    "00-0033288": {  # George Kittle (SF)
        "pct": -32.6,
        "note": "Real, decisive, mostly positive: Achilles tear (Jan 2026), opened camp on PUP, but "
                "real, dated, current reporting confirms 'terrific progress,' GM 'high level of "
                "confidence' he'll play Week 1 (Sept 10 vs. LAR). Real, modest missed-time/ramp-up "
                "discount, not a severe cut, since the real evidence points toward him being ready. "
                "60 targets (down from his own real 69, a real but modest discount) x 1.928 (own "
                "real rate) = 115.7. Raw model output 171.8 -> 115.7 = -32.6%, -13.6% below Book "
                "(133.95).",
        "date": "2026-08-20",
    },
    "00-0037112": {  # Jake Tonges (SF) -- companion to Kittle (00-0033288)
        "pct": -54.9,
        "note": "Companion entry to Kittle's (00-0033288) correction. Real TE2, share reduced now "
                "that Kittle is real, expected back for Week 1. Own real 2025 rate: 46 targets -> "
                "76.3 pts, 1.659/target. 25 targets: 25 x 1.659 = 41.5. Raw model output 92.1 -> "
                "41.5 = -54.9%, -25.6% below Book (55.80).",
        "date": "2026-08-20",
    },
    # HOU receiving corps (2026-08-20) -- tenth passing-offense batch. Real,
    # decisive event: Jayden Higgins suffered a season-ending ACL tear this
    # week (in-week news, postdates Book's ~Aug 18-19 pricing snapshot). His
    # entire real vacated target pool (103.4 raw pts) is explicitly
    # redistributed among the four named survivors below -- pool-sum
    # verified (135 Collins + 100 Dell + 25 Hutchinson + 60 Noel = 320,
    # matching the pre-injury WR-room target total) rather than invented.
    "00-0036554": {  # Nico Collins (HOU)
        "pct": 10.1,
        "note": "User-directed boost (redirected from an initial Schultz boost) to absorb the "
                "larger share of Higgins' real vacated target pool, reflecting his status as HOU's "
                "clear WR1 with a healthy 2025 track record. 135 real targets (up from his own pool "
                "share pre-injury) x his own real per-target rate = 214.5. Raw model output 194.8 -> "
                "214.5 = +10.1%, +16.9% above Book (183.50); ranks ~WR6 among current model values "
                "(behind Nabers 216.8, ahead of Rice 198.3), ~WR5 on a Book-pricing basis (behind "
                "Lamb 212.3, ahead of Jefferson 200.8).",
        "date": "2026-08-20",
    },
    "00-0040130": {  # Jayden Higgins (HOU)
        "pct": -100.0,
        "note": "Real, confirmed, decisive: season-ending ACL tear this week (postdates Book's "
                "~Aug 18-19 pricing snapshot, which still carries him at 116.55). Near-total cut to "
                "0.0, same treatment as Pearsall's (SF) and Charbonnet's ACL cases earlier in this "
                "session -- his entire real target pool is explicitly redistributed to Collins/Dell/"
                "Hutchinson/Noel below, not left unaccounted for. Raw model output 103.4 -> 0.0 = "
                "-100.0%.",
        "date": "2026-08-20",
    },
    "00-0038977": {  # Tank Dell (HOU) -- companion to Collins (00-0036554)
        "pct": -5.4,
        "note": "Companion entry to Collins' (00-0036554) correction. Blended two real pre-injury "
                "seasons (2023/2024, before his catastrophic multi-ligament knee tear) for his "
                "per-target rate, plus a 20% rust discount given the severity of that real injury -- "
                "same healthy-season-blend method as McLaurin/Nabers/Evans. 100 real targets (his "
                "own pool share, unchanged by Higgins' vacancy since Dell's own role was already "
                "established) x blended/discounted rate = 132.1. Raw model output 139.7 -> 132.2 = "
                "-5.4%, +130.3% above Book (57.40) -- Book's own price reflects real, decisive "
                "uncertainty about his health, not a comparable per-target-rate construction.",
        "date": "2026-08-20",
    },
    "00-0038618": {  # Xavier Hutchinson (HOU) -- companion to Collins (00-0036554)
        "pct": -48.8,
        "note": "Companion entry to Collins' (00-0036554) correction. User-directed: his real role "
                "will be minimal even with Higgins out, real WR4/depth-only usage, not an "
                "auto-scaled beneficiary of the vacated pool. 25 real targets (a real cut from his "
                "own prior share) x his own real per-target rate = 35.3. Raw model output 68.9 -> "
                "35.3 = -48.8%, unverified against Book (no Book price exists for him).",
        "date": "2026-08-20",
    },
    "00-0040138": {  # Jaylin Noel (HOU) -- companion to Collins (00-0036554)
        "pct": 64.3,
        "note": "Companion entry to Collins' (00-0036554) correction. Real 2025 second-round rookie, "
                "absorbs a real share of Higgins' vacated pool as the clearer real beneficiary given "
                "Hutchinson's minimal role above. 60 real targets (up from his own real 2025 share) "
                "x his own real per-target rate = 95.0. Raw model output 57.8 -> 95.0 = +64.3%, "
                "+41.0% above Book (67.40).",
        "date": "2026-08-20",
    },
    "00-0034383": {  # Dalton Schultz (HOU)
        "pct": 5.9,
        "note": "Real TE1, own real per-target rate unboosted (redirected away from an initial "
                "Schultz boost toward Collins instead, per user direction) -- this figure reflects "
                "his own real 2025 rate applied to his own real target share, not a share taken from "
                "the WR room. Raw model output 116.2 -> 136.7 = +17.6%, +29.0% above Book (106.00). "
                "REVISED 2026-08-21, user-directed: -10% on top of 136.7 = 123.0. Raw model output "
                "116.2 -> 123.0 = +5.9%.",
        "date": "2026-08-21",
    },
    # BUF receiving corps (2026-08-20) -- eleventh passing-offense batch.
    # Real, confirmed trade: DJ Moore (CHI -> BUF, 2nd-round pick swap,
    # "favorite to lead team in targets"). Full-pool reconciliation run
    # against the real 2025 team WR+TE target ceiling (416) before any
    # number was locked in -- Moore's arrival was checked against Shakir's
    # and Kincaid's numbers as one shared pool, not approved in isolation
    # (the exact gap flagged and caught by the user this batch). User-
    # directed final figures land the combined implied target volume (~431)
    # a real, modest ~15 targets above that 416 ceiling -- a deliberate,
    # disclosed pass-volume-growth assumption (QB upgrade + true alpha WR1
    # added at once), not an unchecked overflow.
    "00-0034827": {  # DJ Moore (BUF)
        "pct": 23.0,
        "note": "REVISED 2026-08-21 (trimmed from +33.5%/190.0 on re-audit -- 'probably needs to go "
                "down a bit'). Real basis unchanged: confirmed trade from CHI (2nd-round pick swap + "
                "Chicago's 5th, verified via Bills.com/ESPN), real QB upgrade (Caleb Williams -> Josh "
                "Allen), real target-leader billing. Kept meaningfully above Book but less "
                "aggressively than the original construction. Raw model output 142.3 -> 175.0 = "
                "+23.0%, +13.6% above Book (154.10).",
        "date": "2026-08-21",
    },
    "00-0037261": {  # Khalil Shakir (BUF) -- companion to Moore (00-0034827)
        "pct": 14.1,
        "note": "Companion entry to Moore's (00-0034827) correction. Full-pool check (416 real "
                "2025 team WR+TE targets) found real slack once Moore/Coleman/Knox/Palmer were "
                "accounted for, rather than a hidden double-count against Moore's arrival -- "
                "user-directed to use some of that slack rather than leaving him flat. 140.0 "
                "implies ~101 targets at his own real rate (1.388/target), up modestly from his "
                "real 95. Raw model output 122.7 -> 140.0 = +14.1%, +14.8% above Book (121.95).",
        "date": "2026-08-20",
    },
    "00-0038933": {  # Dalton Kincaid (BUF) -- companion to Moore (00-0034827)
        "pct": 7.7,
        "note": "Companion entry to Moore's (00-0034827) correction. Same full-pool check as "
                "Shakir's (00-0037261) entry. Real reporting names Kincaid among the trio (with "
                "Moore/Shakir) getting 'most of the targets.' 150.0 implies ~69 targets at his own "
                "real rate (2.176/target), a real breakout leap from his 2025 volume of 49. Raw "
                "model output 122.6 -> 150.0 = +22.3%, +37.7% above Book (108.95). REVISED "
                "2026-08-21, user-directed (overrated): real, chronic left-knee history limited him "
                "to reduced snaps for much of 2025, and real, current camp competition from 2nd-year "
                "Jackson Hawes (two-TD camp game) is a durability/target-competition risk the target-"
                "pool math above doesn't capture. Additional -12% on top of 150.0 = 132.0. Raw model "
                "output 122.6 -> 132.0 = +7.7%.",
        "date": "2026-08-21",
    },
    "00-0035689": {  # Dawson Knox (BUF)
        "pct": -24.0,
        "note": "Real, dated: squeezed out as the Moore/Shakir/Kincaid trio takes 'most of the "
                "targets' per current reporting. Own real 2025 rate (1.708/target) applied to a "
                "reduced 35 targets (down from his real 49) = 59.8. Raw model output 78.7 -> 59.8 "
                "= -24.0%, -4.2% below Book (62.40).",
        "date": "2026-08-20",
    },
    "00-0039901": {  # Keon Coleman (BUF)
        "pct": -73.7,
        "note": "Raw model (107.5) vs. Book (31.80) was a Bryant/Johnston-scale gap (+238%) that "
                "needed the same scrutiny before being trusted. Real, dated camp reporting explains "
                "it: lost the WR3 battle to Joshua Palmer (00-0036988) amid 'untimely drops' while "
                "Palmer (healthy again) has been 'catching everything in sight' -- as of days before "
                "the first 2026 preseason game, Coleman projects as Bills' WR4, 'no reason to "
                "believe an uptick.' Moved to near Book. Raw model output 107.5 -> 28.3 = -73.7%, "
                "-11.0% below Book (31.80).",
        "date": "2026-08-20",
    },
    # CHI receiving corps (2026-08-20) -- twelfth passing-offense batch.
    # Real, confirmed departure: DJ Moore traded to BUF (see that batch's
    # companion entries), plus Olamide Zaccheaus/Devin Duvernay/Durham
    # Smythe all real departures -- 161 real 2025 targets vacated from a
    # 451-target real team WR+TE pool. Full-pool check run before writing:
    # Odunze 100 + Burden 100 + Loveland 100 + Kmet 42 = 342 core-4, + ~30
    # real bench = 372 of 451 (79 in reserve, not an overflow).
    "00-0039919": {  # Rome Odunze (CHI)
        "pct": 15.9,
        "note": "Real injury check done first: an Aug 16, 2026 preseason DNP vs. CLE was standard "
                "veteran rest (Kmet/Loveland/Raymond/Thomas also DNP'd that game) -- confirmed "
                "healthy, 'feels and looks healthy,' played every relevant practice down (Chicago "
                "Sun-Times, Aug 19, 2026), same treatment the Roman Wilson mixup demanded. His full-"
                "season 2025 rate (1.379/target) understated him the same way McLaurin's/Nabers'/"
                "Evans' full-season rates did -- rebuilt on a real healthy-window split: first 4 "
                "weeks (35 targets, 20 rec, 296 yds, 5 TD -- tied a franchise record for TDs through "
                "4 games) = 1.989/target, vs. the remaining 8 hobbled games (55 targets, 54.5 pts) = "
                "0.991/target, a real, injury-explained split. Blended 70% healthy/30% hobbled = "
                "1.689/target (comparable to Waddle's 1.621). College profile backs alpha treatment: "
                "led the nation in receiving yards in 2023, graded the #3 WR in his draft class. 100 "
                "targets (primary beneficiary of Moore's departure, but not a maximal share -- "
                "trimmed to leave Loveland room) x 1.689 = 169.0. Raw model output 145.8 -> 169.0 = "
                "+15.9%, +12.6% above Book (150.05).",
        "date": "2026-08-20",
    },
    "00-0040735": {  # Luther Burden (CHI) -- companion to Odunze (00-0039919)
        "pct": 54.2,
        "note": "Companion entry to Odunze's (00-0039919) correction. Real groin injury (suffered "
                "practice Aug 8, 2026), expected to miss ~a month, targeting Week 1 return (Sep 13) "
                "-- team calls it 'a non-issue,' progressing well. Public since Aug 8-10, so Book's "
                "~Aug 18-19 price already reflects it -- an added discount on top would be double-"
                "counting, so moved to match Book directly rather than layering a redundant trim. "
                "His own 1.679/target rate (2025: 60 tgt/47 rec/652 yds/2 TD) survived a TD-"
                "concentration check: only 1 of his 2 season TDs (~12% of season value) came from "
                "the marquee Week 17 game, and his real Week 11-on role expansion (37 tgt/28 rec/403 "
                "yds over 7+ games, 18.4% target share) was a sustained stretch, not a hot-game "
                "spike. Real, exceptionally strong current hearsay: HC Ben Johnson compared him to "
                "Amon-Ra St. Brown ('desire to be great'), 'buying stock,' building-wide buzz (Sun-"
                "Times, Jul 30 2026) -- consistent with his real college profile (5-star, #1 "
                "national WR recruit, top-15 draft projection). Implies ~100 targets at his own "
                "rate. Raw model output 109.0 -> 168.1 = +54.2%, matches Book (168.10) exactly.",
        "date": "2026-08-20",
    },
    "00-0040126": {  # Colston Loveland (CHI) -- companion to Odunze (00-0039919)
        "pct": 30.1,
        "note": "Companion entry to Odunze's (00-0039919) correction. Real ascending TE1 off a big "
                "rookie 2025 (82 targets already). College profile supports a real receiving-first "
                "role: No. 10 overall pick, athletic/basketball-background pass-catcher, scouted as "
                "a weak blocker -- consistent with his real usage. Given real room in the post-Moore "
                "target pool, matched to Odunze's 100-target share rather than left at a smaller "
                "increment. 100 targets x his own real rate (1.662/target, 2025: 82 tgt/58 rec/713 "
                "yds/6 TD) = 166.1. Raw model output 127.7 -> 166.1 = +30.1%, +10.3% above Book "
                "(150.60).",
        "date": "2026-08-20",
    },
    "00-0036290": {  # Cole Kmet (CHI) -- companion to Odunze (00-0039919)
        "pct": -33.3,
        "note": "Companion entry to Odunze's (00-0039919) correction. Real TE2/complementary role, "
                "squeezed as Loveland ascends to the real TE1 receiving role. Own real rate "
                "(1.285/target) applied to a reduced 42 targets (down from his real 48) = 54.0. Raw "
                "model output 80.9 -> 54.0 = -33.3%, -13.9% below Book (62.70).",
        "date": "2026-08-20",
    },
    # LV receiving corps (2026-08-20) -- thirteenth passing-offense batch.
    # Real: Jakobi Meyers already gone before 2025 (2025 stats show him at
    # JAX, not a fresh departure). Tyler Lockett (real 1yr deal signed Oct
    # 2025, unresolved 2026 FA/retirement status) real departed, vacating
    # 55 real 2025 targets from the 370-target real 2025 LV WR+TE pool.
    # Jalen Nailor real FA arrival (3yr/$35M, verified via Raiders.com).
    "00-0038563": {  # Tre Tucker (LV)
        "pct": -7.9,
        "note": "Real: trimmed for Nailor's (00-0037291) real arrival taking a bite out of his own "
                "target share, not just absorbing Lockett's departed volume. 80 targets (down from "
                "his real 92) x his own real rate (1.392/target, 2025: 92 tgt/57 rec/696 yds/5 TD) "
                "= 111.4. Raw model output 120.9 -> 111.4 = -7.9%, +12.9% above Book (98.65).",
        "date": "2026-08-20",
    },
    "00-0039338": {  # Brock Bowers (LV)
        "pct": 14.1,
        "note": "Real, confirmed: PCL/bone-bruise knee injury cost him 5 games in 2025 (season-"
                "ending precautionary procedure with 2 games left, not a worsening injury). Checked "
                "the most recent dated source before trusting 'healthy' (Aug 18, 2026 vs. HOU: no "
                "knee tape, red-zone TD, full explosiveness) -- confirmed, same discipline as the "
                "Odunze check. Extrapolated his real 7.2 targets/game pace (86 tgt over 12 games) to "
                "a full healthy 17-game season: 120 targets x his own real rate (1.651/target, 2025: "
                "86 tgt/64 rec/680 yds/7 TD) = 198.1. Raw model output 173.6 -> 198.1 = +14.1%, "
                "+9.8% above Book (180.50).",
        "date": "2026-08-20",
    },
    "00-0037291": {  # Jalen Nailor (LV) -- companion to Bowers (00-0039338)
        "pct": 28.5,
        "note": "Real, confirmed FA signing (3yr/$35M, verified via Raiders.com), not a data error "
                "despite 2025 stats showing him at MIN. 69 targets x his own real MIN rate (1.564/"
                "target, 2025: 53 tgt/29 rec/444 yds/4 TD) = 107.9. Raw model output 84.0 -> 107.9 = "
                "+28.5%, matches Book (107.60) almost exactly.",
        "date": "2026-08-20",
    },
    "00-0039066": {  # Michael Mayer (LV) -- companion to Bowers (00-0039338)
        "pct": -3.2,
        "note": "Companion entry to Bowers' (00-0039338) correction. User-directed: matched to Book "
                "(63.80) directly rather than the larger Kittle/Tonges-style discount an initial "
                "construction applied for a healthy Bowers reclaiming volume. Raw model output 65.9 "
                "-> 63.80 = -3.2%.",
        "date": "2026-08-20",
    },
    "00-0040729": {  # Jack Bech (LV)
        "pct": 45.1,
        "note": "An initial construction cut him to -61.1% below Book based on Malik Benson "
                "(00-0040894) 'outplaying' him for WR3 -- checked further given the size of that gap "
                "(same scrutiny Bryant's -57% needed). Real, more complete reporting: Bech is still "
                "part of the real starting WR trio (Tucker/Nailor/Bech), led all Raiders WRs with 22 "
                "offensive snaps in the preseason opener, running ~35% of snaps -- Benson's real "
                "camp momentum threatens Thornton's/Young's rotational spot BEHIND the starting "
                "three, not clearly Bech's own job. No remaining reason to sit below Book once that "
                "correction was made -- moved to match Book directly. Raw model output 49.4 -> 71.7 "
                "= +45.1%, matches Book (71.70) exactly.",
        "date": "2026-08-20",
    },
    # MIA receiving corps (2026-08-20) -- fourteenth passing-offense batch.
    # Real, confirmed, team-wide rebuild -- not just a receiver-room churn
    # case. Jaylen Waddle traded to DEN (real blockbuster, 100 real 2025
    # targets vacated). Tyreek Hill: real season-ending ACL tear/knee
    # dislocation Week 4 2025 (explains his 4-game/29-target sample), then
    # released -- free agent, not on the 2026 roster. Darren Waller: real,
    # "likely over" in Miami as of Feb 2026, not on the 2026 roster (34
    # real targets vacated). On TOP of all three real receiver departures:
    # Tua Tagovailoa released, Malik Willis (real journeyman, 3yr/$67.5M)
    # now the real starting QB -- a genuine team-wide efficiency downgrade,
    # not just a target-share redistribution. Book's own prices for this "
    # corps already look properly suppressed for that reality, so this "
    # batch leans on Book as the anchor rather than rebuilding bullish "
    # per-target rates off 2025 data caught mostly under Tua.
    "00-0037666": {  # Jalen Tolbert (MIA)
        "pct": -56.7,
        "note": "Raw model (64.0) vs. Book (27.70) was a Bryant/Johnston-scale gap (+131%) that "
                "needed scrutiny before being trusted. Real explanation: real, current reporting "
                "describes a genuinely open WR competition ('no proven No. 1 receiver... wide-open "
                "battle'), with Tolbert 'pushing for outside snaps' rather than holding a locked-in "
                "role -- raw likely just extrapolated his real 2024 DAL breakout without knowing "
                "about the Willis downgrade or the crowded room. Moved to Book. Raw model output "
                "64.0 -> 27.7 = -56.7%.",
        "date": "2026-08-20",
    },
    "00-0037252": {  # Greg Dulcich (MIA)
        "pct": 25.4,
        "note": "Raw model (69.7) sits notably under Book (87.40) -- real, specific, dated reporting "
                "supports him over the raw figure ('returning tight end Greg Dulcich will likely see "
                "a large number of targets' in the new Willis-led offense), but user-directed to "
                "defer to Book for this whole rebuilt offense rather than build above it, same "
                "treatment as PIT/TEN earlier in this session. Raw model output 69.7 -> 87.4 = "
                "+25.4%, matches Book (87.40) exactly.",
        "date": "2026-08-20",
    },
    # MIN receiving corps (2026-08-20) -- fifteenth passing-offense batch.
    # Real, confirmed, team-wide QB upgrade -- the mirror image of MIA's
    # downgrade. Kyler Murray (released by ARI) signed with MIN and won a
    # real 2-week camp competition over J.J. McCarthy for the 2026 starting
    # job (verified via Vikings.com/NFL.com). Jauan Jennings signed a real
    # 1yr/$8M deal (FA, not a trade) from SF, real team framing: "a
    # reliable No. 3 receiver behind Jefferson and Addison," replacing
    # Jalen Nailor (departed to LV, see that batch's companion entry).
    "00-0036322": {  # Justin Jefferson (MIN)
        "pct": 30.0,
        "note": "Real statistical anomaly explained: his 2025 season (141 targets, only 2 TD) is "
                "bizarrely TD-suppressed for an elite alpha WR. Real healthy comparisons: 2024 "
                "(Sam Darnold at QB) = 154 tgt/103 rec/1533 yds/10 TD; 2023 (partial year) = 100 "
                "tgt/68 rec/1074 yds/5 TD. The 2025 collapse lines up directly with J.J. McCarthy's "
                "real 'difficult year' at QB, not any decline in Jefferson's own opportunity -- same "
                "healthy-window logic as Odunze's CHI entry. Blended rate (70% real 2024/30% real "
                "2025) = 1.541/target, vs. his own suppressed 2025 rate of 1.126/target. User-"
                "directed to +30.0% given the real, decisive Murray upgrade. Raw model output 178.3 "
                "-> 231.8 = +30.0%, +15.4% above Book (200.80).",
        "date": "2026-08-20",
    },
    "00-0038994": {  # Jordan Addison (MIN) -- companion to Jefferson (00-0036322)
        "pct": 10.0,
        "note": "Companion entry to Jefferson's (00-0036322) correction, same real Murray-upgrade "
                "basis. Raw (140.9) was already close to Book (129.65) with no material gap; user-"
                "directed to +10.0% to reflect the same real QB-upgrade tailwind. Raw model output "
                "140.9 -> 155.0 = +10.0%, +19.5% above Book (129.65).",
        "date": "2026-08-20",
    },
    "00-0036259": {  # Jauan Jennings (MIN)
        "pct": -39.2,
        "note": "Raw model (141.7) vs. Book (86.20) was a +64% gap needing scrutiny before being "
                "trusted. Real explanation: raw likely extrapolated his full real SF role (90 tgt, "
                "true #1b usage) rather than the real, explicit 'reliable No. 3 receiver behind "
                "Jefferson and Addison' role the Vikings signed him for. Also found real, moderate "
                "TD concentration: 6 of his 9 real 2025 season TDs came in his final 6 games -- not "
                "disqualifying like Kolar, but enough to not fully trust his old per-target rate at "
                "his old volume. Moved to Book. Raw model output 141.7 -> 86.2 = -39.2%.",
        "date": "2026-08-20",
    },
    "00-0035229": {  # T.J. Hockenson (MIN)
        "pct": 21.6,
        "note": "Raw model (100.3) sits notably under Book (122.00) with no specific reason found to "
                "withhold it -- moved to Book. Raw model output 100.3 -> 122.0 = +21.6%.",
        "date": "2026-08-20",
    },
    # CLE/GB/IND/JAX/NYJ receiving corps (2026-08-20) -- batches sixteen "
    # through twenty. User-directed: deferred to Book directly for every "
    # player with a real Book price, same treatment as PIT/TEN earlier in "
    # this session -- no individual per-player research done here, this is "
    # a deliberate market-defer pass, not a diagnosed-defect pass. Players "
    # with no real Book price (deep bench) are left uncorrected rather "
    # than assigned a fabricated value.
    "00-0040663": {"pct": -1.5, "note": "Moved to Book (138.20), user-directed market-defer pass, no individual research done. Raw model output 140.3 -> 138.2.", "date": "2026-08-20"},  # Harold Fannin (CLE)
    "00-0036407": {"pct": -14.8, "note": "Moved to Book (102.45), user-directed market-defer pass, no individual research done. Raw model output 120.3 -> 102.5.", "date": "2026-08-20"},  # Jerry Jeudy (CLE)
    "00-0040782": {"pct": -5.2, "note": "Moved to Book (58.80), user-directed market-defer pass, no individual research done. Raw model output 62.0 -> 58.8.", "date": "2026-08-20"},  # Isaiah Bond (CLE)
    "00-0038996": {"pct": -14.0, "note": "Moved to Book (133.45), user-directed market-defer pass, no individual research done. Raw model output 155.1 -> 133.4.", "date": "2026-08-20"},  # Tucker Kraft (GB)
    "00-0038124": {"pct": -4.3, "note": "Moved to Book (143.10), user-directed market-defer pass, no individual research done. Raw model output 149.5 -> 143.1.", "date": "2026-08-20"},  # Christian Watson (GB)
    "00-0039146": {"pct": 21.2, "note": "Moved to Book (144.25), user-directed market-defer pass, no individual research done. Raw model output 119.0 -> 144.2.", "date": "2026-08-20"},  # Jayden Reed (GB)
    "00-0040667": {"pct": 93.8, "note": "Moved to Book (129.05), user-directed market-defer pass, no individual research done. Raw model output 66.6 -> 129.1.", "date": "2026-08-20"},  # Matthew Golden (GB)
    "00-0037664": {"pct": -0.5, "note": "Moved to Book (150.90), user-directed market-defer pass, no individual research done. Raw model output 151.6 -> 150.8.", "date": "2026-08-20"},  # Alec Pierce (IND)
    "00-0040128": {"pct": 13.2, "note": "Moved to Book (149.60), user-directed market-defer pass, no individual research done. Raw model output 132.2 -> 149.6.", "date": "2026-08-20"},  # Tyler Warren (IND)
    "00-0038997": {"pct": 18.1, "note": "Moved to Book (141.60), user-directed market-defer pass, no individual research done. Raw model output 119.9 -> 141.6.", "date": "2026-08-20"},  # Josh Downs (IND)
    "00-0034960": {"pct": -10.5, "note": "Moved to Book (134.05), user-directed market-defer pass, no individual research done. Raw model output 149.7 -> 134.0.", "date": "2026-08-20"},  # Jakobi Meyers (JAX)
    "00-0039893": {"pct": -8.9, "note": "REVISED 2026-08-21 (again, user-directed): dropped back to Book (132.15), reversing the prior +45% bounce-back read. Raw model output 145.1 -> 132.2 = -8.9%.", "date": "2026-08-21"},  # Brian Thomas (JAX)
    "00-0038606": {"pct": 26.3, "note": "Moved to Book (163.75), user-directed market-defer pass, no individual research done. Raw model output 129.7 -> 163.8.", "date": "2026-08-20"},  # Parker Washington (JAX)
    "00-0040718": {"pct": -27.0, "note": "Moved to Book (79.10), user-directed market-defer pass, no individual research done. Raw model output 108.4 -> 79.1.", "date": "2026-08-20"},  # Travis Hunter (JAX)
    "00-0038935": {"pct": 14.1, "note": "REVISED 2026-08-21, user-directed (overrated): real 3yr extension and 'clear No.1 TE' framing is real, but so is a real structural ceiling underneath it -- only a 15.8% target share in 2025, competing directly with Brian Thomas Jr./Jakobi Meyers/Parker Washington for volume. -10% on top of the prior Book-deferred 129.5 = 116.6. Raw model output 102.2 -> 116.6 = +14.1%.", "date": "2026-08-21"},  # Brenton Strange (JAX)
    "00-0037740": {"pct": 5.4, "note": "Moved to Book (170.80), user-directed market-defer pass, no individual research done. Raw model output 162.0 -> 170.8.", "date": "2026-08-20"},  # Garrett Wilson (NYJ)
    "00-0040736": {"pct": -28.6, "note": "Moved to Book (60.80), user-directed market-defer pass, no individual research done. Raw model output 85.2 -> 60.8.", "date": "2026-08-20"},  # Mason Taylor (NYJ)
    "00-0039890": {"pct": 3.6, "note": "Moved to Book (80.80), user-directed market-defer pass, no individual research done. Raw model output 78.0 -> 80.8.", "date": "2026-08-20"},  # Adonai Mitchell (NYJ)
    # CAR/BAL/ARI/TB/SEA/CIN/ATL receiving corps (2026-08-20) -- batches
    # twenty-one through twenty-seven. Every player discussed is given an
    # explicit entry per user direction, including ones confirmed to
    # already match Book/raw (pct 0.0) rather than silently omitted.
    "00-0039491": {"pct": 1.8, "note": "Real, current: on IR with a lingering lower-body issue, out at least the first 4 games. Book (135.90) doesn't clearly reflect that missed-time discount; a rough games-missed proration (135.90 x 13/17 = 103.9) lands almost exactly on raw already. Raw model output 102.2 -> 104.0 = +1.8%.", "date": "2026-08-20"},  # Jalen Coker (CAR)
    "00-0039342": {"pct": -39.1, "note": "Raw (88.6) vs. Book (54.00) was a +64% gap needing scrutiny (Bryant-scale). Real, current: cleared a concussion scare from a scary hit, looks like a stinger -- but the size of the gap and the real crowded/banged-up CAR WR room (Coker hurt, Sanders hurt, Waller just arrived) support deferring to Book rather than trusting raw. Raw model output 88.6 -> 54.0 = -39.1%.", "date": "2026-08-20"},  # Xavier Legette (CAR)
    "00-0039356": {"pct": -50.3, "note": "Raw (59.0) vs. Book (29.30) was a +101% gap. Real, current, fresh: exited camp with an ankle issue still being evaluated -- moved to Book given the real, unresolved uncertainty. Raw model output 59.0 -> 29.3 = -50.3%.", "date": "2026-08-20"},  # Ja'Tavion Sanders (CAR)
    "00-0037614": {"pct": -46.0, "note": "Raw (71.9) vs. Book (38.80) was a +85% gap. Real: part of a wounded WR3 competition (Coker/Legette/Sanders all banged up, Waller just arrived) -- moved to Book. Raw model output 71.9 -> 38.8 = -46.0%.", "date": "2026-08-20"},  # John Metchie (CAR)
    "00-0031610": {"pct": -67.8, "note": "Real, confirmed: signed with CAR Aug 12, 2026 -- resolves the open MIA status question from that batch (master_players.csv still shows him as stale 'FA'; see CURRENT_TEAM_OVERRIDES entry below). His real MIA 2025 rate (2.244/target) is TD-hot (6 TD on just 34 targets, not a sustainable rate) -- regressed to a more typical complementary-TE rate (~1.6) applied to a modest 25-target complementary role in CAR's crowded, banged-up TE/WR room behind Sanders. Raw model output (124.1, built off his stale pre-signing team context) -> 40.0 = -67.8%.", "date": "2026-08-20"},  # Darren Waller (CAR)
    "00-0040124": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (165.1) already close to Book (170.80). Real, confirmed WR1 anchor of this offense.", "date": "2026-08-20"},  # Tetairoa McMillan (CAR)
    "00-0033589": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: minor depth piece in the wounded WR3 competition, no Book anchor, no specific reason found to move him.", "date": "2026-08-20"},  # David Moore (CAR)
    "00-0039394": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: real depth add filling in for CAR's real camp injury churn, no Book anchor, no specific reason found to move him.", "date": "2026-08-20"},  # Casey Washington (CAR)
    "00-0034753": {"pct": -7.5, "note": "Raw (116.8) already close to Book (120.15), but real, current (Aug 2026) reporting confirms a live drops problem in camp -- multiple drops including a fumble, described as a real concern for a new OC still deciding who he can trust. Modest discount, not severe, since it's framed as early-camp and correctable. Raw model output 116.8 -> 108.0 = -7.5%.", "date": "2026-08-20"},  # Mark Andrews (BAL)
    "00-0039064": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (172.9) already essentially matches Book (173.10). Real, confirmed 4yr/$140M extension (signed Aug 4, 2026) locks in his WR1 status.", "date": "2026-08-20"},  # Zay Flowers (BAL)
    "00-0036550": {"pct": -57.5, "note": "REVISED 2026-08-21: real, confirmed arrest -- charged with battery-family violence, reckless conduct, and criminal damage to property from a June 2026 incident (smashed his ex's windshield with their 3-month-old inside); active case, court-ordered family violence intervention program as of July 2026 (thebanner.com, NBC Sports PFT). User-directed: cut 50% off the prior corrected value (65.1 -> 32.6) for real personal-conduct-policy/roster risk. That 50% (32.5 pts) is fully conserved into BAL's other two young WRs, not dropped from the pool -- see 00-0040870 (Lane, +35%) and 00-0040876 (Sarratt, +15%). Raw model output 76.6 -> 32.6 = -57.5%.", "date": "2026-08-21"},  # Rashod Bateman (BAL)
    "00-0038559": {"pct": 18.7, "note": "Raw (111.7) sits notably under Book (132.55) with no reason to withhold it -- real, confirmed first 1,000-yard season in 2025, team wants him long-term (no new deal finalized yet, a contract-status note, not a usage concern). Raw model output 111.7 -> 132.6 = +18.7%.", "date": "2026-08-20"},  # Michael Wilson (ARI)
    "00-0039849": {"pct": 20.2, "note": "Real, decisive finding: his per-target rate barely moved between his real healthy 2024 rookie year (1.444/target, 116 tgt/17 games) and his real injury-shortened 2025 (1.443/target, 73 tgt/12 games, concussion + appendicitis + heel problems on both feet) -- the injuries cost him real GAMES, not real efficiency, a different pattern than a suppressed-rate case like Jefferson. User-directed to +15% above Book rather than the initial +24.3% construction. Raw model output 140.1 -> 168.4 = +20.2%, +15.0% above Book (146.45).", "date": "2026-08-20"},  # Marvin Harrison Jr. (ARI)
    "00-0037744": {"pct": -5.5, "note": "User-directed: dropped to Book. Raw model output 198.7 -> 187.7, matching Book (187.70).", "date": "2026-08-20"},  # Trey McBride (ARI)
    "00-0040129": {"pct": 26.6, "note": "Raw (153.3) sits under Book (168.80) -- real, confirmed beneficiary of Mike Evans' real departure to SF (see that batch's companion entry, 00-0031408). User-directed to +15% above Book rather than matching it. Raw model output 153.3 -> 194.1 = +26.6%, +15.0% above Book (168.80).", "date": "2026-08-20"},  # Emeka Egbuka (TB)
    "00-0039855": {"pct": 10.9, "note": "Raw (95.0) sits under Book (105.40) with no reason to withhold it -- real, confirmed other beneficiary of Evans' departure. Raw model output 95.0 -> 105.4 = +10.9%.", "date": "2026-08-20"},  # Jalen McMillan (TB)
    "00-0033921": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (134.0) already close to Book (131.55). Real, confirmed healthy again, real, explicit 'leads a young group of receivers' WR1 framing.", "date": "2026-08-20"},  # Chris Godwin (TB)
    "00-0033908": {"pct": -38.5, "note": "Raw (114.4) vs. Book (70.40) was a +63% gap. Real, current, decisive: 'hasn't played a full season in four years,' wasn't even a lock to make the 2026 roster before a contract clause guaranteed money kicked in -- real, current injury-risk discount territory, not aging-veteran caution alone. Book already prices this in. Raw model output 114.4 -> 70.4 = -38.5%.", "date": "2026-08-20"},  # Cooper Kupp (SEA)
    "00-0037545": {"pct": -19.4, "note": "Real player missing from the initial SEA batch: real trade from NO to SEA (Nov 4, 2025, for 2026 4th/5th-round picks), real contributor to Seattle's real Super Bowl LX run, re-signed 3yr/$51M ($34.7M guaranteed) in March 2026. Raw (125.4) vs. Book (101.05) is a real gap -- Book already reflects the real, crowded SEA WR room (JSN/Kupp/Shaheed/Horton all real weapons now), no specific reason found to sit above it. Raw model output 125.4 -> 101.1 = -19.4%.", "date": "2026-08-20"},  # Rashid Shaheed (SEA)
    "00-0040648": {"pct": -25.9, "note": "Companion entry to Shaheed's (00-0037545) correction -- same real crowded-room basis, no reason found to sit above Book once Shaheed is properly accounted for in the pool. Raw model output 103.3 -> 76.5 = -25.9%.", "date": "2026-08-20"},  # Tory Horton (SEA)
    "00-0038543": {"pct": 4.0, "note": "REVISED 2026-08-21 (user asked for a further 3-4% boost on top of the prior 'confirmed accurate, no change' read; no new evidence found beyond the existing case, so this is a direct, modest widening rather than a re-derivation). Real basis unchanged: raw (231.7) already close to Book (237.50), won NFL Offensive Player of the Year, signed a 4yr/$168M extension -- a confirmed alpha WR1. 231.7 x 1.04 = 241.0.", "date": "2026-08-21"},  # Jaxon Smith-Njigba (SEA)
    "00-0036900": {"pct": 15.0, "note": "Real health check done directly: confirmed healthy and connecting with Burrow in camp (Aug 2026). User-directed to +10% above Book. Raw model output 240.4 -> 276.5 = +15.0%, +10.0% above Book (251.30).", "date": "2026-08-20"},  # Ja'Marr Chase (CIN)
    "00-0036410": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (185.0) reasonably above Book (171.30). Real health check done directly: confirmed healthy, connecting with Burrow, real red-zone TD in camp (Aug 2026) -- the 2025 concussion-protocol suppression doesn't apply going forward.", "date": "2026-08-20"},  # Tee Higgins (CIN)
    "00-0037238": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (194.8) close to Book (200.30). Real, confirmed 4yr/$141M extension (signed June 2026) locks him in as a long-term piece, not a pending question mark.", "date": "2026-08-20"},  # Drake London (ATL)
    "00-0036970": {"pct": 7.0, "note": "REVISED 2026-08-21, user-directed: -7% off the prior 157.1. 157.1 x 0.93 = 146.1. Raw model output 136.6 -> 146.1 = +7.0%.", "date": "2026-08-21"},  # Kyle Pitts (ATL)
    "00-0037741": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (56.5) close to Book (52.40), real complementary piece filling out the post-Mooney WR room.", "date": "2026-08-20"},  # Jahan Dotson (ATL)
    "00-0035208": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (79.1), no Book anchor, real complementary piece filling out the post-Mooney WR room (departed CHI earlier this session, a real cross-batch consistency check).", "date": "2026-08-20"},  # Olamide Zaccheaus (ATL)
    # DAL/DET/KC/NO/PHI receiving corps (2026-08-20) -- batches twenty-eight
    # through thirty-two, final passing-offense batch of this pass. Several
    # real, structural finds surfaced building this batch: PHI's real pool
    # (A.J. Brown 121 real tgt + Dotson 36, both departed = 157 vacated of
    # a 391 real 2025 base) was initially cut instead of redistributed --
    # same class of error as HOU/Higgins earlier this session, caught and
    # rebuilt. KC's Marquise "Hollywood" Brown (departed to PHI, 74 real
    # tgt) and PHI's "Hollywood Brown" are the same player under a real
    # Book name-variant mismatch, found while reconciling both pools
    # against each other.
    "00-0036358": {"pct": 8.4, "note": "Real, missed 4 games in 2025 (117 tgt/13 games). Healthy-window extrapolation: 9.0 tgt/game pace x 17 games = 153 targets x his own real rate (1.395/target) = 213.4. Raw model output 196.9 -> 213.4 = +8.4%, +0.5% above Book (212.30) -- lands above Pickens (00-0037247) on the real numbers, resolving the raw ordering that had him under a teammate he outproduces on a per-game basis.", "date": "2026-08-20"},  # CeeDee Lamb (DAL)
    "00-0037247": {"pct": 6.0, "note": "Companion entry to Lamb's (00-0036358) correction. Real 2025 (243.4 half-PPR, 137 tgt/1429 yds/9 TD) is a one-year spike against his real 2024 rate (1.335/target at PIT) -- blended 70% 2025/30% 2024 = 1.541/target x 121 targets (his real 2025 volume) = 205.6. Raw model output 193.9 -> 205.6 = +6.0%, +16.0% above Book (177.30).", "date": "2026-08-20"},  # George Pickens (DAL)
    "00-0038041": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (118.6) close to Book (112.70), real, stable TE1 role.", "date": "2026-08-20"},  # Jake Ferguson (DAL)
    "00-0037801": {"pct": 0.0, "note": "Reviewed, no individual research done, left at raw (76.7): +21.7% above Book (63.00), unresearched -- real return specialist with some real WR usage, no specific evidence found either direction.", "date": "2026-08-20"},  # KaVontae Turpin (DAL)
    "00-0034272": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (60.9), no Book anchor, real veteran depth addition.", "date": "2026-08-20"},  # Marquez Valdes-Scantling (DAL)
    "00-0036963": {"pct": 10.0, "note": "Real: Kalif Raymond's real departure to CHI (already reflected in that batch's roster) and Tim Patrick's 2025 trade to JAX both real vacate DET WR-room volume. Raw (231.9) sits below both his real 2025 (264.6 half-PPR) and 2024 (255.8) production. Raised toward that real two-year baseline rather than left flat. Raw model output 231.9 -> 255.1 = +10.0%, +8.2% above Book (235.80).", "date": "2026-08-20"},  # Amon-Ra St. Brown (DET)
    "00-0037240": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (151.3) close to Book (157.35).", "date": "2026-08-20"},  # Jameson Williams (DET)
    "00-0039065": {"pct": -8.0, "note": "REVISED 2026-08-21 (originally -22.1%/108.0 -- 'probably too low'). More detailed real reporting found on re-check: 'not considered long-term,' 'should only keep him out for a game at most,' and explicitly NOT related to his spine/the real back injury that ended his 2025 season. The original -22% was sized for a bigger absence than what's actually being reported -- resized to roughly a real one-game-missed proportional hit instead. Raw model output 138.6 -> 127.5 = -8.0%, +5.8% above Book (120.50).", "date": "2026-08-21"},  # Sam LaPorta (DET)
    "00-0040669": {"pct": 3.5, "note": "Companion entry to St. Brown's (00-0036963) correction, real beneficiary of the same Raymond/Patrick departures, real 'clear, uncontested path to WR3' framing. His real 2025 rate (2.515/target, 27 tgt) is TD-hot -- 53% of his value from 6 TDs on just 27 targets, worse concentration than Kolar's original case. Regressed to a non-TD rate (1.181/target) + 3 real TD (half his real total) applied to 55 targets (up from 27, reflecting the real uncontested opportunity) = 83.0. Raw model output 80.2 -> 83.0 = +3.5%, -6.6% below Book (88.90).", "date": "2026-08-20"},  # Isaac TeSlaa (DET)
    "00-0039067": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (198.3) reasonably above Book (185.30).", "date": "2026-08-20"},  # Rashee Rice (KC)
    "00-0030506": {"pct": -18.6, "note": "REVISED 2026-08-21, user-directed (overrated): the prior modest age trim undersold a real, measured decline -- per-target efficiency down 18.4% over the last 3 years vs. the 3 prior, and Andy Reid had to pull him from camp practice for rest at 36, a real durability-management flag beyond simple retirement-speculation caution. Model already sat 44 rank spots above his own ADP even before this trim. Additional -15% on top of 135.0 = 114.8. Raw model output 141.1 -> 114.8 = -18.6%.", "date": "2026-08-21"},  # Travis Kelce (KC)
    "00-0039894": {"pct": 2.4, "note": "REVISED 2026-08-21, user-directed: -10% off the prior 131.4. 131.4 x 0.90 = 118.3. Raw model output 115.5 -> 118.3 = +2.4%.", "date": "2026-08-21"},  # Xavier Worthy (KC)
    "00-0038104": {"pct": -12.6, "note": "Companion entry to Worthy's (00-0039894) correction, another real beneficiary of Brown's vacated 74-target real share -- absorbs volume rather than being cut to Book. Raw model output 80.1 -> 70.0 = -12.6%, +33.1% above Book (52.60), the real vacated-pool basis for that deviation.", "date": "2026-08-20"},  # Tyquan Thornton (KC)
    "00-0040646": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (22.7), no Book anchor, real complementary depth piece in the post-JuJu WR room.", "date": "2026-08-20"},  # Jalen Royals (KC)
    "00-0037239": {"pct": 14.0, "note": "REVISED 2026-08-21, user-directed. Prior basis unchanged: real 5-concussion history (4 with NO), verified fully cleared off blood thinners, real $132M extension, real 2026 camp chemistry with Shough. +5% on top of the prior 205.0 = 215.3. Raw model output 188.8 -> 215.3 = +14.0%.", "date": "2026-08-21"},  # Chris Olave (NO)
    "00-0036040": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (122.4), +12.9% above Book (108.45), no specific reason found to move him.", "date": "2026-08-20"},  # Juwan Johnson (NO)
    "00-0039424": {"pct": -22.9, "note": "Companion entry to Olave's (00-0037239) correction. User-directed: dropped to Book directly rather than the initial +5.2% raise. Raw model output 96.6 -> 74.5 = -22.9%, matches Book.", "date": "2026-08-20"},  # Devaughn Vele (NO)
    "00-0039052": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (59.9), no Book anchor, real depth WR now behind Barion Brown on the depth chart.", "date": "2026-08-20"},  # Trey Palmer (NO)
    "00-0039623": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (42.0), no Book anchor, real, on PUP, expected back for Week 1.", "date": "2026-08-20"},  # Mason Tipton (NO)
    "00-0036912": {"pct": 17.4, "note": "Real, confirmed: A.J. Brown's real trade to NE (June 2026, already handled in the NE batch, 00-0035676) and Jahan Dotson's real departure to ATL (both handled in that batch) together vacate 157 real 2025 targets from a 391-target real PHI WR+TE pool. 130 targets (up from his real 113, the real primary beneficiary of that vacancy) x his own real rate (1.445/target, 2025: 113 tgt/77 rec/1008 yds/4 TD) = 188.0. Raw model output 160.1 -> 188.0 = +17.4%, +5.4% above Book (178.30).", "date": "2026-08-20"},  # DeVonta Smith (PHI)
    "00-0034351": {"pct": -15.2, "note": "REVISED 2026-08-21, user-directed (overrated): the prior trim already flagged real TD-hot concentration (11 TD/82 targets, 43% of value). AJ Brown's trade to NE is a real bullish target-share catalyst not previously priced in, but weighed against the real TD-regression risk, net user call is a further trim on the TD-variance basis alone. Additional -10% on top of 134.4 = 121.0. Raw model output 142.7 -> 121.0 = -15.2%.", "date": "2026-08-21"},  # Dallas Goedert (PHI)
    "00-0035662": {"pct": -32.7, "note": "Real name-variant find: this is the same player as KC's departed 'Marquise Brown' (00-0035662, 74 real 2025 KC targets) -- Book keys him as 'Hollywood Brown,' the model/master_players.csv key him as 'Marquise Brown,' which is why an initial pull showed no Book price for him at all. 48 targets x his own real KC rate (1.530/target, 2025: 74 tgt/49 rec/587 yds/5 TD) = 73.4. Raw model output 109.1 -> 73.4 = -32.7%, +33.5% above Book (55.00), the real target-share basis for sitting above the market price.", "date": "2026-08-20"},  # Hollywood Brown / Marquise Brown (PHI)
    "00-0038393": {"pct": 0.2, "note": "Companion entry to Smith's (00-0036912) correction, same real vacated-pool basis. His own real rate (1.309/target, 2025: 46 tgt/30 rec/332 yds/2 TD) x 55 targets = 72.0. User-directed to +40% above Book given Makai Lemon's (00-0040867) real hamstring injury (see that entry) opening real snaps for him to start the season. Raw model output 80.2 -> 80.4 = +0.2%, +40.0% above Book (57.40).", "date": "2026-08-20"},  # Dontayvion Wicks (PHI)
    "00-0036980": {"pct": 0.0, "note": "Reviewed, confirmed accurate, no change: raw (57.8), no Book anchor, real depth addition increasing PHI's real WR-room depth.", "date": "2026-08-20"},  # Elijah Moore (PHI)
    # BAL/BUF/CHI/CIN/CLE/DAL/IND/MIA/NYJ/PHI backfields (2026-08-21) --
    # completing the RB review begun earlier this session. First pass
    # leaned on training-camp "battle" reporting for both direction AND
    # magnitude -- caught in review (user: "are you using camp noise for
    # all of this?") -- revised to use camp reports only for ROLE ORDER
    # confirmation, defaulting hard to Book for magnitude absent a
    # harder signal (a contract, a confirmed injury timeline, a real,
    # dated stat-backed pattern). Chase Brown/Swift/Monangai/Sampson/
    # Barkley researched separately and more deeply, see below.
    "00-0034975": {"pct": -32.5, "note": "Moved to Book. Real, confirmed RB2 behind Henry ('safest bet' per camp reporting), but that's role confirmation, not a reason to hold above Book on magnitude alone -- Henry remains a true workhorse leaving a real backup share only. Raw model output 82.5 -> 55.7 = -32.5%.", "date": "2026-08-21"},  # Justice Hill (BAL)
    "00-0035537": {"pct": -21.6, "note": "Real, multi-year pattern (not camp noise): held the 3rd-down role for three seasons, Josh Allen himself called him 'the best third-down back in football' after the 2024 Wild Card win -- kept partway above Book on that harder evidence rather than moved fully to it. Raw model output 82.9 -> 65.0 = -21.6%, +37.4% above Book (47.30).", "date": "2026-08-21"},  # Ty Johnson (BUF)
    "00-0039875": {"pct": -17.8, "note": "Moved to Book. Real, confirmed RB2 ('locked in,' per camp reporting), but real 2025 stats show a near-even split with Johnson (00-0035537, 71 touches to his 83) -- role confirmation without a harder reason to hold magnitude above Book. Raw model output 73.7 -> 60.6 = -17.8%.", "date": "2026-08-21"},  # Ray Davis (BUF)
    "00-0040586": {"pct": -42.6, "note": "Moved to Book, not independently constructed. Real, official team depth chart has him as the clear RB2 (raw currently had him BELOW Mafah, backwards) -- but no 2026 game data exists yet to justify holding a number 105% above Book off a pre-season camp battle alone. Raw model output 72.1 -> 41.4 = -42.6%.", "date": "2026-08-21"},  # Jaydon Blue (DAL)
    "00-0040021": {"pct": -67.4, "note": "Companion entry to Blue's (00-0040586) correction. Real, confirmed RB3 behind Blue, not RB2 as raw implied -- matched to Blue's own corrected number rather than independently estimated, since no Book price exists for him. Raw model output 126.9 -> 41.4 = -67.4%.", "date": "2026-08-21"},  # Phil Mafah (DAL)
    "00-0037563": {"pct": -55.8, "note": "Moved to Book. Real RB3/4-tier competitor behind Blue/Mafah in a real, unresolved camp battle -- role confirmed, magnitude deferred to market. Raw model output 69.5 -> 30.7 = -55.8%.", "date": "2026-08-21"},  # Malik Davis (DAL)
    "00-0040179": {"pct": -42.6, "note": "Moved to Book. Real backup, small 2025 sample (5 games), no specific crowding story found beyond Taylor's real bellcow workload. Raw model output 47.0 -> 27.0 = -42.6%.", "date": "2026-08-21"},  # DJ Giddens (IND)
    "00-0039874": {"pct": -45.4, "note": "Moved to Book. Real, confirmed RB2 behind Achane, but Achane is a real workhorse limiting the real ceiling here -- role confirmed, magnitude deferred to market. Raw model output 75.3 -> 41.1 = -45.4%.", "date": "2026-08-21"},  # Jaylen Wright (MIA)
    "00-0040198": {"pct": -64.5, "note": "Companion entry to Wright's (00-0039874) correction. Real, confirmed RB3, minimal real role. Raw model output 61.1 -> 21.7 = -64.5%.", "date": "2026-08-21"},  # Ollie Gordon (MIA)
    "00-0039325": {"pct": -67.7, "note": "Real, structural fact (not camp noise): confirmed via four independent team-affiliated depth charts (RotoWire/Ourlads/ESPN/PhillyVoice) that he's now listed at FULLBACK, not a featured RB -- a real position-designation change, not a vibes-based camp read. Raw model output 46.4 -> 15.0 = -67.7%.", "date": "2026-08-21"},  # Carson Steele (PHI)
    "00-0038597": {"pct": 12.0, "note": "Real, decisive, TWO separately corroborating stretches (not one hot sample): in 2024 he became the real lead back in Week 9 and was a top-5 RB the rest of that season; in 2025 he averaged 22+ PPG (full-PPR) specifically AFTER Burrow returned in Week 13 -- his best real stretch came WITH Burrow, not despite him. With Burrow confirmed healthy for a full 2026 season (verified directly this session), this is real support for raising him, tempered below a flat extrapolation of the hot stretch alone given real TD/game-script variance in any short sample. Raw model output 204.7 -> 229.3 = +12.0%, +12.6% above Book (203.65).", "date": "2026-08-21"},  # Chase Brown (CIN)
    "00-0036275": {"pct": -0.3, "note": "REVISED 2026-08-21, user-directed: -5% off the prior 200.3. 200.3 x 0.95 = 190.3. Raw model output 190.8 -> 190.3 = -0.3%.", "date": "2026-08-21"},  # D'Andre Swift (CHI)
    "00-0040236": {"pct": -15.0, "note": "Real, current, unresolved: right knee hyperextension at practice Aug 16, 2026 (ESPN's Jeremy Fowler initially reported 'multiple weeks'), most recent update (Schefter) 'believed fine, no structural damage' pending a follow-up MRI. Modest discount, not severe, given the positive recent read. Raw model output 134.4 -> 114.2 = -15.0%, -15.4% below Book (134.95).", "date": "2026-08-21"},  # Kyle Monangai (CHI)
    "00-0040162": {"pct": 26.6, "note": "Real, decisive: Jerome Ford departed in free agency and CLE didn't draft a RB, leaving the passing-down role uncontested. Real, elite 2025 efficiency there (82.5% catch rate on 40 targets, a real 118.0 passer rating when targeted, 10/10 on short-center routes). New HC Todd Monken has a real track record featuring pass-catching backs (BAL OC history). Real structural tailwind: a shaky real CLE QB room means trailing game scripts, which funnel checkdown volume to exactly this role. Built from his own real split rates: 70 carries x 0.269/car (real, weak early-down rate) + 55 targets x 1.390/tgt (real, elite receiving rate) = 95.3. Raw model output 75.3 -> 95.3 = +26.6%, +20.3% above Book (79.20).", "date": "2026-08-21"},  # Dylan Sampson (CLE)
    "00-0034844": {"pct": 0.0, "note": "Reviewed, checked directly, nothing decisive found, no change: real minor ankle tweak at camp ('not believed serious,' no diagnosis announced); real new-OC scheme change (Sean Mannion replacing Kevin Patullo, see Hurts' QB entry) with vague 'whispers' about wanting more shotgun work -- too speculative to price. Raw (221.1) reasonably above Book (213.15).", "date": "2026-08-21"},  # Saquon Barkley (PHI)
    # ATL/KC/SF backfields (2026-08-21) -- the three teams left "reviewed
    # but not corrected" earlier this session, re-examined with the same
    # real research rigor as everything else rather than trusted at face
    # value on the old one-line justifications.
    "00-0038542": {"pct": 6.4, "note": "Real, decisive, positive: signed a real 3yr/$75M extension ($51M guaranteed, the most guaranteed money for a non-rookie RB deal in NFL history), coming off a real All-Pro 2025 season (league-leading 2,298 scrimmage yards). The real camp holdout was contract leverage, resolved before signing, now ramping up normally -- not a role threat. Raw (262.8) sat under Book with no reason to withhold it. Raw model output 262.8 -> 279.6 = +6.4%, matches Book.", "date": "2026-08-21"},  # Bijan Robinson (ATL)
    "00-0038134": {"pct": 28.0, "note": "REVISED 2026-08-21, user-directed (underrated at Book match). Real basis: clean, uncontested KC RB1 -- Pacheco left for Detroit, Brashard Smith is passing-down/return depth only, no real committee threat exists. Prior trim only matched Book; with no real competition identified, the model sitting 15 spots below his own ADP (15.2) wasn't earned. +15% above the already-corrected 197.7 = 227.4. Raw model output 177.6 -> 227.4 = +28.0%.", "date": "2026-08-21"},  # Kenneth Walker III (KC)
    "00-0038705": {"pct": -57.3, "note": "Fixing a real gap: this session's own earlier note already correctly identified him as 'confirmed real 3rd-down/change-of-pace depth role,' but that finding was never actually applied to his number -- raw (75.6) sat +134% above Book (32.30) unaddressed. Applying the already-confirmed research now. Raw model output 75.6 -> 32.3 = -57.3%, matches Book.", "date": "2026-08-21"},  # Emari Demercado (KC)
    "00-0033280": {"pct": -9.1, "note": "Real, current, unresolved: held out of practice since Aug 10, 2026 with unspecified 'tightness,' team calls it managed veteran workload with no diagnosis -- but there's also real, dated speculation this is tied to leveraging his non-guaranteed $12.5M 2026 salary rather than a real injury. Stacked on his own real extensive prior injury history (his risk score already shows acute 100/100, chronic 90/100 from multiple significant injuries), the compounding real risk argues for sitting a bit under Book, not just at it. Raw model output 280.4 -> 254.9 = -9.1%, -3.7% below Book (264.65).", "date": "2026-08-21"},  # Christian McCaffrey (SF)
    # Full-audit batch (2026-08-21) -- user-driven spot-check of Our Score
    # across the whole board, catching several real gaps: two suppressed-
    # season constructions that never got the healthy-window treatment
    # (Wilson, and see A.J. Brown above), a direct instruction (McMillan),
    # and a real, structural finding that three previously-reviewed RBs
    # (Hall/Williams/Judkins) were presented as "confirmed, no change" in
    # an earlier table but never actually written -- sitting on bare,
    # unreviewed FantasyPros numbers this whole time, which is what
    # produced the "RB14-29 looks like ADP" observation.
    "00-0037740": {"pct": 23.1, "note": "REVISED 2026-08-21, user-directed: +5% off the prior 190.0. 190.0 x 1.05 = 199.5. Raw model output 162.0 -> 199.5 = +23.1%. Prior basis: raw was built off an injury-shortened 2025 (7 games, real ongoing knee issues) -- but his rate THAT year (1.381/target) was actually HIGHER than his healthy 2024 (1.326/target), a real volume story. Real, confirmed healthy for 2026.", "date": "2026-08-21"},  # Garrett Wilson (NYJ)
    "00-0040124": {"pct": 15.0, "note": "User-directed: raised 15% from raw. Raw model output 165.1 -> 189.9, +11.2% above Book (170.80).", "date": "2026-08-21"},  # Tetairoa McMillan (CAR)
    "00-0038120": {"pct": 0.0, "note": "Structural fix: presented as 'confirmed, no change' in an earlier audit table (real workhorse, no threat identified) but never actually written -- caught when the RB14-29 range was checked and found to be sitting on the bare, unreviewed FantasyPros number this whole time. Applying the confirmation now. Raw (229.8), +23.6% above Book (185.95) -- a real, earned gap for a real workhorse, not corrected further absent a specific reason to.", "date": "2026-08-21"},  # Breece Hall (NYJ)
    "00-0036997": {"pct": 0.0, "note": "Same structural fix as Hall (00-0038120) -- confirmed but never written. Raw (217.4), +8.5% above Book (200.35).", "date": "2026-08-21"},  # Javonte Williams (DAL)
    "00-0040784": {"pct": 0.0, "note": "Same structural fix as Hall (00-0038120) -- confirmed but never written. Raw (194.5), +13.9% above Book (170.75).", "date": "2026-08-21"},  # Quinshon Judkins (CLE)
    # Second occurrence of the same structural gap (2026-08-21), caught via
    # a direct question about why Derrick Henry's score looked high: his
    # OWN real 2024->2025 decline (325 carries/5.91 ypc -> 307 carries/5.20
    # ypc, both real, at real ages 30->31) plus the real "Age 32: past
    # where RB year-over-year decline accelerates" flag already sitting in
    # his own data were never actually weighed -- he was never pulled into
    # review at all this session despite every other aging veteran back
    # getting one. Taylor/Achane/Cook were separately presented as
    # "confirmed" in their team batches but, like Hall/Williams/Judkins,
    # never actually written -- running on stale, unreviewed FantasyPros
    # numbers the whole time. Perine's write is a THIRD, distinct failure
    # mode: an entry was drafted for him in the very first backfield batch
    # but never actually landed in this file at all -- re-written now.
    "00-0032764": {"pct": -3.0, "note": "Real, now actually reviewed (see the batch comment above): confirmed year-over-year decline already visible in his own real data (2024: 325 car/1921 yds/5.91 ypc; 2025: 307 car/1595 yds/5.20 ypc, both a real drop at real ages 30->31), compounding the real 'age 32: past where RB decline accelerates' flag. No specific positive reason found to hold him above Book given that real trend. Raw model output 234.8 -> 227.75 = -3.0%, matches Book.", "date": "2026-08-21"},  # Derrick Henry (BAL)
    "00-0036223": {"pct": 0.0, "note": "Structural fix: presented as 'confirmed, no change' in the IND batch table but never actually written. Real, confirmed elite bellcow, no threat identified. Raw (270.3), +10.0% above Book (245.75) -- a real, earned gap, not corrected further absent a specific reason to.", "date": "2026-08-21"},  # Jonathan Taylor (IND)
    "00-0039040": {"pct": 0.0, "note": "Same structural fix as Taylor (00-0036223) -- confirmed but never written. Raw (241.6), +13.1% above Book (213.6).", "date": "2026-08-21"},  # De'Von Achane (MIA)
    "00-0037248": {"pct": 0.0, "note": "Same structural fix as Taylor (00-0036223) -- confirmed but never written. Raw (238.3), -0.6% vs. Book (239.75).", "date": "2026-08-21"},  # James Cook (BUF)
    "00-0033526": {"pct": -40.0, "note": "Re-written: an entry for him was drafted in the very first backfield batch (real, confirmed stable CIN backup role behind Chase Brown, moved to Book) but never actually landed in this file -- confirmed missing on the 2026-08-21 audit. Raw model output 86.0 -> 51.6 = -40.0%, matches Book (51.60).", "date": "2026-08-21"},  # Samaje Perine (CIN)
    # Blanket TE trim (2026-08-21), user-directed: -10% off every TE's
    # current (already-corrected, where applicable) value except Loveland,
    # Warren, Bowers, and McBride -- no individual research done, disclosed
    # batch move like the market-defer passes elsewhere in this file. Where
    # a gsis_id already has an earlier entry above, Python dict-literal
    # semantics mean this later entry is the one that actually takes
    # effect (last key wins), so the -10% is applied on top of whatever
    # correction (if any) already existed, not on top of raw twice.
    "00-0038996": {"pct": -22.6, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 133.4 x 0.90 = 120.1. Raw model output 155.1 -> 120.1 = -22.6%.", "date": "2026-08-21"},  # Tucker Kraft
    "00-0039065": {"pct": -17.2, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 127.5 x 0.90 = 114.8. Raw model output 138.6 -> 114.8 = -17.2%.", "date": "2026-08-21"},  # Sam LaPorta
    "00-0040663": {"pct": -11.3, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 138.2 x 0.90 = 124.4. Raw model output 140.3 -> 124.4 = -11.3%.", "date": "2026-08-21"},  # Harold Fannin
    "00-0036970": {"pct": -3.7, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 146.2 x 0.90 = 131.6. Raw model output 136.6 -> 131.6 = -3.7%.", "date": "2026-08-21"},  # Kyle Pitts
    "00-0033288": {"pct": -39.3, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 115.8 x 0.90 = 104.2. Raw model output 171.8 -> 104.2 = -39.3%.", "date": "2026-08-21"},  # George Kittle
    "00-0030506": {"pct": -26.7, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 114.9 x 0.90 = 103.4. Raw model output 141.1 -> 103.4 = -26.7%.", "date": "2026-08-21"},  # Travis Kelce
    "00-0034753": {"pct": -16.8, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 108.0 x 0.90 = 97.2. Raw model output 116.8 -> 97.2 = -16.8%.", "date": "2026-08-21"},  # Mark Andrews
    "00-0038933": {"pct": -3.1, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 132.0 x 0.90 = 118.8. Raw model output 122.6 -> 118.8 = -3.1%.", "date": "2026-08-21"},  # Dalton Kincaid
    "00-0037838": {"pct": 30.9, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 113.8 x 0.90 = 102.4. Raw model output 78.2 -> 102.4 = +30.9%.", "date": "2026-08-21"},  # Isaiah Likely
    "00-0038041": {"pct": -10.0, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 118.6 x 0.90 = 106.7. Raw model output 118.6 -> 106.7 = -10.0%.", "date": "2026-08-21"},  # Jake Ferguson
    "00-0034351": {"pct": -23.7, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 121.0 x 0.90 = 108.9. Raw model output 142.7 -> 108.9 = -23.7%.", "date": "2026-08-21"},  # Dallas Goedert
    "00-0037809": {"pct": 13.8, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 114.2 x 0.90 = 102.8. Raw model output 90.3 -> 102.8 = +13.8%.", "date": "2026-08-21"},  # Chig Okonkwo
    "00-0038935": {"pct": 2.6, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 116.6 x 0.90 = 104.9. Raw model output 102.2 -> 104.9 = +2.6%.", "date": "2026-08-21"},  # Brenton Strange
    "00-0040189": {"pct": -2.0, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 119.4 x 0.90 = 107.5. Raw model output 109.7 -> 107.5 = -2.0%.", "date": "2026-08-21"},  # Oronde Gadsden
    "00-0033090": {"pct": -12.3, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 124.5 x 0.90 = 112.0. Raw model output 127.7 -> 112.0 = -12.3%.", "date": "2026-08-21"},  # Hunter Henry
    "00-0036040": {"pct": -10.0, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 122.4 x 0.90 = 110.2. Raw model output 122.4 -> 110.2 = -10.0%.", "date": "2026-08-21"},  # Juwan Johnson
    "00-0034383": {"pct": -4.6, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 123.1 x 0.90 = 110.8. Raw model output 116.2 -> 110.8 = -4.6%.", "date": "2026-08-21"},  # Dalton Schultz
    "00-0035229": {"pct": 9.5, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 122.0 x 0.90 = 109.8. Raw model output 100.3 -> 109.8 = +9.5%.", "date": "2026-08-21"},  # T.J. Hockenson
    "00-0039793": {"pct": -10.0, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 100.1 x 0.90 = 90.1. Raw model output 100.1 -> 90.1 = -10.0%.", "date": "2026-08-21"},  # AJ Barner
    "00-0037252": {"pct": 12.9, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 87.4 x 0.90 = 78.7. Raw model output 69.7 -> 78.7 = +12.9%.", "date": "2026-08-21"},  # Greg Dulcich
    "00-0040737": {"pct": -19.0, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 61.0 x 0.90 = 54.9. Raw model output 67.8 -> 54.9 = -19.0%.", "date": "2026-08-21"},  # Terrance Ferguson
    "00-0040584": {"pct": 29.5, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 101.4 x 0.90 = 91.3. Raw model output 70.5 -> 91.3 = +29.5%.", "date": "2026-08-21"},  # Gunnar Helm
    "00-0038129": {"pct": -9.2, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 101.7 x 0.90 = 91.5. Raw model output 100.8 -> 91.5 = -9.2%.", "date": "2026-08-21"},  # Cade Otton
    "00-0036894": {"pct": -7.9, "note": "REVISED 2026-08-21, user-directed: blanket -10% TE trim (all TEs except Loveland/Warren/Bowers/McBride), no individual research done. 102.9 x 0.90 = 92.6. Raw model output 100.5 -> 92.6 = -7.9%.", "date": "2026-08-21"},  # Pat Freiermuth
}
# Kenneth Walker III (KC), Bijan Robinson (ATL), Christian McCaffrey (SF),
# and Emari Demercado (KC) were originally left here as "reviewed but not
# corrected." Re-examined 2026-08-21 with the same research rigor as
# everything else in this file rather than trusted at face value on the
# old one-line justifications below -- all four now have real entries
# above (Walker/Bijan/Demercado corrected on real new evidence, McCaffrey
# on a real current camp situation the old note never disclosed).


def apply_model_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Adds model_projection_points_adjusted (+ model_adjustment_pct/
    model_adjustment_note), defaulting to the raw model_projection_points
    for every player except the individually-verified, diagnosed-defect
    cases in MODEL_PROJECTION_CORRECTIONS above. model_projection_points
    itself is never modified -- see that dict's comment."""
    df["model_projection_points_adjusted"] = df["model_projection_points"]
    df["model_adjustment_pct"] = np.nan
    df["model_adjustment_note"] = None
    for gsis_id, correction in MODEL_PROJECTION_CORRECTIONS.items():
        mask = df["gsis_id"].eq(gsis_id) & df["model_projection_points"].notna()
        if not mask.any():
            continue
        df.loc[mask, "model_projection_points_adjusted"] = (
            df.loc[mask, "model_projection_points"] * (1 + correction["pct"] / 100)
        ).round(1)
        df.loc[mask, "model_adjustment_pct"] = correction["pct"]
        df.loc[mask, "model_adjustment_note"] = f"{correction['note']} (verified {correction['date']})"
    return df


# rb_correction_manual_review_prompt.pdf ("scope clarification" mid-review,
# 2026-08-19): a rookie's insufficient_history quantile fallback (above) is
# a generic, distribution-matched substitute -- once the manual per-player
# review pass sources a MORE SPECIFIC real number for a given rookie (here,
# a real sportsbook-derived Book projection, data/processed/
# sportsbook_vs_adp_comparison.csv), that specific number supersedes the
# generic estimate rather than living alongside it as a separate,
# potentially-conflicting figure. Keyed on gsis_id, applied as a direct
# override of model_projection_points_fallback (never model_projection_points
# itself, which stays genuinely blank for real insufficient_history players
# -- same non-negotiable as MODEL_PROJECTION_CORRECTIONS).
ROOKIE_FALLBACK_BOOK_OVERRIDES = {
    "00-0041512": {  # Jadarian Price (SEA)
        "value": 169.6,
        "note": "REVISED 2026-08-21 (user asked for a slight further boost off the prior +10%; no "
                "new evidence found beyond the existing case, so this is a direct, modest widening of "
                "the same boost rather than a re-derivation). Real basis unchanged: 'everyone expects "
                "Jadarian Price to replace Kenneth Walker as the Seahawks' RB1' (Walker's real "
                "departure to KC already priced in that team's own entry above), real 1st-round pick "
                "(#32 overall). The real caveats that kept the original boost modest -- HC Mike "
                "Macdonald 'will make him earn it' in a real 3-way committee (Holani, Price, Wilson), "
                "and a real, specific weakness at only 15 college receptions -- still apply and are "
                "why this stays a slight move (10% -> 18%) rather than a large one. 143.75 x 1.18 = "
                "169.6.",
        "date": "2026-08-21",
    },
    "00-0041027": {  # Jeremiyah Love (ARI)
        "value": 225.6,
        "note": "Superseded 2026-08-19 from the generic quantile-matched fallback (191.6) -- "
                "sourced during James Conner's (ARI) manual review, but NOT simply Book's number "
                "(193.15), and NOT deferred to Book by default. Independently constructed from a "
                "real, specific, independently-sourced 'realistic projection' for Love (275 total "
                "touches, 1,500 scrimmage yards, 10 TD, ~50 receptions): 150 (yardage) + 60 (10 TD) "
                "+ 25 (50-reception half-PPR bonus) = 235.0. Love is ARI's real #3-overall pick, "
                "'consensus top RB in the draft class' -- real coaching hedges ('he's definitely a "
                "rookie,' 'looked like a rookie so far') exist, but 'almost no chance Love isn't the "
                "Cardinals' top RB for the regular season, barring injury' per reporting. This number "
                "is MORE bullish than Book, a genuine independent construction, not a copy -- shares "
                "the same 420-touch ARI backfield pool assumption as Conner's (00-0033553)/Allgeier's "
                "(00-0037263) entries. REVISED 2026-08-21, user-directed: -4% off 235.0 = 225.6.",
        "date": "2026-08-21",
    },
    "00-0041496": {  # Jonah Coleman (DEN)
        "value": 60.9,
        "note": "Lowered 2026-08-20 from the generic quantile-matched fallback (82.7) -- sourced "
                "during the DEN backfield review (Dobbins/Harvey, 00-0036158/00-0040730). Real 4th-"
                "round rookie pick, but landing in a real, crowded, established two-veteran backfield "
                "(Dobbins re-signed, Harvey a proven 2025 2nd-round rookie with a real efficient "
                "season) -- Book's real, lower current pricing (60.9) is a better real signal than "
                "the generic quantile estimate for a player facing this much real established "
                "competition ahead of him.",
        "date": "2026-08-20",
    },
}


def compute_rookie_fallback(full_position: pd.DataFrame, position: str, veteran_points: pd.Series) -> pd.DataFrame:
    """rb_model_fix_plan.pdf, Phase 1: model_projection_status ==
    "insufficient_history" previously left MODEL PROJ blank for real
    rookies who DO have a real, already-computed pre-draft composite
    (draftkit/rookie_projection.py, data/processed/rookie_projections.csv)
    -- dropping them out of any rank comparison entirely instead of
    substituting something. The archetype/composite itself is real and
    already validated (see that module's docstring); this is a plumbing
    fix wiring it into a MODEL PROJ-comparable value, not a new model.

    Quantile-matched, not an arbitrary scale factor: each rookie's real
    composite score is placed at its percentile within the real
    population of ALL {position} draft-class composites ever computed
    (current board + already-confirmed past classes, rookie_projections.csv)
    -- that percentile is then read off the real, current, VALIDATED
    veteran model_projection_points distribution (`veteran_points`, the
    same "projected"-status population the real outcome-trained model
    itself produces) at the matching percentile. High-composite prospects
    land near real difference-making veteran production, weak ones land
    near real replacement-level -- grounded in two real distributions on
    either side, not a made-up conversion constant. Written to a SEPARATE
    column (model_projection_points_fallback), never model_projection_points
    itself -- that column staying blank for these players remains a real,
    meaningful statement ("no outcome-trained prediction exists"), not
    something this substitute should silently overwrite."""
    full_position = full_position.copy()
    full_position["model_projection_points_fallback"] = np.nan
    full_position["model_projection_fallback_note"] = None
    if position not in ROOKIE_FALLBACK_POSITIONS or veteran_points.empty:
        return full_position

    rookies = pd.read_csv(ROOKIE_PROJECTIONS_PATH)
    rookies = rookies[rookies["position"].eq(position)].dropna(subset=["composite"]).copy()
    if rookies.empty:
        return full_position
    rookies["key"] = rookies["player_name"].apply(_clean_name_ext)
    rookies = rookies.drop_duplicates("key", keep="first")
    composite_pool = rookies["composite"].to_numpy()
    n_rookies = len(composite_pool)
    n_veterans = len(veteran_points)
    veteran_pool = veteran_points.to_numpy()

    composite_by_key = rookies.set_index("key")["composite"]
    tier_by_key = rookies.set_index("key")["display_tag"]
    key = full_position["player_name"].apply(_clean_name_ext)
    is_fallback_target = (
        full_position["model_projection_status"].eq("insufficient_history") & key.isin(composite_by_key.index)
    )

    for idx in full_position.index[is_fallback_target]:
        k = key.loc[idx]
        composite = float(composite_by_key[k])
        tier = tier_by_key.get(k)
        pct = float((composite_pool < composite).sum()) / n_rookies * 100
        fallback_value = round(float(np.percentile(veteran_pool, pct)), 1)
        full_position.loc[idx, "model_projection_points_fallback"] = fallback_value
        full_position.loc[idx, "model_projection_fallback_note"] = (
            f"Rookie-score fallback (not a real outcome-trained prediction): pre-draft composite "
            f"{composite:.1f} ({tier}) sits at the {pct:.0f}th percentile of {n_rookies} real {position} "
            f"draft-class composites -- mapped to the {pct:.0f}th percentile of {n_veterans} real, "
            f"validated veteran {position} model_projection_points"
        )
    n_filled = int(is_fallback_target.sum())
    print(f"  rookie fallback: {n_filled} insufficient_history {position} players given a substitute MODEL PROJ value")

    n_overridden = 0
    for gsis_id, override in ROOKIE_FALLBACK_BOOK_OVERRIDES.items():
        mask = full_position["gsis_id"].eq(gsis_id)
        if not mask.any():
            continue
        full_position.loc[mask, "model_projection_points_fallback"] = override["value"]
        full_position.loc[mask, "model_projection_fallback_note"] = (
            f"{override['note']} (verified {override['date']})"
        )
        n_overridden += 1
    if n_overridden:
        print(f"  rookie fallback: {n_overridden} {position} players' fallback superseded by a sourced Book override")
    return full_position


# rb_correction_manual_review_prompt.pdf's passing-offense pass (2026-08-20):
# a real rookie can have model_projection_status=="insufficient_history" at
# a position compute_rookie_fallback() doesn't cover (ROOKIE_FALLBACK_POSITIONS
# is RB-only -- extending it to WR/TE is a real, larger engineering lift, not
# something to build for one player). This is a deliberately minimal,
# position-agnostic escape hatch: a Book-sourced placeholder value, applied
# directly by gsis_id after the per-position loop, clearly labeled as
# exactly that -- a placeholder pending the real WR/TE fallback build this
# entry itself flags as still-needed, not a disguised model output.
MANUAL_FALLBACK_PLACEHOLDERS = {
    "00-0041550": {  # Brenen Thompson (LAC), WR -- no fallback mechanism exists for this position
        "value": 32.0,
        "note": "PLACEHOLDER, not a model output: real 2026 3rd-round-ish rookie (105th overall), "
                "competing with Tre' Harris (00-0040727) for LAC's WR3/4 role, but real reporting "
                "gives Harris the clear real edge ('clearly' top-3 per HC Harbaugh) -- Thompson is "
                "real early-down/situational depth behind him. No independent construction possible: "
                "model_projection_status is insufficient_history and ROOKIE_FALLBACK_POSITIONS is "
                "RB-only, so no generic fallback exists either for this WR to override. Using Book's "
                "real current pricing (32.0) directly as a stopgap so the cell isn't blank, pending a "
                "real WR/TE rookie-fallback build (same gap this note exists to flag).",
        "date": "2026-08-20",
    },
    "00-0041040": {  # Antonio Williams (WAS), WR -- same structural gap as Thompson above
        "value": 61.2,
        "note": "PLACEHOLDER, not a model output: real rookie WR, model_projection_status is "
                "insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap as Thompson, 00-0041550). Using Book's real current "
                "pricing (61.2) directly as a stopgap, pending the same real WR/TE rookie-fallback "
                "build.",
        "date": "2026-08-20",
    },
    "00-0041042": {  # Malachi Fields (NYG), WR -- same structural gap as Thompson/Williams above
        "value": 67.6,
        "note": "PLACEHOLDER, not a model output: real 3rd-round rookie, model_projection_status is "
                "insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap as Thompson/Williams above). Using Book's real "
                "current pricing (67.6) directly as a stopgap, pending the same real WR/TE "
                "rookie-fallback build.",
        "date": "2026-08-20",
    },
    "00-0041489": {  # Germie Bernard (PIT), WR -- same structural gap as Thompson/Williams/Fields above
        "value": 80.9,
        "note": "PLACEHOLDER, not a model output: real 2026 2nd-round rookie, model_projection_status "
                "is insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap as the other WR rookies above). Real, current "
                "reporting confirms he's currently behind Roman Wilson (00-0039739) for the WR3 "
                "role ('soundly held off... for the WR3 job'). Using Book's real current pricing "
                "(80.9) directly as a stopgap, pending the same real WR/TE rookie-fallback build.",
        "date": "2026-08-20",
    },
    "00-0041438": {  # Carnell Tate (TEN), WR -- same structural gap above, but INDEPENDENTLY BUILT, not a bare Book copy
        "value": 162.0,
        "note": "NOT a bare Book placeholder like the other entries in this dict -- independently "
                "constructed, same standard as Love's entry in ROOKIE_FALLBACK_BOOK_OVERRIDES "
                "above (which this mechanism can't reach for a WR, since ROOKIE_FALLBACK_POSITIONS "
                "is RB-only). Real #4 overall 2026 pick, real, explicit 'primary read on most "
                "passing dropbacks' framing, real elite college production (121 catches, 1,872 "
                "yds, 14 TD at Ohio State). 120 targets (real, large allocation matching the "
                "'primary read' framing) x 1.35 (proxy rate, no NFL history -- blended from real "
                "TEN teammate rates) = 162.0. +16.7% above Book (138.80), part of the same 460-"
                "touch TEN pool as Robinson (00-0038117)/Ridley (00-0034837)/Ayomanor "
                "(00-0040170)/Dike (00-0040705)/Helm (00-0040584).",
        "date": "2026-08-20",
    },
    "00-0041035": {  # De'Zhaun Stribling (SF), WR -- same structural gap as the other WR rookies above
        "value": 126.5,
        "note": "PLACEHOLDER, not a bare Book copy: real 2026 rookie (SF's first draft selection), "
                "model_projection_status is insufficient_history with no fallback available (same "
                "RB-only ROOKIE_FALLBACK_POSITIONS gap as the other WR rookies above). Real, "
                "current context: SF's receiving corps is real, decisively thin given Pearsall's "
                "season-ending injury and Aiyuk's departure, a real opportunity driver for him. "
                "User-directed figure: Book's real current pricing (105.4) boosted 20% to 126.5, "
                "reflecting that real opportunity beyond what Book alone captures, pending the same "
                "real WR/TE rookie-fallback build.",
        "date": "2026-08-20",
    },
    "00-0040894": {  # Malik Benson (LV), WR -- same structural gap above, but no Book price exists either
        "value": 30.0,
        "note": "PLACEHOLDER, independently built (no Book price exists for him at all, unlike the "
                "other entries in this dict): real late-round 2026 rookie, model_projection_status "
                "is insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap). Real, very fresh (Aug 17-19, 2026) camp reporting: "
                "'battling to be a starter or at worst a solid rotational weapon,' working into "
                "1st-team snaps -- real competition for the rotational spot behind LV's starting "
                "trio (Tucker/Nailor/Bech, see LV batch companion entries), not yet a threat to the "
                "starters themselves. 25 targets (a modest, real rotational-depth role, not a "
                "starting share) x 1.2 (proxy rate, no NFL history -- blended from real LV teammate "
                "rates) = 30.0.",
        "date": "2026-08-20",
    },
    "00-0041525": {  # Chris Bell (MIA), WR -- same structural gap above, bare Book copy per user direction
        "value": 74.7,
        "note": "PLACEHOLDER: real 3rd-round rookie, model_projection_status is insufficient_history "
                "with no fallback available (same RB-only ROOKIE_FALLBACK_POSITIONS gap). Real, "
                "borderline-1st-round college talent (72-917-6 final Louisville season) before "
                "tearing his ACL Nov 22, 2025 -- Miami's own beat writer calls him 'the missing "
                "ingredient.' Real, acute availability risk (activated off NFI Aug 17, 2026, "
                "practicing in a red non-contact jersey, no definitive timetable) initially argued "
                "for an independently-built, risk-discounted figure below Book -- but user-directed "
                "to defer to Book for this whole rebuilt MIA offense, same as every other player in "
                "this batch. Using Book's real current pricing (74.7) directly.",
        "date": "2026-08-20",
    },
    "00-0041523": {  # Caleb Douglas (MIA), WR -- same structural gap above, bare Book copy
        "value": 26.5,
        "note": "PLACEHOLDER, not independently built: real 3rd-round rookie, model_projection_status "
                "is insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap). Real, current reporting calls him 'a reach' (Sharp "
                "Football), sitting behind Tolbert (00-0037666)/Atwell (00-0036849) on the depth "
                "chart -- no specific reason found to deviate from Book, consistent with deferring "
                "to Book for this whole rebuilt MIA offense. Using Book's real current pricing "
                "(26.5) directly as a stopgap, pending the real WR/TE rookie-fallback build.",
        "date": "2026-08-20",
    },
    "00-0041547": {  # KC Concepcion (CLE), WR -- boosted off Book on real camp hype
        "value": 136.1,
        "note": "REVISED 2026-08-21 (boosted 15% off Book's 118.35 on re-audit -- 'needs a boost "
                "based on camp hype,' and this holds up as real production, not just buzz): real "
                "2026 1st-round pick (24th overall, the return piece completing the 2025 Travis "
                "Hunter trade from JAX). Real, dated preseason production (not just camp reports): "
                "3 catches and the Browns' lone receiving TD (on a jet sweep) in the preseason "
                "opener vs. CHI, plus a long punt return -- real, multi-phase usage (motion at the "
                "line, punt returns) the team is already featuring him in. model_projection_status "
                "is insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap). 118.35 x 1.15 = 136.1.",
        "date": "2026-08-21",
    },
    "00-0041037": {  # Denzel Boston (CLE), WR -- same structural gap above, bare Book copy
        "value": 93.15,
        "note": "PLACEHOLDER, not independently built: real 2026 2nd-round pick (Washington), same "
                "structural gap as Concepcion (00-0041547) above. Using Book's real current pricing "
                "(93.15) directly, consistent with the market-defer treatment given to the rest of "
                "this CLE batch.",
        "date": "2026-08-20",
    },
    "00-0041511": {  # Omar Cooper Jr. (NYJ), WR -- same structural gap above, bare Book copy
        "value": 104.6,
        "note": "PLACEHOLDER, not independently built: real 2026 rookie (Indiana), ranked 9th "
                "overall in the 2026 rookie class, real reporting specifically notes he fills a "
                "'slot WR' gap NYJ's depth chart lacked -- a real weapon missing from NYJ's initial "
                "three-name list (Wilson/Taylor/Mitchell). model_projection_status is "
                "insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap). Using Book's real current pricing (104.6) directly, "
                "consistent with the market-defer treatment given to the rest of this NYJ batch.",
        "date": "2026-08-20",
    },
    "00-0041032": {  # Kenyon Sadiq (NYJ), TE -- same structural gap above, bare Book copy
        "value": 94.6,
        "note": "PLACEHOLDER, not independently built: real 2026 rookie TE, projected to split slot "
                "reps with Cooper (00-0041511) above. Same structural gap as Cooper. Using Book's "
                "real current pricing (105.1) directly, consistent with the market-defer treatment "
                "given to the rest of this NYJ batch. REVISED 2026-08-21, user-directed: blanket "
                "-10% TE trim (all TEs except Loveland/Warren/Bowers/McBride). 105.1 x 0.90 = 94.6.",
        "date": "2026-08-21",
    },
    "00-0041047": {  # Chris Brazzell II (CAR), WR -- same structural gap above, near-total cut
        "value": 0.0,
        "note": "PLACEHOLDER: real 3rd-round rookie (Louisville, 72-917-6 final college season, "
                "borderline 1st-round talent before tearing his ACL in November), tore his LCL in a "
                "non-contact 7-on-7 rep at 2026 camp -- real, confirmed, will miss all of 2026. "
                "model_projection_status is insufficient_history with no fallback available (same "
                "RB-only ROOKIE_FALLBACK_POSITIONS gap). Near-total cut, same treatment as Pearsall's "
                "(SF)/Higgins' (HOU) season-ending cases -- no Book price exists for him.",
        "date": "2026-08-20",
    },
    "00-0040870": {  # Ja'Kobi Lane (BAL), WR -- same structural gap above, boosted off Book
        "value": 111.7,
        "note": "REVISED 2026-08-21: real, confirmed Bateman arrest (see 00-0036550) creates real "
                "personal-conduct-policy/roster risk behind him -- 35 of the 50 points Bateman lost "
                "(65.1 x 50% = 32.5 pool, split 35/15 per user direction) land here. Prior real basis "
                "unchanged: 3rd-round rookie, real escalating camp story (ESPN's Jamison Hensley called "
                "it the best rookie camp he's seen; team site had him as No. 3 WR behind Flowers/Bateman "
                "as of Aug 5, 2026). 88.9 + 22.8 (35% of Bateman's 65.1) = 111.7.",
        "date": "2026-08-21",
    },
    "00-0040876": {  # Elijah Sarratt (BAL), WR -- same structural gap above, bare Book copy
        "value": 82.9,
        "note": "REVISED 2026-08-21: real, confirmed Bateman arrest (see 00-0036550) creates real "
                "personal-conduct-policy/roster risk behind him -- 15 of the 50 points Bateman lost "
                "(65.1 x 50% = 32.5 pool, split 35/15 per user direction) land here. Prior real basis "
                "unchanged: 4th-round rookie, real competition for snaps against Lane (00-0040870) and "
                "Devontez Walker behind Flowers/Bateman. 73.1 + 9.8 (15% of Bateman's 65.1) = 82.9.",
        "date": "2026-08-21",
    },
    "00-0040890": {  # Cyrus Allen (KC), WR -- same structural gap above, independently built
        "value": 80.0,
        "note": "NOT a bare Book copy (no Book price exists for him at all): real 5th-round rookie "
                "(Cincinnati), real, dated camp buzz -- earning real first-team slot reps with "
                "Mahomes as of early August 2026, ahead of expectations for his draft slot. "
                "model_projection_status is insufficient_history with no fallback available (same "
                "RB-only ROOKIE_FALLBACK_POSITIONS gap). User-directed to 80.0, reflecting real "
                "conviction in that camp momentum beyond a token depth-role value.",
        "date": "2026-08-20",
    },
    "00-0041029": {  # Jordyn Tyson (NO), WR -- same structural gap above, independently built with a real injury discount
        "value": 65.0,
        "note": "NOT a bare Book copy: real, current, decisive: hamstring injury in camp, expected "
                "to miss roughly two months, outlook called 'bleak,' real chance of opening the "
                "season on IR (verified via NFL.com/ESPN, Aug 2026) -- Book's real current pricing "
                "(134.70) is stale relative to this news and shouldn't be used as-is. Discounted "
                "well below Book to reflect the real missed-time risk, same treatment class as "
                "Chris Bell's (MIA) NFI situation earlier this session. model_projection_status is "
                "insufficient_history with no fallback available (same RB-only "
                "ROOKIE_FALLBACK_POSITIONS gap).",
        "date": "2026-08-20",
    },
    "00-0040867": {  # Makai Lemon (PHI), WR -- same structural gap above, independently built with a real injury discount
        "value": 75.0,
        "note": "NOT a bare Book copy: real 1st-round pick (No. 20 overall, PHI traded up for him), "
                "which Book's real current pricing (138.50) likely reflects on draft capital alone. "
                "But real, current, dated (Aug 20, 2026): hamstring injury since late May, "
                "re-injured in early August, missed most of camp, only returned to practice Aug 20 "
                "-- real doubt he starts by Week 1 (verified via Inquirer/Yahoo). Book's price "
                "predates or doesn't fully reflect that missed-camp reality. Discounted well below "
                "Book. model_projection_status is insufficient_history with no fallback available "
                "(same RB-only ROOKIE_FALLBACK_POSITIONS gap).",
        "date": "2026-08-20",
    },
}


def apply_manual_fallback_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Applies MANUAL_FALLBACK_PLACEHOLDERS to model_projection_points_fallback
    directly, position-agnostic -- covers a real rookie at a position
    compute_rookie_fallback() doesn't reach. Never touches
    model_projection_points itself."""
    n_applied = 0
    for gsis_id, override in MANUAL_FALLBACK_PLACEHOLDERS.items():
        mask = df["gsis_id"].eq(gsis_id)
        if not mask.any():
            continue
        df.loc[mask, "model_projection_points_fallback"] = override["value"]
        df.loc[mask, "model_projection_fallback_note"] = f"{override['note']} (verified {override['date']})"
        n_applied += 1
    if n_applied:
        print(f"  manual fallback placeholders: {n_applied} players given a Book-sourced stopgap value")
    return df


# rb_correction_manual_review_prompt.pdf's SF batch (2026-08-20): a real,
# structurally distinct gap from every other placeholder mechanism above.
# Brandon Aiyuk exists in master_players.csv (real name/team/position) but
# has NO gsis_id anywhere in the crosswalk -- genuinely absent from BOTH
# stats_player_reg_by_season/2025.csv and roster_2026.csv (checked
# directly, not a name-spelling mismatch like Gainwell/Brooks/Knight --
# this is real data absence, not a fixable join). Neither
# MODEL_PROJECTION_CORRECTIONS (needs a real raw value to adjust) nor
# MANUAL_FALLBACK_PLACEHOLDERS (matches on gsis_id) can reach a player
# with no gsis_id at all. Keyed on player_name instead -- Home.py's own
# merge of this file onto master_players.csv is ALSO by player_name, so
# this is consistent with how the live app actually joins this data, not
# a workaround invented for this one case.
NAME_KEYED_FALLBACK_OVERRIDES = {
    "Brandon Aiyuk": {
        "value": 0.0,
        "note": "Real, structurally distinct gap: no gsis_id exists anywhere in the crosswalk for "
                "him (confirmed absent from both stats_player_reg_by_season/2025.csv and "
                "roster_2026.csv directly -- not a name-spelling mismatch, genuine data absence). "
                "Real, decisive current status: 'played his last game for the team,' amid an "
                "ongoing real public campaign to be released so he can sign with WAS. User-directed "
                "value: 0.0, reflecting his real, effectively-done SF status.",
        "date": "2026-08-20",
    },
    "Joshua Palmer": {
        "value": 33.5,
        "note": "Real, distinct gap from Aiyuk's above: he DOES have a real gsis_id "
                "(00-0036988) in both stats_player_reg_by_season/2025.csv and "
                "roster_2026.csv, but both key him as 'Josh Palmer' while "
                "master_players.csv/Book key him as 'Joshua Palmer' -- unlike Gainwell's "
                "Kenneth/Kenny variant (00-0036919, above), which still resolved via the "
                "existing key match, this Josh/Joshua variant genuinely breaks the join "
                "(model_projection_status came back no_crosswalk_match). Real, dated camp "
                "reporting (as of days before Buffalo's first 2026 preseason game): 'the WR3 "
                "spot appears to be his to lose' over Keon Coleman (00-0039901, see BUF batch "
                "companion entry) -- healthy again and 'catching everything in sight' while "
                "Coleman deals with drops. Value set above both Coleman's corrected figure "
                "and his own stale Book price (26.60) to reflect that real, current flip in "
                "the depth chart.",
        "date": "2026-08-20",
    },
    "Matt Hibner": {
        "value": 14.0,
        "note": "Real, structurally distinct gap, same class as Aiyuk's above: no gsis_id exists "
                "anywhere in the crosswalk for him (confirmed absent from both "
                "stats_player_reg_by_season/2025.csv and roster_2026.csv -- a real 2026 4th-round "
                "rookie TE, acquired via a trade with SF, so genuinely has no NFL history to key on "
                "regardless). Real, current context: a developmental depth piece behind Mark Andrews "
                "(00-0034753) and Isaiah Likely, not currently a target-earner. No Book price exists. "
                "Independently built: 10 targets (a token depth-role share) x 1.4 (generic TE proxy "
                "rate, no NFL history) = 14.0.",
        "date": "2026-08-20",
    },
    "Barion Brown": {
        "value": 50.0,
        "note": "Real, structurally distinct gap, same class as Aiyuk's/Hibner's above: no gsis_id "
                "exists anywhere in the crosswalk for him (a real 2026 rookie WR, no NFL history to "
                "key on). Real, current context: real reporting has him currently ahead of Trey "
                "Palmer (00-0039052) on the real NO depth chart, and Jordyn Tyson's (00-0041029) "
                "real hamstring absence (see that entry) opens further real opportunity behind Olave "
                "(00-0037239). No Book price exists. Independently built: 35 targets (a real, modest "
                "WR3/4-tier share reflecting his real depth-chart edge over Palmer) x 1.4 (proxy "
                "rate, no NFL history) = 50.0.",
        "date": "2026-08-20",
    },
}


def apply_name_keyed_fallback_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Applies NAME_KEYED_FALLBACK_OVERRIDES to model_projection_points_fallback
    directly, keyed on player_name -- covers a real player entirely absent
    from the gsis_id crosswalk, which no other mechanism can reach. Never
    touches model_projection_points itself."""
    n_applied = 0
    for player_name, override in NAME_KEYED_FALLBACK_OVERRIDES.items():
        mask = df["player_name"].eq(player_name)
        if not mask.any():
            continue
        df.loc[mask, "model_projection_points_fallback"] = override["value"]
        df.loc[mask, "model_projection_fallback_note"] = f"{override['note']} (verified {override['date']})"
        n_applied += 1
    if n_applied:
        print(f"  name-keyed fallback overrides: {n_applied} players (no gsis_id in crosswalk) given a value")
    return df


# QB pass (2026-08-20). Structurally different from every other position
# in this file, deliberately: the module docstring excludes QB because the
# real QB MODEL is out-of-sample validated and loses to ADP by ~33-36% --
# wiring that model in would replace a working market signal with a worse
# one. Nothing here reverses that. No QB model is run, no QB feature is
# built, and model_projection_points stays NaN for every QB row (truthful:
# there is no model output). These are Book-anchored values written to
# model_projection_points_fallback -- the same "this is NOT a real model
# output" column the rookie placeholders use -- so the exclusion holds
# while QBs still get a usable, disclosed number in the live app.
#
# Every QB with a real Book price gets Book directly. The top 20 by Book
# additionally carry two disclosed adjustments:
#   sched_pct -- built from the repo's own real team_schedule_risk.csv,
#     reusing the SAME direction and components as the established
#     schedule_weather_venue_score() in draftkit/scripts/build_risk_
#     variables.py (season_sos_risk = (1 - rank/32)*2, so rank 1 = hardest
#     = penalty; weather = road_cold_games x 0.5 outdoor / 0.3 dome).
#     Scaled to +/-3.0% SOS and -0.4x weather. REAL LIMITATION, disclosed
#     rather than buried: season_sos_rank is a documented PROXY -- it
#     measures opponent OFFENSE EPA as a team-quality stand-in, not real
#     defensive strength ("no true defensive-SOS data exists in this
#     repo", per build_schedule_data.py). For QB scoring specifically the
#     opposing PASS DEFENSE is what matters, and there's a real
#     counter-effect this proxy can't capture (facing strong offenses ->
#     shootouts/trailing scripts -> MORE pass volume -> more QB points).
#     Both argue the true QB effect is small, which is why the scale is
#     deliberately modest. Real external corroboration at the tail:
#     build_schedule_data.py documents CIN grading 32/32 (weakest
#     opponents), matching real-world "easiest schedule in football"
#     reporting.
#   player_pct -- real, per-player research, only where found.
QB_TOP20_ADJUSTMENTS = {
    "Josh Allen":       {"value": 355.9, "sched_pct": -1.67, "player_pct": 0.0,  "note": "Schedule-only (BUF SOS 12, 4 real road cold games)."},
    "Joe Burrow":       {"value": 324.0, "sched_pct":  2.60, "player_pct": 0.0,  "note": "Schedule-only. CIN grades 32/32 -- the real easiest schedule by this proxy, externally corroborated. Health verified directly earlier in this pass (confirmed practicing/connecting with Chase, Aug 2026)."},
    "Trevor Lawrence":  {"value": 315.3, "sched_pct": -0.50, "player_pct": 0.0,  "note": "Schedule-only (JAX SOS 17)."},
    "Lamar Jackson":    {"value": 311.4, "sched_pct":  1.63, "player_pct": 0.0,  "note": "Schedule-only (BAL SOS 28, real easy end). FLAGGED, not priced: John Harbaugh was really fired Jan 2026 after 18 seasons and a new HC/OC (Declan Doyle) is still building trust -- real scheme uncertainty, but no specific evidence found that Jackson's own output drops, so no player_pct applied."},
    "Caleb Williams":   {"value": 305.0, "sched_pct": -1.26, "player_pct": 3.2,  "note": "Real, user-directed to sit above Herbert, supported by real data: higher real 2025 passing EPA than Herbert (42.2 vs 16.1) despite a worse CPOE, and his real -3.51 CPOE is BELOW expectation -- positive regression room, the mirror image of Maye. Half Herbert's real INT rate (1.23% vs 2.54%). Real weapon upgrades already built this session: Odunze (00-0039919, +15.9%) and Burden (00-0040735, +54.2%) in year 2 of Ben Johnson's scheme."},
    "Jalen Hurts":      {"value": 304.0, "sched_pct": -3.01, "player_pct": 0.0,  "note": "Schedule-only, but the schedule hit independently reinforces a real per-player case left unpriced: PHI has the real 3rd-hardest schedule, stacking on a real new OC (Sean Mannion replacing Kevin Patullo), A.J. Brown's real trade to NE, and a real career-low rushing season (421 yds/8 TD in 2025 vs 630 yds/14 TD in 2024) in the exact skill driving his fantasy value."},
    "Brock Purdy":      {"value": 302.1, "sched_pct": -0.69, "player_pct": 0.0,  "note": "Schedule-only (SF SOS 15). Verified on the field and sharp in camp."},
    "Jayden Daniels":   {"value": 301.3, "sched_pct": -1.85, "player_pct": 0.0,  "note": "Schedule-only (WAS SOS 8). Real 2025 was injury-wrecked (7 games: knee/hamstring/elbow/ankle) but verified fully healthy and mobile, 'no questions surrounding his health' -- Book already prices the bounce-back, no player_pct added."},
    "Drake Maye":       {"value": 300.0, "sched_pct":  0.08, "player_pct": -6.7, "note": "Real, decisive regression case: his real 2025 fired all three classic unsustainable-QB markers at once -- CPOE +10.78 (vs +2.84 in 2024, and more than double any other top-20 QB: Burrow 4.65, Hurts 3.18, Herbert 1.57, Williams -3.51), TD rate 4.44% -> 6.30%, INT rate 2.96% -> 1.63%, with passing EPA swinging -26.5 -> +165.2. CPOE that extreme is among the most regression-prone stats in football. A.J. Brown's real arrival at NE is the real offset keeping this from going lower. Even at 300.0 he sits ~15% under his real 2025 output (~353 fantasy pts); Book was already discounting ~9%."},
    "Patrick Mahomes":  {"value": 299.8, "sched_pct": -0.31, "player_pct": 0.0,  "note": "Schedule-only (KC SOS 18) -- but this originally shipped with a blank rationale that missed a real, major fact: Mahomes suffered a real torn ACL and LCL (Dec 14, 2025 vs. LAC), with surgery and a real ~9-month recovery timeline landing right at Week 1 (Sept 10, 2026). Real, current, encouraging trajectory: cleared to practice Jul 24, 2026, walking unassisted by March, HC Andy Reid said he'd 'never bet against' him starting Week 1. No further number change proposed -- Book's 300.68 already looks like it reasonably prices this real, well-covered injury in -- but the note is corrected to actually disclose what was checked rather than leave it blank. Real, related context: KC signed Kenneth Walker III (00-0038134, see that RB entry) specifically to feature him and protect the run game while Mahomes ramps back."},
    "Jared Goff":       {"value": 296.8, "sched_pct":  1.21, "player_pct": 0.0,  "note": "Schedule-only (DET SOS 24, dome)."},
    "Bo Nix":           {"value": 295.8, "sched_pct":  0.27, "player_pct": 0.0,  "note": "Schedule-only (DEN SOS 21)."},
    "Justin Herbert":   {"value": 292.0, "sched_pct": -0.34, "player_pct": -2.3, "note": "Companion to Caleb Williams' entry. Real: 13 real INTs in 2025 (2.54% rate, double Williams') is both a direct fantasy penalty and a real regression marker. Real LAC losses: Keenan Allen departed to IND (122 real 2025 targets, see CURRENT_TEAM_OVERRIDES) and Najee Harris to NYG, plus real LT Rashawn Slater missing half of camp injured."},
    "Baker Mayfield":   {"value": 290.5, "sched_pct":  0.66, "player_pct": 0.0,  "note": "Schedule-only (TB SOS 22)."},
    "Dak Prescott":     {"value": 284.7, "sched_pct": -2.23, "player_pct": 0.0,  "note": "Schedule-only (DAL SOS 6, real hard end). No negative camp reports found."},
    "Tyler Shough":     {"value": 278.2, "sched_pct":  2.18, "player_pct": 0.0,  "note": "Schedule-only (NO SOS 29, real easy end). Real, verified rapport with Olave (00-0037239) -- already reflected in that WR entry, not double-counted here."},
    "Jaxson Dart":      {"value": 277.9, "sched_pct": -1.46, "player_pct": 0.0,  "note": "Schedule-only (NYG SOS 10)."},
    "Matthew Stafford": {"value": 275.3, "sched_pct": -2.35, "player_pct": 0.0,  "note": "Schedule-only (LAR SOS 5, real hard end)."},
    "Kyler Murray":     {"value": 268.9, "sched_pct": -3.36, "player_pct": 0.0,  "note": "Schedule-only, the largest real penalty in the group: MIN grades SOS 1 -- the real hardest schedule by this proxy. Real, confirmed starter (won a real 2-week camp competition over J.J. McCarthy), already the basis for Jefferson's (00-0036322) boost."},
    "Malik Willis":     {"value": 268.3, "sched_pct": -3.22, "player_pct": 0.0,  "note": "Schedule-only (MIA SOS 4 plus 4 real road cold games). Real, confirmed starter after Tua's real release; Book already prices the real downgrade conservatively."},
}


def build_qb_book_projections() -> pd.DataFrame:
    """Builds Book-anchored QB rows (see QB_TOP20_ADJUSTMENTS). No QB model
    is run -- model_projection_points stays NaN and the value lands in
    model_projection_points_fallback, preserving the module docstring's
    real QB-model exclusion while still giving QBs a usable number.

    Keyed on player_name, consistent with NAME_KEYED_FALLBACK_OVERRIDES and
    with how Home.py itself merges this file onto master_players.csv."""
    mp = pd.read_csv(MASTER_PLAYERS_PATH)
    mp = mp[mp["position"].eq("QB")].copy()
    book = pd.read_csv(SPORTSBOOK_PATH)
    book = book[book["position"].eq("QB")][["player_name", "sportsbook_half_ppr_points"]]

    qb = mp[["player_id", "player_name", "position"]].drop_duplicates("player_name").merge(
        book, on="player_name", how="left"
    )
    qb["gsis_id"] = np.nan
    qb["model_ppg"] = np.nan
    qb["model_projection_points"] = np.nan  # no QB model is run, by design
    for col in ("team_changed", "competitor_departed", "competitor_arrived"):
        qb[col] = False
    qb["continuity_note"] = None

    has_book = qb["sportsbook_half_ppr_points"].notna()
    qb["model_projection_status"] = np.where(has_book, "book_anchored", "no_book_price")
    qb["model_projection_points_fallback"] = qb["sportsbook_half_ppr_points"]
    qb["model_projection_fallback_note"] = np.where(
        has_book,
        "Book-anchored QB value (sportsbook_half_ppr_points), used directly -- no QB model is run "
        "(see QB_TOP20_ADJUSTMENTS). Outside the top 20 by Book, no manual adjustment is applied.",
        None,
    )

    n_adj = 0
    for player_name, adj in QB_TOP20_ADJUSTMENTS.items():
        mask = qb["player_name"].eq(player_name)
        if not mask.any():
            print(f"  WARNING: QB adjustment for '{player_name}' matched no row in master_players.csv")
            continue
        bk = qb.loc[mask, "sportsbook_half_ppr_points"].iloc[0]
        qb.loc[mask, "model_projection_points_fallback"] = adj["value"]
        qb.loc[mask, "model_projection_fallback_note"] = (
            f"Book {bk} -> {adj['value']} (schedule {adj['sched_pct']:+.2f}%, "
            f"player {adj['player_pct']:+.1f}%). {adj['note']}"
        )
        n_adj += 1

    qb = qb.drop(columns=["sportsbook_half_ppr_points"])
    print(f"QB: {len(qb)} live players, {int(has_book.sum())} with a real Book price, {n_adj} top-20 manually adjusted (no QB model run, by design)")
    return qb


def main() -> None:
    e6 = pd.read_csv(E6_PATH)
    training_df = build_training_frame()

    all_projections = []
    for position in POSITIONS:
        crosswalk = _build_crosswalk([position])
        crosswalk["model_projection_status"] = "no_crosswalk_match"

        candidates = crosswalk[crosswalk["gsis_id"].notna()].copy()
        baseline = _recency_weighted_baseline_live(e6, candidates["gsis_id"], position)
        candidates = candidates.merge(baseline, left_on="gsis_id", right_on="player_id", how="left", suffixes=("", "_bl"))
        candidates["model_projection_status"] = np.where(
            candidates["n_prior_seasons"].fillna(0).gt(0), "projected", "insufficient_history"
        )
        # Write the real status back onto `crosswalk` itself -- computing it
        # only on `candidates` (a separate copy) and never updating the
        # source frame `full_position` selects from below was a real bug
        # caught in review: every row showed the stale "no_crosswalk_match"
        # default, including players (e.g. Jonathan Taylor) who plainly got
        # a real projection.
        # Keyed on gsis_id, not player_id -- player_id (FantasyPros' own
        # numeric ID) is null for a real, pre-existing pair of players in
        # master_players.csv (Bam Knight, Kenneth Gainwell), and pandas
        # treats multiple NaN index keys as matching each other, which
        # corrupted status AND (before the fix above) projected values
        # across unrelated players. gsis_id is never null in `candidates`
        # (that's exactly its filter condition), so it's the safe key --
        # except master_players.csv ALSO has a real duplicate listing
        # ("Kenny Gainwell" / "Kenneth Gainwell", two rows, same real
        # person, same gsis_id), so still dedupe before indexing.
        status_map = candidates.drop_duplicates("gsis_id").set_index("gsis_id")["model_projection_status"]
        crosswalk["model_projection_status"] = crosswalk["gsis_id"].map(status_map).fillna(crosswalk["model_projection_status"])
        # Status for the whole crosswalk-matched set is known now (real
        # player, real prior-season check) -- write it back so unprojected
        # matched rows (insufficient_history) still get a precise status in
        # the output, not silently dropped. Only "projected" rows continue
        # through feature-building/model-fit below.
        matched = candidates[candidates["model_projection_status"].eq("projected")].copy()

        efficiency = _load_efficiency_by_season_player()
        eff_2025 = efficiency[efficiency["season"].eq(2025)].rename(
            columns={c: f"{c}_prior" for c in POSITION_EFFICIENCY_COLS[position]}
        )
        matched = matched.merge(eff_2025, left_on="gsis_id", right_on="player_id", how="left", suffixes=("", "_eff"))

        team_ctx = _load_team_context()
        team_2026 = team_ctx[team_ctx["season"].eq(TARGET_SEASON)]
        matched = matched.merge(team_2026, on="team", how="left", suffixes=("", "_tm"))

        features = ["recency_weighted_ppg_baseline"] + [f"{c}_prior" for c in POSITION_EFFICIENCY_COLS[position]] + TEAM_CONTEXT_COLS
        if position in POSITIONS_WITH_TREND_LAYER:
            trend = _load_snap_trend_by_season_player()
            trend_2025 = trend[trend["season"].eq(2025)].rename(columns={c: f"{c}_prior" for c in TREND_COLS_RAW})
            matched = matched.merge(trend_2025, left_on="gsis_id", right_on="player_id", how="left", suffixes=("", "_tr"))
            matched["has_snap_trend"] = matched["last_5_games_snap_pct_prior"].notna().astype(float)
            features = features + [f"{c}_prior" for c in TREND_COLS_RAW] + ["has_snap_trend"]

        model, medians = _train_production_model(e6, position, features, training_df)
        for c in features:
            matched[c] = pd.to_numeric(matched[c], errors="coerce")
        x = matched[features].fillna(medians)
        matched["model_ppg"] = model.predict(x)
        matched["model_projection_points"] = (matched["model_ppg"] * GAMES_PER_SEASON).round(1)

        n_total = len(crosswalk)
        n_matched = len(crosswalk[crosswalk["gsis_id"].notna()])
        n_projected = len(matched)
        print(f"{position}: {n_total} live players, {n_matched} crosswalk-matched, {n_projected} projected (n_prior_seasons>0)")

        # Write EVERY crosswalk row for this position, not just the
        # projected ones -- per review, a blank cell downstream needs to
        # distinguish "couldn't identify this player" (no_crosswalk_match)
        # from "identified, but not enough real career history yet"
        # (insufficient_history) rather than looking identical to both a
        # real projection and to each other. Only "projected" rows carry a
        # real model_ppg/model_projection_points value.
        #
        # Joined on gsis_id, NOT player_id (the FantasyPros numeric ID) --
        # a second real bug caught in review: master_players.csv has real
        # players (Bam Knight, Kenneth Gainwell) with a NULL player_id, and
        # pandas merge treats multiple NaN keys as matching each other, so
        # an on="player_id" merge cross-contaminated the two of them with
        # each other's real model_ppg values while still showing
        # no_crosswalk_match. gsis_id is never null for a matched row (it's
        # exactly the "matched" filter condition), so it's the safe key --
        # deduped on gsis_id first (same Gainwell double-listing as above)
        # to avoid a many-to-many fan-out if the duplicate happened to land
        # in the "projected" bucket on both sides.
        full_position = crosswalk[["player_id", "player_name", "position", "gsis_id", "model_projection_status"]].merge(
            matched.drop_duplicates("gsis_id")[["gsis_id", "model_ppg", "model_projection_points"]], on="gsis_id", how="left"
        )

        # Step 1 + Step 4 (roster continuity flags): computed on the
        # already-deduped `crosswalk` (has real "team" from master_players
        # and "model_projection_status"), not `matched`, since
        # compute_roster_continuity() needs the team_2025/current_team
        # comparison for every projected player.
        continuity = compute_roster_continuity(crosswalk.drop_duplicates("gsis_id"), position)
        if not continuity.empty:
            full_position = full_position.merge(continuity, on="gsis_id", how="left")
        for col in ("team_changed", "competitor_departed", "competitor_arrived"):
            if col not in full_position.columns:
                full_position[col] = False
            full_position[col] = full_position[col].fillna(False)
        if "continuity_note" not in full_position.columns:
            full_position["continuity_note"] = None

        n_flagged = int((full_position["team_changed"] | full_position["competitor_departed"] | full_position["competitor_arrived"]).sum())
        print(f"  roster continuity: {n_flagged} projected players flagged (team change or competitor change)")

        full_position = compute_rookie_fallback(full_position, position, matched["model_projection_points"])

        all_projections.append(full_position)

    result = pd.concat(all_projections, ignore_index=True)
    result = apply_model_corrections(result)
    result = apply_manual_fallback_placeholders(result)
    result = apply_name_keyed_fallback_overrides(result)
    # QB rows appended AFTER the RB/WR/TE correction/placeholder passes --
    # those all key on gsis_id or on RB/WR/TE player names, so none of them
    # should touch (or be able to touch) a QB row. Appending last makes that
    # structurally true rather than merely likely.
    result = pd.concat([result, build_qb_book_projections()], ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nProjections written: {OUTPUT_PATH} ({len(result)} rows)")

    print("\nSpot check -- top 5 by model_projection_points per position:")
    for position in POSITIONS:
        top = result[result["position"].eq(position)].sort_values("model_projection_points", ascending=False).head(5)
        print(f"\n{position}:")
        print(top[["player_name", "model_ppg", "model_projection_points"]].to_string(index=False))


if __name__ == "__main__":
    main()
