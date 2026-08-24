"""Backtest report: rookie_projection_model_v1 vs. real confirmed outcomes.

The spec's own Step 5 ("Backtest against last year's rookie class before
trusting current-year output"), extended to three real classes (2023-2025).
Reuses build_rookie_projections() against data/raw/backtest_rookie_inputs.csv
(see build_backtest_rookie_inputs.py) -- same formulas, same real
confirmed-tier join (rb_archetypes.csv/wr_archetypes.csv) already built into
build_rookie_projections.py.

Real methodological caveat, reported not hidden: team_context comes from
CURRENT risk_variables.csv (this season's real offense_environment_score/
schedule_weather_venue_score), not the team's real situation in the
player's actual draft year -- team offense quality changes year to year
(coaching/roster turnover), so a 2023 draftee's team-context component here
reflects today's team, not 2023's. No historical risk_variables.csv exists
in this repo to correct this. Flagged per-player where the current team is
known to have changed significantly is out of scope for this pass -- this
is reported as an honest limitation of the backtest, not silently ignored.

Real data status: as of this build, only the 2023 class has real Combine
measurables (rb_combine_measurables_2023.csv); no class has real dominator
data yet (needs team-season offense totals for 2018-2022, not available in
this repo). So this backtest currently validates draft-capital + (for 2023
RBs) real athletic-score signal only -- not yet the full formula. Re-run
once more real data lands.

Usage:
    python -m draftkit.scripts.run_rookie_backtest
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.scripts.build_backtest_rookie_inputs import OUTPUT_CSV as BACKTEST_INPUTS_CSV  # noqa: E402
from draftkit.scripts.build_rookie_projections import build_rookie_projections  # noqa: E402

# RB tier hierarchy (bellcow > committee_back > handcuff > unconfirmed) --
# used to judge "close" misses (adjacent tier) vs. "far" misses, not just
# exact-match accuracy. RB's confirmed archetype is one of 4 real usage
# tiers (rb_archetypes.py, migrated 2026-08-15 -- primary is a usage tier
# only, Receiving/Goal-Line/Explosive are secondary lean tags now, never a
# real confirmed archetype_primary value); the projection model only ever
# outputs bellcow/committee pre-draft, so "handcuff" is reported separately
# ("other"), not scored as a flat miss -- the projection model was never
# designed to predict that finer real tier pre-draft. WR's own hierarchy
# (alpha > boom_bust/high_floor > complementary) uses exact-tag matching
# further down instead of a coarse-bucket dict -- see wr_matches below.
RB_COARSE_TIER = {
    "bellcow": "bellcow", "committee_back": "committee",
    "handcuff": "other", "unconfirmed": "unconfirmed",
    # blend_rookie_tag() returns the PROJECTED tier's own vocabulary
    # ("bellcow"/"committee", not "committee_back") for low-sample-weight
    # blended rows -- real gap surfaced once the 72-player backfill scope
    # added enough low-confidence blended RBs to hit this path; "bellcow"
    # already round-trips via the confirmed-vocabulary entry above,
    # "committee" needed its own entry or it silently mapped to NaN.
    "committee": "committee",
}


def run_backtest() -> pd.DataFrame:
    board = build_rookie_projections(inputs_csv=BACKTEST_INPUTS_CSV)
    return board


def main() -> int:
    board = run_backtest()
    print(f"{len(board)} real backtest players projected\n")

    print("Real status breakdown (projected = never played, blended = partial "
          "real sample, confirmed = full real sample):")
    print(board["status"].value_counts().to_string())
    print()

    wr = board[board["position"] == "WR"].copy()
    rb = board[board["position"] == "RB"].copy()

    if not wr.empty:
        wr_matches = wr[wr["status"] != "projected"].copy()
        wr_matches["match"] = wr_matches["projected_tier"] == wr_matches["tag"]
        print(f"WR: {len(wr_matches)}/{len(wr)} have real confirmed/blended data to compare against.")
        if not wr_matches.empty:
            print(f"  Exact tier match: {wr_matches['match'].sum()}/{len(wr_matches)}")
            print(wr_matches[["player_name", "projected_tier", "tag", "status", "composite"]].to_string(index=False))
    print()

    if not rb.empty:
        rb_matches = rb[rb["status"] != "projected"].copy()
        rb_matches["confirmed_coarse"] = rb_matches["tag"].map(RB_COARSE_TIER)
        rb_matches["match"] = rb_matches["projected_tier"] == rb_matches["confirmed_coarse"]
        print(f"RB: {len(rb_matches)}/{len(rb)} have real confirmed/blended data to compare against.")
        if not rb_matches.empty:
            print(f"  Coarse tier match (bellcow/committee only): {rb_matches['match'].sum()}/{len(rb_matches)}")
            print(rb_matches[["player_name", "projected_tier", "tag", "confirmed_coarse", "status", "composite"]].to_string(index=False))
    print()

    no_real_data = board[board["status"] == "projected"]
    print(f"{len(no_real_data)}/{len(board)} real players have NO confirmed/blended NFL data yet "
          f"(no row in rb_archetypes.csv/wr_archetypes.csv for this player_name -- check name-matching "
          f"before assuming they're simply inactive).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
