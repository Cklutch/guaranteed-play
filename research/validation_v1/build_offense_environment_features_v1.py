"""
Step 13: team offensive environment, as a MEAN-REVERSION signal.

Persistence was measured before this was designed (`stats_team`, 448
team-seasons 2012-2025):

    pass rate  r=+0.401    plays/game r=+0.356
    EPA/game   r=+0.403    TDs/game   r=+0.350

For context: garbage time +0.17 (rejected on this test), route
participation +0.51 (built, failed). At r~0.40, roughly **60% of team
offensive context is new each year** -- QB changes, coordinator changes,
personnel turnover.

That number is what the feature is built around. Twelve previous attempts
all claimed some version of "here is a player-quality signal ADP missed,"
and ADP kept already knowing. The claim here is different: **the market
over-extrapolates last season's team context, which mean-reverts hard.**
With r~0.40 the optimal forecast of next season is roughly 40% last season
plus 60% league average; if the market leans harder on last season than
that, skill players on last year's breakout offenses are systematically
overpriced. That is a hypothesis about the market's *forecasting error*,
not about hidden information -- the one framing not yet tested at team
level, and structurally the same shape as Step 4b's divergence features,
which produced the project's only near-miss.

Emits two deliberately separate families:

  LEVELS    -- prior-season team pace/pass-rate/scoring/efficiency.
               Expected to FAIL; included as the control that makes the
               reversion result interpretable rather than a lone number.
  REVERSION -- each metric minus that team's own trailing 3-year baseline.
               Positive = the offense outran its own recent norm and is a
               regression candidate. This is the actual hypothesis.

Known limitation, stated not buried: prior-season context is partly
obsolete on arrival. A team that swapped QB or coordinator has an offense
its trailing numbers no longer describe, and nothing in `stats_team` flags
that. This caps the ceiling and is the likeliest reason LEVELS fail.

Leakage: every output is shifted +1 season, and the 3-year baseline uses
only seasons strictly before the measured one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR

OUTPUT = VALIDATION_DIR / "data" / "offense_environment_team_seasons.csv"
SEASONS = tuple(range(2012, 2026))
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_reg_{season}.csv"

# Trailing window for a team's own norm. 3 seasons balances "enough history
# to be a baseline" against "recent enough to describe this roster."
BASELINE_YEARS = 3

METRICS = ["plays_pg", "pass_rate", "tds_pg", "epa_pg"]


def fetch_team_seasons() -> pd.DataFrame:
    """One row per team-season. Only 32 rows/season, so no caching needed."""
    frames = []
    for season in SEASONS:
        try:
            frames.append(pd.read_csv(BASE_URL.format(season=season), low_memory=False))
        except Exception as exc:  # a missing season should not kill the run
            print(f"  {season}: skipped ({exc.__class__.__name__})")
    if not frames:
        return pd.DataFrame()

    t = pd.concat(frames, ignore_index=True)
    t = t[t["games"].fillna(0) > 0].copy()
    plays = t["attempts"].fillna(0) + t["carries"].fillna(0)
    t["plays_pg"] = plays / t["games"]
    t["pass_rate"] = t["attempts"].fillna(0) / plays.replace(0, np.nan)
    t["tds_pg"] = (t["passing_tds"].fillna(0) + t["rushing_tds"].fillna(0)) / t["games"]
    t["epa_pg"] = (t["passing_epa"].fillna(0) + t["rushing_epa"].fillna(0)) / t["games"]
    return t[["season", "team"] + METRICS].sort_values(["team", "season"]).reset_index(drop=True)


def build_features(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy()

    for metric in METRICS:
        # Trailing baseline over the team's own PRIOR seasons only --
        # shift(1) before rolling so the measured season never enters its
        # own baseline.
        baseline = (
            out.groupby("team")[metric]
            .apply(lambda s: s.shift(1).rolling(BASELINE_YEARS, min_periods=2).mean())
            .reset_index(level=0, drop=True)
        )
        out[f"{metric}_baseline"] = baseline
        # Positive = outran its own recent norm = regression candidate.
        out[f"{metric}_vs_baseline"] = out[metric] - baseline

    # +1 shift: a 2024 team-season becomes a 2025 prior-season input.
    out["season"] = out["season"] + 1

    rename = {}
    for metric in METRICS:
        rename[metric] = f"prior_team_{metric}"
        rename[f"{metric}_vs_baseline"] = f"prior_team_{metric}_vs_baseline"
    out = out.rename(columns=rename)

    keep = ["season", "team"] + [f"prior_team_{m}" for m in METRICS] \
        + [f"prior_team_{m}_vs_baseline" for m in METRICS]
    return out[keep].reset_index(drop=True)


def main() -> None:
    raw = fetch_team_seasons()
    if raw.empty:
        raise SystemExit("No team-season data fetched.")
    feats = build_features(raw)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUTPUT, index=False)

    print(f"Team offensive environment written: {OUTPUT}")
    print(f"Rows: {len(feats)} | seasons (as prior-season input): "
          f"{int(feats.season.min())}-{int(feats.season.max())}")
    for col in feats.columns:
        if col.startswith("prior_team_"):
            print(f"  {col:42s} non-null={int(feats[col].notna().sum())}")

    latest = feats[feats.season == feats.season.max()]
    rev = "prior_team_epa_pg_vs_baseline"
    if rev in latest.columns and latest[rev].notna().any():
        print(f"\nBiggest regression candidates for {int(feats.season.max())} "
              "(2025 EPA/game most above their own 3-yr norm):")
        print(latest.nlargest(5, rev)[["team", "prior_team_epa_pg", rev]].round(3).to_string(index=False))
        print("\nBiggest bounce-back candidates (most BELOW their own norm):")
        print(latest.nsmallest(5, rev)[["team", "prior_team_epa_pg", rev]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
