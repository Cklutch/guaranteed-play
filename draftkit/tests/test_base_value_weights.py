"""Base Value weight stability sanity check (2026-08-27): turns
research/validation_v1/data/base_value_sensitivity_2026-08-21.csv's
one-off conclusion (CLAUDE.md: "the board is stable, Spearman rho > 0.999,
across +-25% variation in any single weight") into an automated,
repeatable check instead of a frozen CSV nobody re-runs.

Plain assert-based, no pytest -- matches this repo's established convention.
Also runnable directly for a differentiated exit code (e.g. before trusting
a freshly re-scored board, or from a pre-commit hook):

    python -m draftkit.tests.test_base_value_weights

    Exit 0 -- everything passed cleanly.
    Exit 1 -- Check A failed: a BASE_VALUE_* weight drifted from the
              validated baseline. Re-run the sensitivity analysis (or at
              least Check B against the new value) before trusting anything.
    Exit 2 -- Check B failed on a row the CSV itself recorded as
              near-baseline (rho > 0.999 there), OR an extreme/stress row
              regressed beyond RHO_DEGRADATION_TOLERANCE past what the CSV
              already recorded for that exact row. Either way: a REAL
              regression the sensitivity analysis did not already account
              for. Investigate now.
    Exit 3 -- Check B is otherwise clean, but an extreme/stress row's top-N
              boundary reshuffled while its rho stayed consistent with what
              the CSV already recorded (see the Josh Allen/Egbuka note
              below). Non-blocking by design, but never silently swallowed.

Two checks:
  A. The live BASE_VALUE_* constants in draft_analysis.py still match the
     baseline values the sensitivity analysis actually validated (the
     is_baseline rows in the CSV). Instant, no player data needed.
  B. Re-running the CSV's own perturbation grid against TODAY's actual
     scored pool (not the frozen 2026-08-21 snapshot): every near-baseline
     row (the CSV recorded rho > 0.999 there) must still clear 0.999 today,
     and every extreme/stress row (the CSV's own 4 rows that never cleared
     0.999 even in the original run) must not regress further than
     RHO_DEGRADATION_TOLERANCE past what was already recorded for that
     exact row. This catches what A can't: code weights unchanged, but the
     underlying DATA (projections, risk, market) drifted enough that the
     same weights are no longer actually safe.

Known, accepted edge case (do not "fix" by loosening RHO_DEGRADATION_TOLERANCE
or silently dropping the check): at the CSV's most extreme stress value,
MARKET_MAX_SWING=60 (a 5x blowout nobody would ever configure), Josh Allen
and Emeka Egbuka swap at the #29/#30 boundary. They sit ~0.09 points apart
in baseline Base Value -- an effectively-tied pair sitting exactly on the
cutoff. rho itself is in line with what the CSV already recorded for this
same row (0.9926 today vs. 0.9947 on 2026-08-21 -- both "extreme, expected
degradation," not a new regression); the boundary swap alone is expected
noise under an unrealistic input, not evidence the board is unsafe --
surfaced as a WARN (exit 3), not a hard failure, because EXTREME_TOP30_STRICT
is False by default. Set it True to make even this count as a hard failure.

Deliberately NOT a flat "rho > 0.999" bar for extreme rows: the CSV's own 4
designated stress rows never cleared 0.999 either (that's why they're the
stress test -- see the table above). Whether an extreme row counts as
"error" is instead relative to what the CSV itself recorded for that same
row (RHO_DEGRADATION_TOLERANCE below); only a NEW regression beyond what
was already measured and accepted on 2026-08-21 counts as a hard failure.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

import draftkit.draft_analysis as da  # noqa: E402

SENSITIVITY_CSV = REPO_ROOT / "research/validation_v1/data/base_value_sensitivity_2026-08-21.csv"

# CSV `parameter` name -> the draft_analysis.py module constant it perturbs.
# calculate_base_value_score() reads these as module globals (not function
# arguments), so a perturbation is applied by temporarily overwriting the
# attribute -- the exact same mechanism the original analysis must have used.
WEIGHT_ATTR_BY_PARAM = {
    "VOR_WEIGHT": "BASE_VALUE_VOR_WEIGHT",
    "PROJECTION_BONUS_MAX": "BASE_VALUE_PROJECTION_BONUS_MAX",
    "MARKET_MAX_SWING": "BASE_VALUE_MARKET_MAX_SWING",
    "RISK_MAX_PENALTY": "BASE_VALUE_RISK_MAX_PENALTY",
}

# A row counts as "reasonable" (must clear the documented rho > 0.999
# guarantee) vs. "extreme/stress" using the CSV's OWN recorded overall_rho
# for that row, not a fixed +-pct cutoff -- the grid spacing isn't uniform
# across parameters (VOR_WEIGHT's near rows are +-25%, but
# PROJECTION_BONUS_MAX/MARKET_MAX_SWING/RISK_MAX_PENALTY's are +-50%), so a
# pct-based split misclassifies perfectly-stable +-50% rows as "extreme."
# The CSV's recorded rho already tells you which 4 rows (one per parameter)
# were the deliberate stress test: all 12 near-baseline rows recorded
# rho > 0.999; all 4 stress rows recorded rho <= 0.999. That split is exact.
REASONABLE_RHO_THRESHOLD = 0.999

# For an extreme/stress row (recorded rho already <= 0.999 in the CSV),
# today's rho is allowed to sit this far below what was ORIGINALLY recorded
# for that same row before counting as a NEW regression (error) rather than
# consistent-with-the-original-finding (at worst a warn). Not compared
# against 0.999 directly -- none of the CSV's 4 stress rows cleared that
# bar even in the original analysis, so holding extreme rows to it would
# flag every one of them as failing, always, by construction.
RHO_DEGRADATION_TOLERANCE = 0.01

# The stability guarantee is about "the board" -- draftable players -- not
# literally every row calculate_base_value_score() touches. Confirmed by
# measurement: rho over the full ~3900-player universe (including hundreds
# of replacement-level scrubs whose rank order is inherently noisy near a
# ~0 VOR baseline, and irrelevant to anyone using the app) reads ~0.996 for
# a perturbation the CSV recorded at 0.9998; restricting to a draftable-
# sized pool recovers >0.999. 300 matches live_draft.build_candidate_pool's
# own definition of the draftable pool -- not a new number invented here.
DRAFTABLE_POOL_SIZE = 300
TOP_N_STABILITY = 30

# Whether a top-N boundary swap on a stress row (rho itself still fine)
# escalates to a hard failure ("error") or stays a non-blocking WARN. False
# (default) matches the known Josh Allen/Egbuka boundary-noise case above:
# surfaced every run, never silently dropped, but doesn't block on its own
# unless rho itself also regresses. Flip to True for zero tolerance.
EXTREME_TOP30_STRICT = False


@dataclass
class CheckResult:
    name: str
    severity: str  # "pass" | "warn" | "error" -- tri-state, not a bool +
    # severity combo: a row can be non-blocking yet still worth flagging
    # (warn), which a plain passed/failed boolean can't represent without
    # one of the two fields silently overriding the other.
    detail: str

    @property
    def passed(self) -> bool:
        return self.severity != "error"


def _load_sensitivity_rows():
    with open(SENSITIVITY_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_a() -> CheckResult:
    """Pure constant comparison against draft_analysis.py -- no data, no
    scoring run. If this fails, a weight was edited without re-running the
    sensitivity analysis, full stop."""
    rows = _load_sensitivity_rows()
    baseline = {r["parameter"]: float(r["value"]) for r in rows if r["is_baseline"] == "True"}
    if set(baseline) != set(WEIGHT_ATTR_BY_PARAM):
        return CheckResult(
            "A: weights match validated baseline", "error",
            f"sensitivity CSV's baseline rows don't cover the expected 4 parameters, "
            f"got: {sorted(baseline)}",
        )

    drifted = []
    for param, attr in WEIGHT_ATTR_BY_PARAM.items():
        live_value = getattr(da, attr)
        expected = baseline[param]
        if live_value != expected:
            drifted.append(f"  {attr}: live={live_value} vs validated={expected} (param {param})")

    if drifted:
        return CheckResult(
            "A: weights match validated baseline", "error",
            "Weight(s) changed since the last sensitivity analysis:\n" + "\n".join(drifted)
            + f"\n  -> Re-run the sensitivity analysis and regenerate {SENSITIVITY_CSV.name} "
              "before trusting the board.",
        )
    return CheckResult(
        "A: weights match validated baseline", "pass",
        f"All 4 weights match the validated baseline exactly: {baseline}",
    )


def _check_b_all() -> list[CheckResult]:
    """Replay the CSV's own perturbation grid against today's actual scored
    pool. Catches data drift (new projections/risk/market) invalidating a
    previously-safe set of weights, which _check_a() can't see."""
    rows = _load_sensitivity_rows()

    board = da.build_recommendation_rankings_df()
    if board.empty:
        return [CheckResult(
            "B: perturbation grid", "error",
            "build_recommendation_rankings_df() returned nothing to test against",
        )]

    def score_with(overrides: dict) -> pd.Series:
        saved = {attr: getattr(da, attr) for attr in overrides}
        try:
            for attr, value in overrides.items():
                setattr(da, attr, value)
            scored = da.calculate_base_value_score(board)
        finally:
            for attr, value in saved.items():
                setattr(da, attr, value)
        return scored["base_value_score"]

    full_baseline_score = score_with({})
    draftable_idx = (
        full_baseline_score.rank(ascending=False, method="first")
        .loc[lambda r: r <= DRAFTABLE_POOL_SIZE].index
    )
    baseline_score = full_baseline_score.loc[draftable_idx]
    baseline_topn = set(
        baseline_score.rank(ascending=False, method="first").loc[lambda r: r <= TOP_N_STABILITY].index
    )
    baseline_by_param = {r["parameter"]: float(r["value"]) for r in rows if r["is_baseline"] == "True"}

    results = []
    for row in rows:
        if row["is_baseline"] == "True":
            continue
        param, value = row["parameter"], float(row["value"])
        attr = WEIGHT_ATTR_BY_PARAM[param]
        baseline_value = baseline_by_param[param]
        pct_off = abs(value - baseline_value) / baseline_value
        recorded_rho = float(row["overall_rho"])
        is_reasonable = recorded_rho > REASONABLE_RHO_THRESHOLD

        perturbed_score = score_with({attr: value}).loc[draftable_idx]
        rho = baseline_score.corr(perturbed_score, method="spearman")
        perturbed_topn = set(
            perturbed_score.rank(ascending=False, method="first").loc[lambda r: r <= TOP_N_STABILITY].index
        )
        topn_stable = perturbed_topn == baseline_topn

        if is_reasonable:
            # Near-baseline row: the CSV recorded rho > 0.999 here, so today
            # must too -- this is the actual documented guarantee. Any miss
            # is a hard error, no warn tier -- this IS the range the
            # sensitivity analysis certified as safe.
            severity = "pass" if (rho > REASONABLE_RHO_THRESHOLD and topn_stable) else "error"
            basis = f"must exceed {REASONABLE_RHO_THRESHOLD} (near-baseline row)"
        else:
            # Stress row: the CSV itself never cleared 0.999 here either --
            # judge against how far rho sits from what was ALREADY recorded
            # for this exact row, not an absolute bar it was never held to.
            # A genuine regression is always an error; a top-N boundary
            # swap with rho otherwise in line is a non-blocking warn unless
            # EXTREME_TOP30_STRICT asks for zero tolerance instead.
            regressed = rho < recorded_rho - RHO_DEGRADATION_TOLERANCE
            if regressed:
                severity = "error"
            elif not topn_stable:
                severity = "error" if EXTREME_TOP30_STRICT else "warn"
            else:
                severity = "pass"
            basis = f"must not regress > {RHO_DEGRADATION_TOLERANCE} below the recorded {recorded_rho:.4f}"

        detail = (
            f"{param}={value} ({pct_off:.0%} off baseline, {'reasonable' if is_reasonable else 'EXTREME'}, "
            f"{basis}): rho={rho:.4f} (CSV recorded {recorded_rho:.4f}), "
            f"top-{TOP_N_STABILITY} stable={topn_stable}"
        )
        if not topn_stable:
            swapped_in = board.loc[list(perturbed_topn - baseline_topn), "player_name"].tolist()
            swapped_out = board.loc[list(baseline_topn - perturbed_topn), "player_name"].tolist()
            detail += f"\n    swapped in={swapped_in}, swapped out={swapped_out}"

        results.append(CheckResult(f"B: {param}={value}", severity, detail))

    return results


def test_a_live_weights_match_validated_baseline():
    r = _check_a()
    assert r.passed, r.detail
    print(f"A PASS -- {r.detail}")


def test_b_live_board_stable_under_the_validated_perturbation_grid():
    results = _check_b_all()
    errors = [r for r in results if r.severity == "error"]
    warnings_ = [r for r in results if r.severity == "warn"]
    assert not errors, "Base Value weight stability check failed:\n" + "\n".join(
        f"  - {r.detail}" for r in errors
    )
    if warnings_:
        print("B: non-blocking WARN(s) on extreme-only perturbations (see module docstring):")
        for r in warnings_:
            print(f"  - {r.detail}")
    print(f"B PASS -- board stable across {len(results) - len(warnings_)} of {len(results)} recorded perturbations "
          f"({len(warnings_)} known extreme-case warning(s))")


_TAG_BY_SEVERITY = {"pass": "PASS", "warn": "WARN", "error": "FAIL"}


def main() -> int:
    a = _check_a()
    print(f"[{_TAG_BY_SEVERITY[a.severity]}] {a.name}\n  {a.detail}")
    if a.severity == "error":
        print("\nStopping before Check B: fix the weight drift first.")
        return 1

    print("\nReplaying sensitivity grid against today's top-{} candidate pool...".format(DRAFTABLE_POOL_SIZE))
    b_results = _check_b_all()
    hard_fail = any(r.severity == "error" for r in b_results)
    warn_only = (not hard_fail) and any(r.severity == "warn" for r in b_results)
    for r in b_results:
        print(f"\n[{_TAG_BY_SEVERITY[r.severity]}] {r.name}\n  {r.detail}")

    print("\n" + "=" * 70)
    if hard_fail:
        print("RESULT: Check B failed on a realistic perturbation. Investigate before trusting the board.")
        return 2
    if warn_only:
        print("RESULT: Check B passed on all realistic perturbations. One or more EXTREME-only cases "
              "were flagged (see WARN above) -- known boundary noise, not a live-board issue, but not "
              "silently ignored.")
        return 3
    print("RESULT: All checks passed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
