"""
Player archetype assignment.

Assigns quantifiable, automatically-derived archetypes (dual-threat QB,
deep-threat WR, long-run RB, red-zone TE, etc.) from prior-season usage.

Design constraints, per research/archetype_engine_design.md:
  - Quantifiable and auto-assignable -- no hand-curated player lists.
  - Pre-draft safe -- every input is a prior_* column (prior season only).
  - Predictive intent, not descriptive labels -- the point is feature
    engineering, so this emits BOTH a hard label (readable, for the app)
    and continuous 0-1 membership scores (for the model). The continuous
    scores are the useful part; roles are gradients, not boxes, and hard
    labels throw away exactly the gradation that carries signal.

Thresholds are expressed as within-(season, position) PERCENTILES rather
than absolute cutoffs, because the NFL's usage baselines drift a lot over
1999-2025 (pass rates, snap counts, and target volumes are all much higher
now). An absolute "aDOT > 14 = deep threat" rule would label a different
share of the league in 2004 than in 2024; a percentile rule stays
comparable across eras, which is what "historically reproducible" requires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation_utils import VALIDATION_DIR

INPUT_PATH = VALIDATION_DIR / "predraft_validation_dataset_divergence_v1.csv"
OUTPUT_PATH = VALIDATION_DIR / "predraft_validation_dataset_archetypes_v1.csv"

# Minimum prior-season volume before a *rate* is meaningful. Below this, a
# player's aDOT / explosive rate / red-zone share is noise (3 carries for 40
# yards is not a "long run back"), so he gets no archetype rather than a
# spurious one.
#
# These were originally 20/20 and that was measurably too low: at 20 carries,
# three long runs reads as a 15% explosive rate, so the long-run archetype
# filled up with backups (Demercado at 24 carries, Homer at 21) while the
# median labeled back had 65 carries against a league median of 103. 75
# carries is roughly the 35th percentile of RB seasons -- a real rotational
# workload, and enough attempts for an explosive rate to mean something.
# Targets moved to 30 for the same reason on the receiving side.
MIN_TARGETS_FOR_RATE = 30
MIN_CARRIES_FOR_RATE = 75

# The QB archetypes split on rushing share, which makes "pocket passer" an
# inverse score -- and an inverse score rewards the mere ABSENCE of rushing.
# Without a volume floor it filled with backups who barely played (Nick
# Mullens, Cooper Rush, Philip Rivers) and, worse, labeled Anthony
# Richardson a pocket passer off an injury-shortened season, which inverts
# his actual profile. Roughly half a season of starter passing volume is
# the floor for the split to describe a real role.
MIN_PASSING_YARDS_FOR_QB = 1500

# Percentile cutoffs used across archetype rules.
HIGH = 0.70
VERY_HIGH = 0.80
LOW = 0.30

ARCHETYPE_SCORE_COLS = [
    "arch_dual_threat_qb",
    "arch_pocket_passer_qb",
    "arch_bellcow_rb",
    "arch_receiving_back_rb",
    "arch_early_down_rb",
    "arch_long_run_rb",
    "arch_goal_line_rb",
    "arch_deep_threat_wr",
    "arch_possession_wr",
    "arch_alpha_wr",
    "arch_red_zone_te",
    "arch_receiving_te",
    "arch_blocking_te",
]


def _pct(df: pd.DataFrame, col: str) -> pd.Series:
    """Percentile rank within (season, position). Missing stays missing."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = pd.to_numeric(df[col], errors="coerce")
    return values.groupby([df["season"], df["position"]]).rank(pct=True)


def _squash(*components: pd.Series, min_components: int = 2) -> pd.Series:
    """
    Combine percentile components into a 0-1 membership score by averaging
    whichever are present.

    Missing components are skipped rather than treated as zero -- absent
    data is not evidence of a low value. But a score built from ONE
    component must not compete on equal footing with one built from three:
    that let players with snap share but no play-by-play data score a
    perfect 1.0 bellcow while Christian McCaffrey (288 carries, 142
    targets) ranked below them. So a row needs at least `min_components`
    present, or it scores NaN and stays unclassified.

    Single-component archetypes pass min_components=1 explicitly.
    """
    frame = pd.concat(components, axis=1)
    enough = frame.notna().sum(axis=1) >= min(min_components, frame.shape[1])
    return frame.mean(axis=1, skipna=True).where(enough)


def build_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pos = out["position"].astype(str).str.upper()

    targets = pd.to_numeric(out.get("prior_targets_pbp"), errors="coerce")
    carries = pd.to_numeric(out.get("prior_carries_pbp"), errors="coerce")
    enough_targets = targets >= MIN_TARGETS_FOR_RATE
    enough_carries = carries >= MIN_CARRIES_FOR_RATE

    # ---- Derived rates (recomputed here so this script stands alone if the
    # upstream pbp aggregation hasn't been refreshed with them yet) ----
    if "prior_adot" not in out.columns:
        out["prior_adot"] = pd.to_numeric(out.get("prior_air_yards"), errors="coerce") / targets.replace(0, np.nan)
    rush_yds = pd.to_numeric(out.get("prior_rushing_yards"), errors="coerce")
    if "prior_yards_per_carry" not in out.columns:
        out["prior_yards_per_carry"] = rush_yds / carries.replace(0, np.nan)

    out["prior_rush_share_of_offense"] = carries / (carries.fillna(0) + targets.fillna(0)).replace(0, np.nan)

    # QB rushing production as a share of his own yardage output -- the
    # cleanest available separator of dual-threat from pocket QBs.
    pass_yds = pd.to_numeric(out.get("prior_passing_yards"), errors="coerce")
    out["prior_qb_rush_yard_share"] = rush_yds / (rush_yds.fillna(0) + pass_yds.fillna(0)).replace(0, np.nan)

    # ---- Percentiles ----
    p_carries = _pct(out, "prior_carries_pbp")
    p_targets = _pct(out, "prior_targets_pbp")
    p_snap = _pct(out, "prior_snap_share")
    p_rz_carry = _pct(out, "prior_redzone_carry_share")
    p_rz_target = _pct(out, "prior_redzone_target_share")
    p_adot = _pct(out, "prior_adot")
    p_air_share = _pct(out, "prior_air_yards_share")
    p_target_share = _pct(out, "prior_target_share")
    p_wopr = _pct(out, "prior_wopr")
    p_expl_run = _pct(out, "prior_explosive_run_rate")
    p_expl_rec = _pct(out, "prior_explosive_rec_rate")
    p_ypc = _pct(out, "prior_yards_per_carry")
    p_qb_rush = _pct(out, "prior_qb_rush_yard_share")
    p_rush_yds = _pct(out, "prior_rushing_yards")

    is_qb, is_rb, is_wr, is_te = pos.eq("QB"), pos.eq("RB"), pos.eq("WR"), pos.eq("TE")

    # ================= QB =================
    enough_passing = pass_yds >= MIN_PASSING_YARDS_FOR_QB
    qb_split = _squash(p_qb_rush, p_rush_yds)
    out["arch_dual_threat_qb"] = qb_split.where(is_qb & enough_passing)
    out["arch_pocket_passer_qb"] = (1 - qb_split).where(is_qb & enough_passing)

    # ================= RB =================
    # Bellcow: dominant snaps AND carries AND real passing-down usage.
    out["arch_bellcow_rb"] = _squash(p_snap, p_carries, p_targets).where(is_rb)
    # Receiving back: targets high relative to carries.
    out["arch_receiving_back_rb"] = _squash(p_targets, 1 - p_carries).where(is_rb)
    # Early-down grinder: carries high, targets near-absent. A fade signal.
    out["arch_early_down_rb"] = _squash(p_carries, 1 - p_targets).where(is_rb)
    # Long-run / explosive back: big-play rate on real volume.
    out["arch_long_run_rb"] = _squash(p_expl_run, p_ypc).where(is_rb & enough_carries)
    # Goal-line back: red-zone carry share exceeding overall carry share.
    out["arch_goal_line_rb"] = _squash(p_rz_carry, (p_rz_carry - p_carries).rank(pct=True)).where(is_rb)

    # ================= WR =================
    # Deep threat: high aDOT and explosive-catch rate.
    out["arch_deep_threat_wr"] = _squash(p_adot, p_expl_rec).where(is_wr & enough_targets)
    # Possession/slot: low aDOT but high target volume -- the inverse profile.
    out["arch_possession_wr"] = _squash(1 - p_adot, p_targets).where(is_wr & enough_targets)
    # Alpha: dominant share of team passing opportunity.
    out["arch_alpha_wr"] = _squash(p_target_share, p_air_share, p_wopr).where(is_wr)

    # ================= TE =================
    # Red-zone TE: TD-leveraged, red-zone looks beyond overall target share.
    out["arch_red_zone_te"] = _squash(p_rz_target, (p_rz_target - p_target_share).rank(pct=True)).where(is_te)
    # Receiving TE ("move" TE): deployed like a receiver.
    out["arch_receiving_te"] = _squash(p_targets, p_air_share, p_snap).where(is_te)
    # Blocking TE: snaps without targets. A fade signal -- snap share alone
    # would misread these as valuable, so targets-per-snap is the separator.
    out["arch_blocking_te"] = _squash(p_snap, 1 - p_targets).where(is_te)

    # ---- Hard label: highest-scoring archetype, only if it's both strong
    # and meaningfully ahead of the runner-up. Otherwise "unclassified" --
    # a genuinely ambiguous role should not be forced into a box.
    scores = out[ARCHETYPE_SCORE_COLS]
    # Rows with no prior-season data at all (rookies, players who didn't
    # play) have every archetype score NaN. idxmax raises on those, so they
    # are excluded up front and simply stay "unclassified" -- which is the
    # honest answer for a player with no usage history to classify.
    has_any = scores.notna().any(axis=1)
    top_score = scores.max(axis=1)
    second = scores.apply(
        lambda r: r.nlargest(2).iloc[-1] if r.notna().sum() >= 2 else np.nan, axis=1
    )
    top_name = pd.Series(pd.NA, index=out.index, dtype="object")
    if has_any.any():
        top_name.loc[has_any] = scores.loc[has_any].idxmax(axis=1)

    # A single non-NaN score has no runner-up to clear; treat the margin as
    # the score itself so a lone strong signal can still classify.
    margin = (top_score - second.fillna(0.0))
    decisive = has_any & (top_score >= HIGH) & (margin >= 0.05)

    out["archetype_primary"] = np.where(
        decisive & top_name.notna(),
        top_name.astype(str).str.replace("arch_", "", regex=False),
        "unclassified",
    )
    out["archetype_confidence"] = np.where(decisive, margin.round(3), 0.0)
    return out


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found -- run build_divergence_features_v1.py first.")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    out = build_archetypes(df)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Archetype dataset written: {OUTPUT_PATH}")
    print(f"Rows: {len(out)}")
    print("\nArchetype score coverage:")
    for col in ARCHETYPE_SCORE_COLS:
        print(f"  {col:28s} notna={int(out[col].notna().sum()):6d}")
    print("\nPrimary archetype distribution:")
    print(out["archetype_primary"].value_counts().to_string())


if __name__ == "__main__":
    main()
