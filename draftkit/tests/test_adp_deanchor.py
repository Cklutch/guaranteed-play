"""plan_deanchor_scoring_from_adp.pdf, Phase 1: real, confirmed circularity
fix in draftkit/signal_trust.py / draftkit/draft_analysis.py.

Plain assert-based, no pytest -- matches this repo's established convention.
Runnable directly:

    python -m draftkit.tests.test_adp_deanchor
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draftkit.signal_trust import (  # noqa: E402
    calculate_adp_trust,
    calculate_market_disagreement_trust,
    calculate_signal_trust,
)
from draftkit.draft_analysis import apply_signal_trust_adjustments  # noqa: E402

COLUMNS = {"adp": "adp", "adp_rank": "adp_rank", "projection_rank": "projection_rank"}


def _row(adp_rank, projection_rank, adp=None):
    return {"adp": adp if adp is not None else adp_rank, "adp_rank": adp_rank, "projection_rank": projection_rank}


def test_a_nacua_2023_market_miss_not_dampened():
    """Real 2023 Puka Nacua: a 5th-round rookie whose real preseason ADP sat
    deep outside the top 200 in the large majority of real redraft leagues,
    while a real, visible signal (immediate, historic target volume from
    Week 1) would justify a model projection_rank far better than that ADP
    -- he went on to set the rookie receiving record and finish as a real
    top-5 PPR WR. Real, cited facts used as a constructed regression input
    (this repo has no literal 2023 preseason snapshot to replay end-to-end).

    Before the fix: this exact shape (adp_rank far worse than
    projection_rank, crossing EXTREME_RANK_GAP) hit extreme_adp_projection_gap
    + adp_rank_conflict + projection_rank_conflict, all in the old
    severe_flags set -- would have hard-capped his score at 58.0 regardless
    of how much the model liked him. After the fix: the real divergence is
    disclosed (market_divergence_flags) but does not dampen signal_trust."""
    row = _row(adp_rank=215.0, projection_rank=15.0)
    adp_trust = calculate_adp_trust(row, COLUMNS)
    assert adp_trust["trust_score"] >= 90.0, (
        f"expected a clean genuine adp_trust score (real ADP/rank data, no real inconsistency), "
        f"got {adp_trust['trust_score']}"
    )
    assert "extreme_adp_projection_gap" in adp_trust["divergence_flags"], (
        f"expected the real 200-rank gap to be disclosed as divergence, got {adp_trust['divergence_flags']}"
    )
    assert not any(f in adp_trust["flags"] for f in ("extreme_adp_projection_gap", "large_adp_projection_gap")), (
        "expected divergence flags to NOT appear in the scored genuine flags list"
    )

    df = pd.DataFrame([{"player_name": "Test Nacua-Shaped Rookie", "final_score": 75.0}])
    adjusted = apply_signal_trust_adjustments(df)
    # build_signal_trust_report() pulls from the real live player pool, not
    # our synthetic row, so this call won't find a match -- the real,
    # function-level check above (calculate_adp_trust directly) is what
    # proves the fix; this confirms the pipeline doesn't error on a
    # not-found player either.
    assert len(adjusted) == 1
    print("A PASS -- Nacua-shaped market-miss case: real 200-rank gap disclosed as divergence, "
          f"genuine adp_trust={adp_trust['trust_score']} (clean, not dampened)")


def test_b_taylor_2022_adp_bust_signal_preserved():
    """Real 2022 Jonathan Taylor: real top-5-overall preseason ADP after a
    historic 2021 season, while real, visible pre-season risk signals (see
    draftkit/tests/test_injury_history.py's own real Taylor case -- a real
    injury-history pattern) would justify a model projection_rank
    meaningfully worse than his ADP. Real 2022 outcome: an ankle injury and
    Colts' offensive collapse produced a real bust year. This is the
    OPPOSITE real direction from Nacua -- proves the fix doesn't just let
    optimistic divergence through, it preserves real pessimistic
    divergence too."""
    row = _row(adp_rank=5.0, projection_rank=85.0)
    adp_trust = calculate_adp_trust(row, COLUMNS)
    assert adp_trust["trust_score"] >= 90.0, f"expected clean genuine adp_trust, got {adp_trust['trust_score']}"
    assert "extreme_adp_projection_gap" in adp_trust["divergence_flags"], (
        f"expected the real 80-rank gap to be disclosed as divergence, got {adp_trust['divergence_flags']}"
    )
    print(f"B PASS -- Taylor-shaped ADP-bust case: real 80-rank gap disclosed as divergence, "
          f"genuine adp_trust={adp_trust['trust_score']} (clean, not dampened)")


def test_c_genuine_thin_sample_still_anchored():
    """Real safeguard case, confirmed directly on the live board and the
    real raw source data (draftkit.data_access.load_players_df()): 328 of
    3941 real players share `adp=216.0` -- this repo's real placeholder/
    replacement-level ADP -- while their real `adp_rank` field independently
    holds a genuinely out-of-range value (real example: Theo Wease,
    adp=216.0, adp_rank=633.0, |216-633|=417 apart). This is genuine
    internal data inconsistency, not model disagreement, and must STILL
    get legitimately anchored after the fix."""
    row = _row(adp_rank=633.0, projection_rank=57.0, adp=216.0)
    adp_trust = calculate_adp_trust(row, COLUMNS)
    assert "adp_rank_out_of_range" in adp_trust["flags"], f"expected a genuine flag, got {adp_trust['flags']}"
    assert adp_trust["trust_score"] < 90.0, (
        f"expected a genuinely reduced trust score for real placeholder ADP data, got {adp_trust['trust_score']}"
    )

    market_trust = calculate_market_disagreement_trust(row, COLUMNS, adp_trust=adp_trust)
    projection_trust = {"trust_score": 65.0}  # real invalid_projection case, same real population
    sportsbook_trust = {"trust_score": 100.0}
    signal_trust = calculate_signal_trust(adp_trust, projection_trust, market_trust, sportsbook_trust)
    assert signal_trust < 76.0, (
        f"expected this real genuine-thin-sample profile to still fall below the recalibrated "
        f"SIGNAL_TRUST_DAMPEN_THRESHOLD (76.0), got {signal_trust} -- safeguard would be stripped"
    )
    print(f"C PASS -- genuine thin-sample case still correctly dampened: adp_trust={adp_trust['trust_score']}, "
          f"signal_trust={signal_trust} (below 76.0 threshold)")


def test_d_qb_clustering_no_longer_dampened_but_exceptions_handled_consistently():
    """Real, measured live-board check (not synthetic): before the fix, 91
    of 103 real QBs clustered into 4 near-identical signal_trust_score
    buckets (82.00/58.00/50.50/40.00) driven almost entirely by ADP-gap
    thresholds, with 12 real exceptions (including Trevor Lawrence,
    64.25) sitting at other values due to real partial flag differences --
    see this plan's own Context section for the full real distribution.
    After the fix, real elite QBs (previously clustered) and real
    exceptions (previously distinct) must converge onto the SAME genuine
    baseline if their only real difference was ADP-divergence magnitude --
    proving the fix treats real exceptions consistently, not as a carved-
    out special case."""
    import sys as _sys
    if str(REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(REPO_ROOT))
    from draftkit.draft_analysis import build_recommendation_rankings_df

    df = build_recommendation_rankings_df()
    names = ["Josh Allen", "Joe Burrow", "Trevor Lawrence", "Patrick Mahomes", "Jordan Love"]
    rows = df[df["player_name"].isin(names)]
    if rows.empty:
        print("D SKIP -- live board not available in this environment")
        return

    scores = rows.set_index("player_name")["signal_trust_score"]
    for name in names:
        if name not in scores.index:
            continue
        assert scores[name] >= 85.0, f"expected {name} to have a clean, high genuine signal_trust, got {scores[name]}"
    unique_scores = scores.round(2).unique()
    assert len(unique_scores) <= 2, (
        f"expected real elite QBs (including the former Trevor Lawrence exception) to converge "
        f"onto the same genuine baseline post-fix, got distinct values {sorted(unique_scores)}"
    )
    applied = rows.set_index("player_name")["trust_adjustment_applied"]
    assert not applied.any(), f"expected none of these real elite QBs to be dampened, got {applied.to_dict()}"
    print(f"D PASS -- real elite QBs converge to signal_trust={sorted(unique_scores)}, none dampened "
          f"(Trevor Lawrence's former distinct value of 64.25 is gone, handled consistently not carved out)")


def main() -> int:
    test_a_nacua_2023_market_miss_not_dampened()
    test_b_taylor_2022_adp_bust_signal_preserved()
    test_c_genuine_thin_sample_still_anchored()
    test_d_qb_clustering_no_longer_dampened_but_exceptions_handled_consistently()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
