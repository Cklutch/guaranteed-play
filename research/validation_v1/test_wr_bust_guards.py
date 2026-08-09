"""
Standalone guard tests for the WR bust validation harness.

Run:
    cd research/validation_v1
    python -m unittest discover -v

These are deliberately adversarial. Each test constructs a violation that
previously produced (or could produce) a silently wrong result, and asserts
the harness FAILS LOUDLY rather than degrading. In-script assertions are not
relied on alone: v1's invalid +0.0274 arose precisely because a silent
degradation went unnoticed.

Uses unittest (stdlib) since pytest is not installed in this environment.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from wr_bust_final_validation import (
    ADP_CONTINUOUS,
    FoldSchemaError,
    assert_continuous_adp_present,
    assert_fold_contract,
    assert_no_train_test_overlap,
    assert_shared_candidate_pool,
    eligible_pool,
    run_policy,
)

FEATS = ["overall_adp", "f1", "f2"]


def frame(n=40, season=2020, seed=0, f1_all_nan=False):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "season": season,
        "player_name": [f"P{i}_{season}" for i in range(n)],
        "overall_adp": np.linspace(1, 150, n),
        "f1": np.full(n, np.nan) if f1_all_nan else rng.normal(size=n),
        "f2": rng.normal(size=n),
        "is_bust": rng.integers(0, 2, n),
        "vor": rng.normal(scale=40, size=n),
        "final_fantasy_points": rng.normal(150, 40, n),
        "risk": rng.random(n),
    })


class TestFoldContract(unittest.TestCase):
    """Cases 1-3: availability contract."""

    def test_1_raw_feature_all_missing_in_training_fails(self):
        tr = frame(seed=1, f1_all_nan=True)   # f1 unobserved in TRAIN
        te = frame(season=2021, seed=2)
        with self.assertRaises(FoldSchemaError) as ctx:
            assert_fold_contract(tr, te, FEATS, 2021, [], [])
        self.assertIn("f1", str(ctx.exception))

    def test_2_transformed_schema_mismatch_across_folds_fails(self):
        tr = frame(seed=3)
        te = frame(season=2021, seed=4)
        ref = []
        assert_fold_contract(tr, te, FEATS, 2021, ref, [])          # establishes schema
        self.assertTrue(ref, "first fold should record a reference schema")
        with self.assertRaises(FoldSchemaError) as ctx:              # narrower feature set
            assert_fold_contract(tr, te, ["overall_adp", "f2"], 2022, ref, [])
        self.assertIn("schema", str(ctx.exception).lower())

    def test_3_nan_imputer_statistic_fails(self):
        """The exact condition under which SimpleImputer drops a column."""
        tr = frame(seed=5, f1_all_nan=True)
        stats = SimpleImputer(strategy="median").fit(tr[FEATS]).statistics_
        self.assertTrue(np.isnan(stats).any(), "precondition: imputer should produce a NaN statistic")
        with self.assertRaises(FoldSchemaError):
            assert_fold_contract(tr, frame(season=2021, seed=6), FEATS, 2021, [], [])

    def test_contract_passes_on_clean_fold(self):
        miss = []
        assert_fold_contract(frame(seed=7), frame(season=2021, seed=8), FEATS, 2021, [], miss)
        self.assertEqual(len(miss), len(FEATS), "missingness must be recorded for every feature")


class TestLeakage(unittest.TestCase):
    """Case 4."""

    def test_4_train_test_player_season_overlap_fails(self):
        tr = frame(seed=9)
        te = tr.iloc[:5].copy()  # same player-seasons on both sides
        with self.assertRaises(FoldSchemaError) as ctx:
            assert_no_train_test_overlap(tr, te)
        self.assertIn("overlap", str(ctx.exception).lower())

    def test_disjoint_seasons_pass(self):
        assert_no_train_test_overlap(frame(season=2019, seed=10), frame(season=2020, seed=11))


class TestNestedArms(unittest.TestCase):
    """Case 5."""

    def test_5_continuous_adp_absent_from_an_arm_fails(self):
        with self.assertRaises(FoldSchemaError) as ctx:
            assert_continuous_adp_present(["overall_adp", "f1"], ["f1", "f2"])  # 2nd arm lacks ADP
        self.assertIn("overall_adp", str(ctx.exception))

    def test_adp_band_is_not_a_substitute_for_continuous_adp(self):
        with self.assertRaises(FoldSchemaError):
            assert_continuous_adp_present(["adp_band", "f1"])

    def test_both_arms_with_continuous_adp_pass(self):
        assert_continuous_adp_present(ADP_CONTINUOUS, ADP_CONTINUOUS + ["f1", "f2"])


class TestReplacementEligibility(unittest.TestCase):
    """Cases 6-7."""

    def test_6_replacement_never_earlier_in_adp_than_faded_player(self):
        te = frame(seed=12).set_index("player_name")
        faded = te.index[20]
        pool = eligible_pool(te, faded)
        self.assertGreater(len(pool), 0)
        self.assertTrue((pool["overall_adp"] >= te.loc[faded, "overall_adp"]).all(),
                        "eligible pool must exclude players already off the board")
        self.assertNotIn(faded, pool.index)

    def test_6b_last_player_has_no_legal_replacement(self):
        """Exclusion is expected at the tail and must be counted, not hidden."""
        te = frame(seed=13).set_index("player_name")
        last = te["overall_adp"].idxmax()
        self.assertTrue(eligible_pool(te, last).empty)
        res = run_policy(te, [last], "model", np.random.default_rng(0))
        self.assertIsNone(res, "a fade with no legal replacement yields no swap")

    def test_7_divergent_candidate_universes_fail(self):
        te = frame(seed=14).set_index("player_name")
        faded = te.index[10]
        pool = eligible_pool(te, faded)
        assert_shared_candidate_pool(pool, pool, faded)               # identical -> ok
        with self.assertRaises(FoldSchemaError) as ctx:
            assert_shared_candidate_pool(pool, pool.iloc[2:], faded)  # control sees fewer
        self.assertIn("universes differ", str(ctx.exception))


class TestRandomControlDeterminism(unittest.TestCase):
    """Case 8."""

    def test_8_random_control_is_deterministic_under_fixed_seed(self):
        te = frame(seed=15).set_index("player_name")
        fades = list(te.index[5:12])
        a = run_policy(te, fades, "random", np.random.default_rng(17))
        b = run_policy(te, fades, "random", np.random.default_rng(17))
        self.assertIsNotNone(a)
        self.assertEqual(a, b, "same seed must reproduce the random control exactly")

    def test_8b_different_seeds_actually_differ(self):
        """Guards against a 'deterministic' result that is really a constant."""
        te = frame(seed=16).set_index("player_name")
        fades = list(te.index[5:12])
        outs = {run_policy(te, fades, "random", np.random.default_rng(s))["vor"]
                for s in range(12)}
        self.assertGreater(len(outs), 1, "random control must vary across seeds")


if __name__ == "__main__":
    unittest.main(verbosity=2)
