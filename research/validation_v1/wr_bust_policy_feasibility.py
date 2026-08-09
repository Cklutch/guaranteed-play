"""
Policy-phase FEASIBILITY AUDIT. Structure only — no policy outcomes.

Purpose: determine whether the preregistered matched design (memo section 10)
can answer the policy question at all, BEFORE running it. This script
deliberately computes no VOR, no bust rate, and no policy contrast. It reports
only eligibility and matching structure, so that inspecting it cannot
influence the confirmatory test.

Interpretive rule, fixed in advance:

    If the matched sample is too sparse or too concentrated to support a
    stable season-blocked estimate, the policy phase will be reported as
    INFEASIBLE / UNDERPOWERED rather than as evidence for or against a
    draft-policy edge.

No hard sample-size cutoff is invented here. The observed structure is
presented transparently; a broad interval from a valid but sparse design
means "inconclusive", never "no policy edge".

If exact replacement-opportunity-count matching proves impractically sparse,
that failure is reported FIRST and is not silently relaxed. Coarsened
alternatives (predeclared bins 1 / 2-3 / 4-6 / 7+) are labelled a SECONDARY
SENSITIVITY analysis, never the confirmatory test.

Run:  python research/validation_v1/wr_bust_policy_feasibility.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from wr_bust_final_validation import (
    ADP_CONTINUOUS, FADE_THRESHOLD, OUTDIR, SEED, SPECS,
    eligible_pool, folds, load, make_pipe,
)

# Preregistered bins (memo 10.1 / 10.3). Declared before inspection.
ADP_SHIFT_BINS = [("0-3", 0, 3), ("4-7", 4, 7), ("8-12", 8, 12),
                  ("13-18", 13, 18), ("19+", 19, 10**6)]
OPPORTUNITY_BINS = [("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7+", 7, 10**6)]
ADP_MATCH_TOLERANCE = 6  # "tightly binned ADP range", predeclared
SEP = "=" * 74


def provenance() -> dict:
    def _git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None

    def _hash(path):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except Exception:
            return None

    from wr_bust_final_validation import DATASET
    pkgs = {}
    for m in ("numpy", "pandas", "sklearn", "scipy"):
        try:
            pkgs[m] = __import__(m).__version__
        except Exception:
            pkgs[m] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python research/validation_v1/wr_bust_policy_feasibility.py",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "seeds": {"SEED": SEED},
        "input_data": {str(DATASET.name): _hash(DATASET)},
        "fade_threshold": FADE_THRESHOLD,
        "adp_match_tolerance": ADP_MATCH_TOLERANCE,
        "note": "Feasibility audit only. No policy outcomes computed.",
    }


def _bin(value, bins):
    for label, lo, hi in bins:
        if lo <= value <= hi:
            return label
    return None


def audit_spec(d, feats, start, label):
    expanded = ADP_CONTINUOUS + feats
    season_rows, fade_rows = [], []

    for ts, tr, te in folds(d, feats, start):
        est = make_pipe().fit(tr[expanded], tr["is_bust"])
        te = te.copy()
        te["risk"] = est.predict_proba(te[expanded])[:, 1]
        # Opportunity count is purely structural: how many WRs remain on the
        # board at or after this player's ADP.
        te["opp_count"] = [len(eligible_pool(te, i)) for i in te.index]

        k = max(1, int(round(len(te) * FADE_THRESHOLD)))
        model_fades = te.nlargest(k, "risk").index.tolist()

        after_adp, after_opp = 0, 0
        for fi in model_fades:
            row = te.loc[fi]
            # (a) control fades matched on tight ADP range, same season
            adp_pool = te[(te.overall_adp.between(row.overall_adp - ADP_MATCH_TOLERANCE,
                                                  row.overall_adp + ADP_MATCH_TOLERANCE))
                          & (te.index != fi)]
            has_adp_match = len(adp_pool) > 0
            # (b) additionally matched on EXACT replacement-opportunity count
            opp_pool = adp_pool[adp_pool.opp_count == row.opp_count]
            has_opp_match = len(opp_pool) > 0
            after_adp += int(has_adp_match)
            after_opp += int(has_opp_match)

            pool = eligible_pool(te, fi)
            shifts = (pool.overall_adp - row.overall_adp) if len(pool) else pd.Series(dtype=float)
            bins_present = {lab: int(((shifts >= lo) & (shifts <= hi)).sum())
                            for lab, lo, hi in ADP_SHIFT_BINS}
            fade_rows.append({
                "spec": label, "season": ts, "player": row.get("player_name"),
                "adp": round(float(row.overall_adp), 1), "opp_count": int(row.opp_count),
                "adp_match_candidates": len(adp_pool),
                "exact_opp_match_candidates": len(opp_pool),
                "coarse_opp_bin": _bin(int(row.opp_count), OPPORTUNITY_BINS),
                **{f"bin_{lab}": n for lab, n in bins_present.items()},
            })

        season_rows.append({
            "spec": label, "season": ts, "eligible_wrs": len(te),
            "fade_candidates": len(model_fades),
            "after_adp_match": after_adp,
            "after_exact_opp_match": after_opp,
        })
    return pd.DataFrame(season_rows), pd.DataFrame(fade_rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    d = load()

    print(SEP)
    print("POLICY FEASIBILITY AUDIT -- structure only, no outcomes computed")
    print(SEP)
    print(f"git {meta['git_commit']} (dirty={meta['git_dirty']}) | python {meta['python']} | "
          f"seed {SEED}\ninput sha256[:16] {meta['input_data']}")

    all_seasons, all_fades = [], []
    for label, (feats, start) in SPECS.items():
        feats = [f for f in feats if f in d.columns]
        s, f = audit_spec(d, feats, start, label)
        if s.empty:
            continue
        all_seasons.append(s)
        all_fades.append(f)

        print(f"\n{SEP}\nSPEC {label}\n{SEP}")
        print(s.to_string(index=False))
        tot_f, tot_a, tot_o = s.fade_candidates.sum(), s.after_adp_match.sum(), s.after_exact_opp_match.sum()
        print(f"\n  fade candidates {tot_f} -> after ADP match {tot_a} ({tot_a/tot_f:.0%}) "
              f"-> after EXACT opportunity match {tot_o} ({tot_o/tot_f:.0%})")
        print(f"  effective seasons retained: {(s.after_exact_opp_match > 0).sum()} of {len(s)}")
        print(f"  matched swaps per season -- total {tot_o}, median "
              f"{s.after_exact_opp_match.median():.1f}, min {s.after_exact_opp_match.min()}, "
              f"max {s.after_exact_opp_match.max()}")

        # concentration by season
        if tot_o > 0:
            share = (s.set_index("season").after_exact_opp_match / tot_o).sort_values(ascending=False)
            print(f"  season concentration: top season holds {share.iloc[0]:.0%}, "
                  f"top 3 hold {share.head(3).sum():.0%}")

        # ADP-shift bin availability
        bin_cols = [f"bin_{lab}" for lab, _, _ in ADP_SHIFT_BINS]
        avail = (f[bin_cols] > 0).sum()
        print("\n  fades with >=1 legal replacement per ADP-shift bin "
              f"(of {len(f)} fade candidates):")
        for lab, _, _ in ADP_SHIFT_BINS:
            n = int(avail[f"bin_{lab}"])
            print(f"    {lab:6s} {n:4d}  ({n/len(f):.0%})")
        tot_by_bin = f[bin_cols].sum()
        gt = tot_by_bin.sum()
        if gt:
            print("  share of ALL legal replacement slots by bin: " +
                  ", ".join(f"{lab}={tot_by_bin[f'bin_{lab}']/gt:.0%}" for lab, _, _ in ADP_SHIFT_BINS))

        # strata sparsity -- the make-or-break for exact matching
        empty = int((f.exact_opp_match_candidates == 0).sum())
        thin = int((f.exact_opp_match_candidates.between(1, 2)).sum())
        print(f"\n  EXACT opportunity-count strata: {empty} of {len(f)} fades have ZERO "
              f"matched controls ({empty/len(f):.0%}); {thin} have only 1-2")
        print("  coarse opportunity bins (SECONDARY SENSITIVITY ONLY, not confirmatory):")
        print("   ", f.coarse_opp_bin.value_counts().to_dict())

    if all_seasons:
        pd.concat(all_seasons).to_csv(OUTDIR / "feasibility_by_season.csv", index=False)
        pd.concat(all_fades).to_csv(OUTDIR / "feasibility_by_fade.csv", index=False)
    (OUTDIR / "feasibility_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{SEP}")
    print("INTERPRETIVE RULE: if the matched sample is too sparse or too concentrated")
    print("to support a stable season-blocked estimate, the policy phase is reported")
    print("as INFEASIBLE / UNDERPOWERED -- not as evidence for or against a policy edge.")
    print(SEP)
    print(f"Artifacts + provenance written to {OUTDIR}")


if __name__ == "__main__":
    main()
