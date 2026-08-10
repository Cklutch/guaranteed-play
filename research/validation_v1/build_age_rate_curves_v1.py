"""
Age curves, normalized by pool size.

The existing age study (case_studies/run_fantasy_age_study.py and the
position_comparison_top*.svg charts) plots the COUNT of top-N finishers by
age. Those counts peak at 24-27, which reads as "elite production happens
young." The counts are computed correctly -- the inference is not.

The pool peaks at the same ages. There are 653 RB age-24 seasons in
1999-2025 and 107 at age 31. More 24-year-olds finish top-24 largely
because there are six times as many of them. Dividing by pool size:

    RB top-24 rate:  age 23 0.13 | 24 0.14 | 26 0.18 | 27 0.21 | 31 0.18
    WR top-24 rate:  age 23 0.07 | 24 0.07 | 26 0.12 | 29 0.17 | 31 0.18

RB elite rate PEAKS at 27 and holds through 31. WR elite rate more than
DOUBLES from 23 to 30. The count curve's apparent young-age peak is a base
-rate artifact and points the opposite direction from the rate curve.

Two cautions this script exists to state, not to bury:

1. The rate curve is ALSO survivorship, in the other direction. A 31-year-
   old still in the pool kept earning a role; that is selection, not aging.
   Neither curve licenses "draft old WRs."
2. Neither curve is a tradeable edge. Tested directly against ADP
   (walk-forward, top-decile lift), an age-tilted board produced RB +0.031
   CI[-0.068,+0.116] win 50%, WR +0.047 CI[-0.017,+0.116] win 42% -- every
   CI includes zero and no win rate clears 50%. Age is draft CONTEXT here,
   never a scoring input.

Output: research/validation_v1/data/age_rate_curves.csv
"""
from __future__ import annotations

import pandas as pd

from validation_utils import VALIDATION_DIR

PROJECT_ROOT = VALIDATION_DIR.parents[1]
SOURCE = PROJECT_ROOT / "case_studies" / "data" / "qb_rb_te_wr_elite_age_player_seasons_ppr.csv"
OUTPUT = VALIDATION_DIR / "data" / "age_rate_curves.csv"

TOP_N = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}
AGE_MIN, AGE_MAX = 22, 34
# Below this many player-seasons the rate is too noisy to plot honestly.
MIN_POOL = 25


def build_curves() -> pd.DataFrame:
    d = pd.read_csv(SOURCE, low_memory=False)
    d = d[d["age"].notna() & d["positional_finish"].notna()].copy()
    d["age"] = d["age"].astype(int)

    rows = []
    for position, top_n in TOP_N.items():
        g = d[d["position"] == position]
        agg = g.groupby("age").agg(
            pool=("player_name", "size"),
            elite_count=("positional_finish", lambda s: int((s <= top_n).sum())),
        )
        agg = agg[(agg.index >= AGE_MIN) & (agg.index <= AGE_MAX) & (agg["pool"] >= MIN_POOL)]
        agg["elite_rate"] = agg["elite_count"] / agg["pool"]
        for age, r in agg.iterrows():
            rows.append({
                "position": position, "age": int(age), "top_n": top_n,
                "pool": int(r["pool"]), "elite_count": int(r["elite_count"]),
                "elite_rate": round(float(r["elite_rate"]), 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    curves = build_curves()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(OUTPUT, index=False)

    for position in TOP_N:
        c = curves[curves["position"] == position]
        if c.empty:
            continue
        print(f"\n--- {position} (top-{TOP_N[position]}) ---")
        print("  age  :", " ".join(f"{a:>5d}" for a in c["age"]))
        print("  pool :", " ".join(f"{v:>5d}" for v in c["pool"]))
        print("  COUNT:", " ".join(f"{v:>5d}" for v in c["elite_count"]))
        print("  RATE :", " ".join(f"{v:>5.2f}" for v in c["elite_rate"]))
        peak_count = int(c.loc[c["elite_count"].idxmax(), "age"])
        peak_rate = int(c.loc[c["elite_rate"].idxmax(), "age"])
        print(f"  peak by COUNT: age {peak_count}   peak by RATE: age {peak_rate}")

    print(f"\nWritten: {OUTPUT}")
    print("\nCOUNT peaks track the pool, not aging. Use RATE. And note the rate")
    print("curve is survivorship in the other direction -- neither is a scoring edge.")


if __name__ == "__main__":
    main()
