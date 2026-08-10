"""
Age as draft CONTEXT -- explicitly not a scoring input.

Age was tested as a predictor and failed (attempt #11). Correlation with
the post-ADP residual is -0.003 to -0.031 across all four positions, every
CI spanning zero, and an age-tilted board produced RB +0.031
CI[-0.068,+0.116] win-rate 50%, WR +0.047 CI[-0.017,+0.116] win-rate 42%.
Nothing cleared the bar every other feature was held to, so age must not
enter `final_score`.

What it IS good for is telling you where a player sits on the historical
curve while you draft -- the same role archetypes play.

The rates below come from build_age_rate_curves_v1.py and correct a real
error in the existing age study. Those charts plot the COUNT of top-N
finishers by age, which peaks in the mid-20s and reads as "elite
production happens young." The player POOL peaks at the same ages (653 RB
age-24 seasons vs 107 at 31), so the count curve is mostly a base-rate
artifact. Normalized, every position's peak moves later:

    peak by COUNT -> peak by RATE:  QB 25->33  RB 24->27  WR 26->34  TE 26->30

Read with care in both directions: the rate curve is survivorship too. A
31-year-old still in the pool is one who kept earning a role, which is
selection rather than aging. This surfaces context, not a recommendation.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import pandas as pd

_CURVE_PATH = Path("research/validation_v1/data/age_rate_curves.csv")

# Ages where WITHIN-PLAYER decline accelerates, from the delta method (see
# build_age_rate_curves_v1.py). These came from tracking the same players
# across consecutive seasons, NOT from the pool-normalized rate curve.
#
# An earlier version of this file set RB=32 off that rate curve. That was
# wrong. The rate curve is survivorship pointing the other way: only 40 RBs
# and 62 WRs in 27 seasons even reach ages 33/34, and the ones who do are
# Hall-of-Famers, so their elite rate looks high. It measures how selective
# the league is about who lasts, not how players age.
#
# Within-player year-over-year deltas, which control for that because each
# player is his own baseline:
#   RB  age 26 -20 -> 27 -38 -> 28 -42 -> 30 -45   (decline ~doubles at 27-28)
#   WR  age 24  -5 -> 26 -23 -> 28 -30 -> 31 -42   (steady, no late peak)
#   TE  age 24  -1 -> 26 -28 -> 29 -28
_DECLINE_AGE = {"QB": 30, "RB": 28, "WR": 29, "TE": 29}


@lru_cache(maxsize=1)
def load_age_rates() -> dict:
    """{(position, age): historical elite rate}. Empty dict if unavailable."""
    if not _CURVE_PATH.exists():
        return {}
    try:
        curves = pd.read_csv(_CURVE_PATH)
    except Exception:
        return {}
    return {
        (str(r["position"]).upper(), int(r["age"])): float(r["elite_rate"])
        for _, r in curves.iterrows()
    }


def age_note(position, age) -> str:
    """
    One-line age context for a player, or "" when there is nothing to say.

    Deliberately quiet: most players sit in the flat middle of their
    position's curve, and a note on every row would be noise.
    """
    rates = load_age_rates()
    if not rates or age is None or pd.isna(age):
        return ""
    try:
        age_i = int(round(float(age)))
    except (TypeError, ValueError):
        return ""
    pos = str(position).upper()
    rate = rates.get((pos, age_i))
    if rate is None:
        return ""

    decline_at = _DECLINE_AGE.get(pos, 99)
    if age_i >= decline_at:
        return f"Age {age_i}: past where {pos} year-over-year decline accelerates"
    return ""
