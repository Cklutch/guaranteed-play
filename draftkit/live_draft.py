"""Live-draft pick recommendation engine (v1).

A transparent, tunable "who should I draft next" score, built to grow in
clear stages. This first version scores every still-available player on
four factors the user named:

    draft_score = W_value        * our_score          (the board's model score, 0-100)
                - W_injury       * injury_risk         (0-100)
                - W_overall_risk * overall_risk        (0-100, risk_index)
                - W_bye_conflict * bye_conflict_count  (roster bye stacking)

Everything is on a 0-100 value scale so the weights are directly readable
as "max points this factor can move a player." The function returns the
per-factor contributions alongside the total, so any ranking is explainable
off a single row.

DELIBERATELY NOT HERE YET (next stage, per the plan): positional need and
tier disparity. The scaffolding (roster counts, position_tier on the pool)
is already threaded through so those slot in without reshaping this.

Notes:
  * our_score already contains a modest risk penalty (the board's Base
    Value subtracts ~risk/100*15). The injury/overall-risk terms here are
    ADDITIONAL, draft-time emphasis on top of that -- keep their weights
    modest. injury_risk is also a component of overall risk_index, so the
    two risk terms overlap by design; tune with that in mind.
  * Decoupled from Streamlit: pure functions over DataFrames + plain draft
    state (lists of names), so it is unit-testable and reusable.
"""

import numpy as np
import pandas as pd

from draftkit.draft_analysis import build_recommendation_rankings_df

RISK_VARIABLES_PATH = "data/processed/risk_variables.csv"
MASTER_PLAYERS_PATH = "data/processed/master_players.csv"

# Default factor weights. value dominates; the risk/bye terms are secondary
# nudges. All tunable -- this is the main dial for the engine's behavior.
DEFAULT_WEIGHTS = {
    "value": 1.0,          # x our_score (0-100)
    "injury": 0.15,        # x injury_risk (0-100)  -> up to -15
    "overall_risk": 0.10,  # x overall_risk (0-100) -> up to -10
    "bye_conflict": 4.0,   # flat points per rostered same-position same-bye player
}

# Neutral fill for players missing a risk score, so unknowns are neither
# rewarded nor punished versus a median-risk player.
_NEUTRAL_RISK = 50.0

# A currently-hurt designation floors injury risk high no matter the career
# durability. "Questionable" is deliberately NOT here (benign game-time tag).
_SERIOUS_INJURY_STATUSES = {"IR", "INJURED RESERVE", "PUP", "OUT", "DOUBTFUL", "INACTIVE", "DNR"}
_SERIOUS_INJURY_FLOOR = 85.0

# --- roster construction / positional need ------------------------------
# Starting-lineup targets for a standard 1QB/2RB/3WR/1TE build (FLEX slack
# folded into the WR target). While a position is under target it's a NEED
# (bonus); once full, extra bodies are depth (penalty).
STARTER_TARGETS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
# Positions you roster only one of: once your starter is drafted, stop
# recommending more so a premium TE/QB never keeps topping the board after
# you've taken your one (1-TE / 1-QB leagues).
ROSTER_CAPS = {"QB": 1, "TE": 1}
# Positional-need adjustment. A FLAT bonus for any position whose starters
# aren't full, and a penalty once they are. Deliberately flat (not scaled
# by slot count) so that with an empty roster every position gets the same
# nudge -- it cancels out and the board is pure best-player-available. The
# lean only emerges as you actually fill positions and imbalances appear.
NEED_BONUS = 6.0           # any position with an open starter slot
SATURATION_PENALTY = 6.0   # per extra body once the starter slots are full

# Positional scarcity bonus: NEED_BONUS is deliberately flat, so "barely the
# best RB available" and "best RB by 20 points, with a real cliff to the
# next tier" score identically. This adds a second, targeted bonus ONLY for
# whoever is the actual #1 available player at a position you still need,
# sized by how far ahead they are of the #2 option at that position -- the
# bigger the gap, the more urgent it is to take him now before the cliff.
# Not scaled by need depth (RB) vs weight of need itself -- just the margin.
SCARCITY_BONUS_PER_POINT = 0.3   # x (our_score gap to the #2 player at the position)
SCARCITY_BONUS_MAX = 10.0        # cap so a huge gap can't swamp value/risk

# Charlie's (standard-scoring) QB discipline: don't recommend a mid QB
# early -- only an ELITE QB AT VALUE before the hold round. Dad's league
# needs a good QB, so it's exempt (scoring_mode="dads").
QB_HOLD_UNTIL_ROUND = 10   # before this round, suppress non-elite QBs (standard only)
ELITE_QB_RANK = 3          # a top-N positional QB counts as "elite"

# Charlie's TE discipline: rounds 6-9 are a bad window to spend a pick on a
# TE unless he's fallen well past his ADP (a genuine bargain) -- otherwise
# hold. Standard scoring only; Dad's league is exempt (same reasoning as QB).
TE_HOLD_ROUND_START, TE_HOLD_ROUND_END = 6, 9
TE_HOLD_VALUE_MARGIN = 15  # picks past ADP required to count as "a great deal"

# Recommendation-slate diversity: never let one position monopolize the
# cards -- fixes "all four suggestions are WRs".
MAX_RECS_PER_POSITION = 2

# Risk is managed at the ROSTER level, not as a flat value-killer: a risky
# pick is fine on an otherwise-safe team. The per-player risk penalty is
# scaled by how much risk you've already rostered -- light when your team
# is safe (spend the risk budget), heavier once you're stacked with risk.
RISK_MULT_MIN, RISK_MULT_MAX = 0.5, 1.8
RISK_MULT_SCALE = 60.0     # roster avg-risk points per +1.0 of multiplier

# Opportunity cost: a player very likely to still be there at your NEXT pick
# is worth less NOW (grab the one who won't return). Up to this many points
# are shaved by high return-probability; ~0 for a player about to be gone.
# Weight of 18 (not a token nudge) is deliberate: a near-certain return
# (~99%) should be able to outweigh the flat need bonus (6) AND meaningfully
# cut into a real value gap -- a soft nudge here was getting lost against
# our_score differences of 15-40 points and never actually changed a
# recommendation (verified: an 8-weight penalty on a 99.8%-survival player
# netted to -2 against the need bonus alone).
SURVIVAL_WEIGHT = 18.0
SURVIVAL_K = 0.30          # logistic steepness, matches the survival gauge


def build_candidate_pool(score_col="final_score", pool_size=300):
    """Assemble the draftable candidate pool with everything the engine
    needs, one row per player.

    `score_col` picks which model score is the value base -- "final_score"
    (standard board) or "dads_final_score" (Dad's League), so the draft
    engine can follow whichever scoring the user drafts under. our_score is
    that column min-max scaled to 0-100 across the pool, matching the
    board's OUR SCORE display.

    Columns returned: player_name, position, team, bye_week, our_score,
    injury_risk, overall_risk, position_tier, adp, plus the raw score_col.
    """
    board = build_recommendation_rankings_df()
    # Dad's League value base isn't on the raw engine board -- add it on
    # demand so the draft engine can follow either scoring model.
    if not board.empty and score_col not in board.columns:
        try:
            from draftkit.dads_scoring import add_dads_scores
            board = add_dads_scores(board)
        except Exception:
            pass
    if board.empty or score_col not in board.columns:
        return pd.DataFrame()

    # Draftable only: someone the market prices (ADP) or a service projects.
    has_adp = pd.to_numeric(board.get("adp"), errors="coerce").notna()
    real_proj = board.get("projection_source", pd.Series(index=board.index, dtype=object)) == "real"
    board = board[has_adp | real_proj].copy()

    board = board.sort_values(score_col, ascending=False, na_position="last").head(pool_size)

    # Bring in overall risk (risk_index), bye week, and the historical
    # durability score from the risk file.
    try:
        risk = pd.read_csv(RISK_VARIABLES_PATH)
        keep = [c for c in ("player_name", "risk_index", "bye_week", "injury_score", "injury_status")
                if c in risk.columns]
        board = board.merge(risk[keep].drop_duplicates("player_name"), on="player_name", how="left")
    except (FileNotFoundError, OSError):
        pass

    # Sleeper player_id (from master_players.csv) so a live Sleeper draft can
    # be matched to the pool exactly by id -- see draftkit/sleeper.py.
    try:
        mp = pd.read_csv(MASTER_PLAYERS_PATH)
        if {"player_name", "player_id"}.issubset(mp.columns):
            board = board.merge(
                mp[["player_name", "player_id"]].drop_duplicates("player_name"),
                on="player_name", how="left",
            )
    except (FileNotFoundError, OSError):
        pass

    score = pd.to_numeric(board[score_col], errors="coerce")
    lo, hi = score.min(), score.max()
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        board["our_score"] = ((score - lo) / (hi - lo) * 100).round(1)
    else:
        board["our_score"] = 50.0

    # Injury measure for drafting = historical DURABILITY, from injury_score
    # (a 1-5 games-missed scale) rescaled to 0-100. Deliberately NOT the
    # risk_variables `injury_risk` field: that one is dominated by the
    # current weekly injury_status ("Questionable" etc.) and mislabels
    # durable players -- e.g. Gibbs (2 career games missed) reads 65 there
    # but injury_score 1 -> 0 here, while a genuinely fragile back like
    # McCaffrey reads 5 -> 100. Falls back to any existing injury_risk when
    # injury_score is unavailable.
    if "injury_score" in board.columns:
        _inj = pd.to_numeric(board["injury_score"], errors="coerce")
        board["injury_risk"] = ((_inj - 1) / 4 * 100).clip(0, 100)
    else:
        board["injury_risk"] = pd.to_numeric(board.get("injury_risk"), errors="coerce")

    # Current-injury floor: a player with a serious active designation
    # (IR / PUP / Out / Doubtful / Inactive) is hurt RIGHT NOW regardless of
    # career durability, so floor their injury high even if injury_score
    # missed it. "Questionable" is intentionally excluded -- it's usually a
    # benign game-time tag (e.g. Gibbs), which is what motivated moving off
    # the status-driven number in the first place.
    if "injury_status" in board.columns:
        status = board["injury_status"].astype(str).str.upper().str.strip()
        serious = status.isin(_SERIOUS_INJURY_STATUSES)
        board.loc[serious, "injury_risk"] = board.loc[serious, "injury_risk"].clip(lower=_SERIOUS_INJURY_FLOOR)
    board["overall_risk"] = pd.to_numeric(board.get("risk_index"), errors="coerce")
    if "bye_week" not in board.columns:
        board["bye_week"] = pd.NA

    # De-duplicate name variants (e.g. "Kenneth Gainwell" vs "Kenny Gainwell",
    # "Hollywood" vs "Marquise Brown"): when the same last-name/team/position
    # appears both with and without a Sleeper player_id, drop the id-less
    # phantom. It can never be matched to a Sleeper pick, so it would linger
    # as a recommendation even after the real player is drafted.
    if "player_id" in board.columns and "team" in board.columns:
        last_key = (
            board["player_name"].astype(str).str.split().str[-1].str.lower()
            + "|" + board["team"].astype(str).str.upper()
            + "|" + board["position"].astype(str).str.upper()
        )
        has_id = board["player_id"].notna()
        keyed_with_id = set(last_key[has_id])
        board = board[~((~has_id) & last_key.isin(keyed_with_id))].copy()

    cols = [
        "player_name", "player_id", "position", "team", "bye_week", "our_score",
        "injury_risk", "injury_status", "overall_risk", "position_tier", "adp", score_col,
    ]
    return board[[c for c in cols if c in board.columns]].reset_index(drop=True)


def _roster_bye_counts(pool, my_team):
    """Map (position, bye_week) -> how many of my rostered players sit there,
    for the bye-conflict penalty."""
    if not my_team:
        return {}
    mine = pool[pool["player_name"].isin(my_team)]
    counts = {}
    for _, r in mine.iterrows():
        bye = r.get("bye_week")
        if pd.isna(bye):
            continue
        key = (str(r.get("position")).upper(), int(bye))
        counts[key] = counts.get(key, 0) + 1
    return counts


def diverse_slate(ordered, n, max_per_pos=MAX_RECS_PER_POSITION):
    """Take the top-n by draft_score, but never more than `max_per_pos` of
    one position -- so the cards can't all be the same position. It does NOT
    reach down the board to force a position in: at an empty roster that's
    just best-player-available; the need lean already lives in draft_score."""
    rows = [r for _, r in ordered.iterrows()]
    if not rows:
        return ordered
    chosen, names, pos_count = [], set(), {}

    def take(r):
        chosen.append(r)
        names.add(r["player_name"])
        p = str(r["position"]).upper()
        pos_count[p] = pos_count.get(p, 0) + 1

    for r in rows:                               # by score, capped per position
        if len(chosen) >= n:
            break
        if r["player_name"] in names or pos_count.get(str(r["position"]).upper(), 0) >= max_per_pos:
            continue
        take(r)
    for r in rows:                               # relax the cap only if still short
        if len(chosen) >= n:
            break
        if r["player_name"] not in names:
            take(r)
    return pd.DataFrame(chosen)


def recommend_picks(pool, drafted_players=None, my_team=None, weights=None, top_n=15,
                    current_round=None, current_pick=None, my_next_pick=None,
                    scoring_mode="standard", diverse=False):
    """Rank the still-available players by draft_score.

    drafted_players / my_team are lists of player names (drafted by anyone /
    by me); both are removed from the pool, and my_team drives roster need,
    the TE/QB cap, and the bye penalty. `current_round`/`current_pick` +
    `scoring_mode` enforce Charlie's QB and TE discipline (standard only). With
    `diverse=True` the top_n is a position-diverse slate. Returns a frame
    sorted best-first with each factor's signed contribution.
    """
    if pool is None or pool.empty:
        return pd.DataFrame()

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    drafted_players = drafted_players or []
    my_team = my_team or []

    taken = set(drafted_players) | set(my_team)
    avail = pool[~pool["player_name"].isin(taken)].copy()
    if avail.empty:
        return avail

    # Roster construction: what I already have, unfilled starter needs, and
    # positions capped at one (TE/QB).
    mine = pool[pool["player_name"].isin(my_team)]
    have = mine["position"].astype(str).str.upper().value_counts().to_dict()

    _pos = avail["position"].astype(str).str.upper()
    over_cap = _pos.map(lambda p: p in ROSTER_CAPS and have.get(p, 0) >= ROSTER_CAPS[p])
    avail = avail[~over_cap.values].copy()

    # Charlie's QB discipline (standard scoring only): before the hold round,
    # drop every QB except an elite one available at value (fallen to/past
    # his ADP). Dad's league is exempt.
    if scoring_mode != "dads" and current_round is not None and current_round < QB_HOLD_UNTIL_ROUND:
        elite_qbs = set(
            pool[pool["position"].astype(str).str.upper() == "QB"]
            .sort_values("our_score", ascending=False)["player_name"].head(ELITE_QB_RANK)
        )
        is_qb = avail["position"].astype(str).str.upper() == "QB"
        adp = pd.to_numeric(avail.get("adp"), errors="coerce")
        at_value = adp.notna() & (adp <= (current_pick if current_pick else 0))
        keep_qb = avail["player_name"].isin(elite_qbs) & (at_value if current_pick else True)
        avail = avail[(~is_qb) | keep_qb].copy()

    # Charlie's TE discipline (standard scoring only): rounds 6-9, drop every
    # TE unless he's fallen at least TE_HOLD_VALUE_MARGIN picks past his ADP
    # -- a genuine bargain, not just "slightly better value." Dad's exempt.
    if (scoring_mode != "dads" and current_round is not None
            and TE_HOLD_ROUND_START <= current_round <= TE_HOLD_ROUND_END):
        is_te = avail["position"].astype(str).str.upper() == "TE"
        adp = pd.to_numeric(avail.get("adp"), errors="coerce")
        great_deal = adp.notna() & (
            (current_pick if current_pick else 0) - adp >= TE_HOLD_VALUE_MARGIN
        )
        avail = avail[(~is_te) | great_deal].copy()

    if avail.empty:
        return avail

    bye_counts = _roster_bye_counts(pool, my_team)

    def bye_conflicts(row):
        bye = row.get("bye_week")
        if pd.isna(bye):
            return 0
        return bye_counts.get((str(row.get("position")).upper(), int(bye)), 0)

    def need_pts(pos):
        filled, target = have.get(pos, 0), STARTER_TARGETS.get(pos, 0)
        if filled < target:
            return NEED_BONUS                         # flat: uniform at an empty roster
        return -SATURATION_PENALTY * (filled - target + 1)

    # Roster-level risk budget: scale the per-player risk penalty by how much
    # risk I've already rostered (safe team -> lighter, risky team -> heavier).
    my_avg_risk = pd.to_numeric(mine.get("overall_risk"), errors="coerce").mean() if not mine.empty else 0.0
    if pd.isna(my_avg_risk):
        my_avg_risk = 0.0
    risk_mult = float(np.clip(RISK_MULT_MIN + my_avg_risk / RISK_MULT_SCALE, RISK_MULT_MIN, RISK_MULT_MAX))

    # Opportunity cost: how likely each player is to return at my next pick.
    if my_next_pick:
        _adp = pd.to_numeric(avail.get("adp"), errors="coerce")
        survival = 1.0 - 1.0 / (1.0 + np.exp(-SURVIVAL_K * (my_next_pick - _adp)))
        survival = pd.Series(survival, index=avail.index).clip(0.01, 0.99).fillna(0.5)
    else:
        survival = pd.Series(0.0, index=avail.index)

    _pos = avail["position"].astype(str).str.upper()
    injury = avail["injury_risk"].fillna(_NEUTRAL_RISK)
    overall = avail["overall_risk"].fillna(_NEUTRAL_RISK)
    conflicts = avail.apply(bye_conflicts, axis=1)

    avail["value_pts"] = (w["value"] * avail["our_score"]).round(2)
    avail["injury_pts"] = (-w["injury"] * risk_mult * injury).round(2)
    avail["overall_risk_pts"] = (-w["overall_risk"] * risk_mult * overall).round(2)
    avail["bye_conflict_pts"] = (-w["bye_conflict"] * conflicts).round(2)
    avail["need_pts"] = _pos.map(need_pts).round(2)
    avail["survival_pts"] = (-SURVIVAL_WEIGHT * survival).round(2)
    avail["bye_conflicts"] = conflicts

    # Scarcity bonus: only the #1-by-our_score player at each position gets
    # a nonzero gap (to the #2 player at that position); everyone else is 0.
    # (groupby(...).nth() indexes results by original row, not group key --
    # deliberately avoided here since that misaligns the subtraction to NaN.)
    sorted_by_pos = avail.assign(_p=_pos).sort_values("our_score", ascending=False)
    score_lists = sorted_by_pos.groupby("_p")["our_score"].apply(list)
    gap_by_pos = score_lists.apply(lambda s: (s[0] - s[1]) if len(s) > 1 else 0.0)
    leader_idx = sorted_by_pos.groupby("_p").head(1).index
    gap_pts = pd.Series(0.0, index=avail.index)
    gap_pts.loc[leader_idx] = _pos.loc[leader_idx].map(gap_by_pos).to_numpy()
    avail["scarcity_pts"] = (
        (avail["need_pts"] > 0).astype(float)
        * (SCARCITY_BONUS_PER_POINT * gap_pts).clip(upper=SCARCITY_BONUS_MAX)
    ).round(2)

    avail["draft_score"] = (
        avail["value_pts"] + avail["injury_pts"] + avail["overall_risk_pts"]
        + avail["bye_conflict_pts"] + avail["need_pts"] + avail["survival_pts"]
        + avail["scarcity_pts"]
    ).round(2)

    ordered = avail.sort_values("draft_score", ascending=False, na_position="last")
    if diverse:
        ordered = diverse_slate(ordered, top_n)
    ordered = ordered.head(top_n).reset_index(drop=True)
    ordered.insert(0, "draft_rank", range(1, len(ordered) + 1))
    return ordered
