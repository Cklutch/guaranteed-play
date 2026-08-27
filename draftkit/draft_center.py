"""Draft Command Center -- server-rendered HTML for the live-draft page.

Recreates the "Draft Command Center" design handoff (3-column live in-draft
advisor: sync bar, my-team/next-move, highest-tier-available, best-pick
recommendation cards with survival gauges + model notes, tiered board, plan,
and roster needs) in the same server-rendered-HTML style as Home.py -- no
JS, all state computed in Python from the live Sleeper feed.

State is Sleeper-driven: drafted players, whose pick each was, and the pick
count all come from draftkit.sleeper; this module turns that plus the
candidate pool (draftkit.live_draft) into the finished board. Recommendation
ORDER uses our real draft_score (recommend_picks -- our score minus injury /
overall risk / bye conflicts), the risk/bye-weighted ranking the design
flagged as its intended next step; the per-card model notes explain it.
"""

import html
import math

import pandas as pd

from draftkit.live_draft import (
    recommend_picks, diverse_slate, elite_available_for,
    ROSTER_CAPS, ELITE_HOLD_ROUND, NO_ELITE_HOLD_ROUND, POSITION_VALUE_MARGIN,
)

# --- palette (from the design handoff) ----------------------------------
POS_COLORS = {"RB": "#e0947f", "WR": "#7fa8d9", "TE": "#b79ada", "QB": "#dcc06a"}
TIER_COLORS = {1: "#a8c686", 2: "#7fa8d9", 3: "#dcc06a", 4: "#b79ada"}
# Cycled through for tiers beyond the top few, so a ~20-tier board still has
# a color per tier header instead of a wall of grey.
_TIER_CYCLE = ["#a8c686", "#7fa8d9", "#dcc06a", "#b79ada"]
_MUTED = "#8c948f"
# No tier may hold more than this many players -- the flat tail (which the
# gap-based method otherwise dumps 200+ into one column) is chopped into
# several evenly-sized, fully-visible tiers instead.
MAX_TIER_SIZE = 18
_ROSTER_SLOTS = ["RB", "RB", "WR", "WR", "WR", "FLEX", "TE", "QB", "BN", "BN", "BN", "BN"]
# Half-PPR starting build -> positional-need thresholds.
_NEED_MIN = {"RB": 2, "WR": 3, "TE": 1, "QB": 1}
_PLAN = [
    ("R1", ["RB"]), ("R2", ["RB", "WR"]), ("R3", ["WR"]), ("R4", ["WR", "TE"]),
    ("R5", ["RB", "WR"]), ("R6", ["QB", "TE"]), ("R7", ["RB", "WR"]), ("R8", ["QB", "RB"]),
]


def _pos_color(pos):
    return POS_COLORS.get(str(pos).upper(), _MUTED)


def _tier_color(tier):
    if pd.isna(tier):
        return _MUTED
    return _TIER_CYCLE[(int(tier) - 1) % len(_TIER_CYCLE)]


def _surv_color(pct):
    return "#7fc98a" if pct >= 65 else "#e0b45e" if pct >= 35 else "#e08a7f"


def _headshot_bg(player_id):
    if player_id is None or pd.isna(player_id):
        return ""
    sid = str(player_id)
    if sid.endswith(".0"):
        sid = sid[:-2]
    if not sid or sid in ("nan", "None"):
        return ""
    return f"url('https://sleepercdn.com/content/nfl/players/thumb/{sid}.jpg')"


# --- snake-draft math ---------------------------------------------------
def team_on_clock(pick, teams):
    rnd = math.ceil(pick / teams)
    idx = (pick - 1) % teams
    return idx + 1 if rnd % 2 == 1 else teams - idx


def my_pick_numbers(teams, slot, rounds=16):
    return [(r - 1) * teams + (slot if r % 2 == 1 else teams - slot + 1) for r in range(1, rounds + 1)]


def survival_pct(adp, pick):
    """P(player still available at `pick`) from ADP, per the design's
    logistic (k=0.30), clamped 1-99. No pick -> 100."""
    if not pick or adp is None or pd.isna(adp):
        return 100
    p = 1 - 1 / (1 + math.exp(-0.30 * (pick - float(adp))))
    return max(1, min(99, round(p * 100)))


def _assign_global_tiers(scores, cliff_tiers=8, max_tier_size=MAX_TIER_SIZE):
    """Global draft tiers that stay navigable.

    Breaks at BOTH the real value cliffs (the largest score gaps, so the top
    of the board reads as genuine tiers) AND wherever a tier would exceed
    `max_tier_size` -- so the long flat tail of similar-value players is
    split into several evenly-sized tiers rather than one 200+ column."""
    s = pd.to_numeric(pd.Series(scores).reset_index(drop=True), errors="coerce")
    order = s.sort_values(ascending=False, na_position="last")
    idx = list(order.index)
    vals = order.tolist()
    n = len(vals)

    # Positions (in ranked order) where a large value gap warrants a break.
    gaps = sorted(
        range(1, n),
        key=lambda i: (vals[i - 1] - vals[i]) if pd.notna(vals[i - 1]) and pd.notna(vals[i]) else -1,
        reverse=True,
    )
    cliff_positions = set(gaps[: max(0, cliff_tiers - 1)])

    tier_of = {}
    tier, count = 1, 0
    for pos, ix in enumerate(idx):
        if pos > 0 and (pos in cliff_positions or count >= max_tier_size):
            tier += 1
            count = 0
        tier_of[ix] = tier
        count += 1
    return s.index.map(lambda i: tier_of.get(i))


# --- data prep ----------------------------------------------------------
def prepare_pool(pool, drafted_names, drafted_by, my_team, my_next_pick):
    """Enrich the candidate pool with everything the board needs:
    global tier, our/adp rank + value delta, risk band, drafted flags,
    per-player survival, PPG, colors."""
    df = pool.copy().reset_index(drop=True)

    df["tier"] = _assign_global_tiers(df.get("final_score", df["our_score"]))
    df["our_rank"] = df["our_score"].rank(ascending=False, method="first")
    df["adp_rank"] = pd.to_numeric(df.get("adp"), errors="coerce").rank(ascending=True, method="first")
    df["adp_value_delta"] = df["adp_rank"] - df["our_rank"]

    overall = pd.to_numeric(df.get("overall_risk"), errors="coerce")
    df["risk_band"] = overall.apply(lambda v: "low" if pd.isna(v) or v <= 33 else "mid" if v <= 66 else "high")

    drafted_set = set(drafted_names or [])
    df["drafted"] = df["player_name"].isin(drafted_set)
    df["drafted_by"] = df["player_name"].map(lambda n: (drafted_by or {}).get(n))
    df["survival"] = pd.to_numeric(df.get("adp"), errors="coerce").apply(lambda a: survival_pct(a, my_next_pick))
    df["ppg"] = pd.to_numeric(df.get("projection_points"), errors="coerce") / 17.0
    return df


def _needs(mine_df):
    counts = mine_df["position"].astype(str).str.upper().value_counts().to_dict() if not mine_df.empty else {}
    return [pos for pos, mn in _NEED_MIN.items() if counts.get(pos, 0) < mn]


def _dynamic_plan(mine_df, current_round, n_rounds=None):
    """Round-by-round plan that adapts to your actual picks: past rounds show
    what you drafted (checked); the current and future rounds show the next
    positions to target -- unfilled starters first (RB/WR ahead of TE/QB),
    then FLEX, then best-available depth. Returns (label, [positions],
    is_current, is_done) rows."""
    drafted_positions = (
        [str(p).upper() for p in mine_df["position"].tolist()] if not mine_df.empty else []
    )
    have = {}
    for p in drafted_positions:
        have[p] = have.get(p, 0) + 1

    targets = []
    for pos in ("RB", "WR", "TE", "QB"):
        targets += [pos] * max(0, _NEED_MIN.get(pos, 0) - have.get(pos, 0))
    targets.sort(key=lambda p: {"RB": 0, "WR": 1, "TE": 2, "QB": 3}.get(p, 9))
    future = targets + ["FLEX"] + ["BPA"] * 12

    if n_rounds is None:
        n_rounds = max(8, len(drafted_positions) + 3)
    n_rounds = min(n_rounds, 14)

    rows, fi = [], 0
    for r in range(1, n_rounds + 1):
        if r <= len(drafted_positions):
            rows.append((f"R{r}", [drafted_positions[r - 1]], r == current_round, True))
        else:
            pos = future[fi] if fi < len(future) else "BPA"
            fi += 1
            rows.append((f"R{r}", [pos], r == current_round, False))
    return rows


_INJURY_STATUS_LABELS = {
    "IR": "on IR", "INJURED RESERVE": "on IR", "PUP": "on PUP", "OUT": "Out",
    "DOUBTFUL": "Doubtful", "INACTIVE": "Inactive", "DNR": "did not report",
}


def _model_notes(row, avail, needs, mine_df):
    pos = str(row["position"]).upper()
    notes = []

    # Lead with a current-injury flag when there's a serious active
    # designation -- the one thing you most need to see before drafting.
    status = str(row.get("injury_status") or "").strip().upper()
    if status in _INJURY_STATUS_LABELS:
        notes.append((f"Currently {_INJURY_STATUS_LABELS[status]}", status.split()[0], "#e08a7f"))
    same_pos = avail[avail["position"].astype(str).str.upper() == pos].sort_values("our_rank")
    pos_rank = same_pos["player_name"].tolist().index(row["player_name"]) + 1 if row["player_name"] in same_pos["player_name"].values else 0

    if pos in needs:
        notes.append((f"Fills your biggest need: {pos}", "FIT", "#7fc98a"))
    notes.append((f"#{pos_rank} {pos} left on the board", f"{pos}{pos_rank}", _MUTED))

    top_tier = avail["tier"].min() if not avail.empty else None
    if top_tier is not None and row["tier"] == top_tier:
        left_in_tier = avail[(avail["tier"] == top_tier) & (avail["position"].astype(str).str.upper() == pos)]
        if len(left_in_tier) <= 1:
            notes.append((f"Last {pos} in Tier {int(top_tier)}", "LAST", "#e0b45e"))

    delta = row.get("adp_value_delta")
    if pd.notna(delta) and delta > 1.5:
        notes.append((f"Ranked {delta:.0f} spots above ADP", f"+{delta:.0f}", "#7fc98a"))
    elif pd.notna(delta) and delta < -1.5:
        notes.append((f"Reach — {abs(delta):.0f} below ADP", f"{delta:.0f}", "#e08a7f"))

    band = row.get("risk_band")
    if band == "low":
        notes.append(("Low injury & volatility risk", f"RISK {row['overall_risk']:.0f}", "#7fc98a"))
    elif band == "high":
        notes.append(("Elevated risk profile", f"RISK {row['overall_risk']:.0f}", "#e08a7f"))

    if not mine_df.empty:
        same_team = mine_df[(mine_df["team"] == row["team"]) & (mine_df["player_name"] != row["player_name"])]
        if not same_team.empty:
            partner = same_team.iloc[0]
            partner_pos = str(partner["position"]).upper()
            last = str(partner["player_name"]).split()[-1]
            if "QB" in (pos, partner_pos):
                # Real stack: a QB with one of his pass catchers -- correlated upside.
                notes.append((f"Stacks with {last} ({row['team']})", "STACK", "#7fc98a"))
            elif pos in ("WR", "TE") and partner_pos in ("WR", "TE"):
                # Two pass catchers on one offense compete for the same targets.
                notes.append((f"Overlaps {last}'s targets ({row['team']})", "OVERLAP", "#e0b45e"))
        elif pd.notna(row.get("bye_week")):
            clash = (mine_df["bye_week"] == row["bye_week"]).sum()
            if clash >= 2:
                notes.append((f"Bye {int(row['bye_week'])} overlaps {clash} of your picks", "BYE", "#e0b45e"))

    if pd.notna(row.get("ppg")):
        notes.append(("Projected points per game", f"{row['ppg']:.1f} PPG", _MUTED))
    return notes[:5]


# --- rendering ----------------------------------------------------------
def _e(v):
    return html.escape(str(v))


def _rec_card_html(row, surv_pick, mode="rec", model_rank=None):
    """mode: 'best' (gold, best pick), 'rec' (plain), 'compare' (blue)."""
    is_best = mode == "best"
    is_compare = mode == "compare"
    pos = str(row["position"]).upper()
    pc = _pos_color(pos)
    pct = survival_pct(row.get("adp"), surv_pick)
    col = _surv_color(pct)
    label = ("Should be there next pick" if pct >= 65
             else "Coin flip to return" if pct >= 35 else "Likely gone before you pick again")
    sub = f"Your next pick is #{surv_pick}" if surv_pick else "This is your final pick"
    score = float(row.get("our_score") or 0)
    bye = row.get("bye_week")
    bye_txt = f"{int(bye)}" if pd.notna(bye) else "—"
    adp = row.get("adp")
    adp_txt = f"{float(adp):.1f}" if pd.notna(adp) else "—"

    if is_best:
        card_bg = "linear-gradient(165deg,rgba(168,198,134,.14),#141718 60%)"
        card_border = "2px solid #a8c686"
        card_shadow = "0 10px 30px -12px rgba(0,0,0,.6)"
        accent = "#a8c686"
        ribbon_text = "★ BEST PICK"
    elif is_compare:
        card_bg = "linear-gradient(165deg,rgba(127,168,217,.14),#141718 60%)"
        card_border = "2px solid #7fa8d9"
        card_shadow = "0 10px 30px -12px rgba(0,0,0,.6)"
        accent = "#7fa8d9"
        ribbon_text = f"COMPARING · #{model_rank}" if model_rank else "COMPARING"
    else:
        card_bg, card_border, card_shadow = "#141718", "1px solid #262b2e", "none"
        accent = "#a8c686"
        ribbon_text = None
    head_top = "18px" if ribbon_text else "2px"

    breakout_badge = ""
    if row.get("is_breakout_v1"):
        prob = row.get("breakout_probability_v1")
        prob_txt = f" {float(prob) * 100:.0f}%" if pd.notna(prob) else ""
        breakout_badge = (
            f'<span style="display:inline-block;font:700 9px \'IBM Plex Mono\',monospace;'
            f'letter-spacing:.05em;padding:2px 7px;border-radius:6px;margin-left:6px;'
            f'background:rgba(224,180,94,.16);color:#e0b45e;white-space:nowrap" '
            f'title="Backtested breakout-probability model (WR only, research-stage)">🚀 BREAKOUT{prob_txt}</span>'
        )

    ribbon = (
        f'<div style="position:absolute;top:0;left:0;right:0;background:{accent};color:#0d0f10;'
        f"font:900 9.5px 'IBM Plex Mono',monospace;letter-spacing:.1em;text-align:center;padding:4px 0;"
        f'border-radius:14px 14px 0 0">{ribbon_text}</div>' if ribbon_text else ""
    )

    notes_html = "".join(
        f'<div style="display:flex;align-items:baseline;gap:8px;font:400 11px \'IBM Plex Sans\',sans-serif;color:#c9d0c9">'
        f'<span style="width:6px;height:6px;border-radius:2px;background:{ncolor};flex-shrink:0;position:relative;top:1px"></span>'
        f'<span style="flex:1">{_e(ntext)}</span>'
        f'<span style="font:700 10px \'IBM Plex Mono\',monospace;color:{ncolor};white-space:nowrap">{_e(nval)}</span></div>'
        for ntext, nval, ncolor in row["_notes"]
    )

    verdict_html = (
        f'<div style="font:600 11px/1.45 \'IBM Plex Sans\',sans-serif;color:#e6eae4;background:rgba(168,198,134,.1);'
        f'border-radius:9px;padding:9px 11px;margin-top:11px">{_e(row["_verdict"])}</div>' if is_best else ""
    )

    return (
        f'<div style="min-width:262px;flex:1;border-radius:16px;padding:15px;background:{card_bg};border:{card_border};position:relative;box-shadow:{card_shadow}">'
        f'{ribbon}'
        f'<div style="display:flex;align-items:center;gap:11px;margin-top:{head_top}">'
        f'<span style="width:46px;height:46px;border-radius:11px;background:#1d2124 {_headshot_bg(row.get("player_id"))};background-size:cover;background-position:center top;border:1px solid #2a3033;flex-shrink:0"></span>'
        f'<div style="min-width:0">'
        f'<div style="font:800 15px Archivo,sans-serif;color:#f4f6f2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{_e(row["player_name"])}{breakout_badge}</div>'
        f'<div style="font:400 10.5px \'IBM Plex Sans\',sans-serif;color:#8c948f">{_e(row["team"])} · Bye {bye_txt} · ADP {adp_txt}</div>'
        f'</div>'
        f'<span style="margin-left:auto;font:900 15px Archivo,sans-serif;color:{pc}">{_e(pos)}</span>'
        f'</div>'
        f'<div style="margin-top:12px">'
        f'<div style="display:flex;justify-content:space-between;font:600 9px \'IBM Plex Mono\',monospace;letter-spacing:.06em;color:#6f776f;margin-bottom:4px"><span>OUR SCORE</span><span>{score:.1f}/100</span></div>'
        f'<div style="height:7px;background:#1c2022;border-radius:4px"><div style="height:7px;border-radius:4px;background:{accent};width:{score:.1f}%"></div></div>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:12px;margin-top:13px;padding:10px 12px;border-radius:11px;background:#0f1213;border:1px solid #23282b">'
        f'<div style="width:52px;height:52px;border-radius:99px;background:conic-gradient({col} {pct*3.6:.0f}deg, #23282b 0);flex-shrink:0;display:flex;align-items:center;justify-content:center">'
        f'<div style="width:40px;height:40px;border-radius:99px;background:#0f1213;display:flex;align-items:center;justify-content:center;font:800 13px Archivo,sans-serif;color:{col}">{pct}%</div>'
        f'</div>'
        f'<div>'
        f'<div style="font:700 11.5px \'IBM Plex Sans\',sans-serif;color:{col}">{label}</div>'
        f'<div style="font:400 10px/1.4 \'IBM Plex Sans\',sans-serif;color:#8c948f;margin-top:1px">{_e(sub)}</div>'
        f'</div></div>'
        f'<div style="font:600 9px \'IBM Plex Mono\',monospace;letter-spacing:.12em;color:#6f776f;text-transform:uppercase;margin:13px 0 6px">Model notes</div>'
        f'<div style="display:flex;flex-direction:column;gap:5px">{notes_html}</div>'
        f'{verdict_html}'
        f'</div>'
    )


def _verdict_for(row, needs):
    pos = str(row["position"]).upper()
    band = row.get("risk_band")
    if pos in needs:
        lead = f"Your top {pos} on the board and it's your thinnest spot"
    else:
        lead = "The model's best value on the board"
    tail = ("a low-risk anchor." if band == "low"
            else "worth the risk at this price." if band == "high" else "a steady, high-floor pick.")
    return f"{lead} — {tail}"


# --- compare / search ----------------------------------------------------

def _find_compare(query, prepared, full_scored):
    """Fuzzy-match a search query to a player for comparison.
    Returns (prep_row, scored_row, status)."""
    q = query.strip().lower()
    if not q:
        return None, None, None
    mask = prepared["player_name"].str.lower().str.contains(q, na=False, regex=False)
    matches = prepared[mask]
    if matches.empty:
        return None, None, "not_found"
    exact = matches[matches["player_name"].str.lower() == q]
    best = (exact.iloc[0] if not exact.empty
            else matches.sort_values("our_score", ascending=False).iloc[0])
    name = best["player_name"]
    if best.get("drafted"):
        return best, None, "drafted"
    if full_scored is not None and not full_scored.empty:
        scored = full_scored[full_scored["player_name"] == name]
        if not scored.empty:
            return best, scored.iloc[0], "scored"
    return best, None, "filtered"


def _score_breakdown_html(scored_row, best_scored=None):
    """Compact score component breakdown, optionally vs the #1 pick."""
    components = [
        ("Value", "value_pts"),
        ("Injury risk", "injury_pts"),
        ("Overall risk", "overall_risk_pts"),
        ("Position need", "need_pts"),
        ("Positional edge", "scarcity_pts"),
        ("Survival cost", "survival_pts"),
        ("Bye conflicts", "bye_conflict_pts"),
    ]
    has_best = best_scored is not None
    header = ""
    if has_best:
        header = (
            '<div style="display:flex;gap:6px;padding-bottom:8px;border-bottom:1px solid #1a1e20;margin-bottom:4px">'
            '<span style="flex:1"></span>'
            "<span style=\"font:600 8px 'IBM Plex Mono',monospace;letter-spacing:.1em;color:#6f776f;"
            'min-width:50px;text-align:right">THIS</span>'
            "<span style=\"font:600 8px 'IBM Plex Mono',monospace;letter-spacing:.1em;color:#6f776f;"
            'min-width:55px;text-align:right">#1</span></div>'
        )
    rows = ""
    for lbl, col in components:
        v = float(scored_row.get(col) or 0)
        vc = "#7fc98a" if v > 0 else "#e08a7f" if v < -0.5 else _MUTED
        best_cell = ""
        if has_best:
            bv = float(best_scored.get(col) or 0)
            bc = "#7fc98a" if bv > 0 else "#e08a7f" if bv < -0.5 else _MUTED
            best_cell = (
                f"<span style=\"font:500 10px 'IBM Plex Mono',monospace;color:{bc};"
                f'min-width:55px;text-align:right">{bv:+.1f}</span>'
            )
        rows += (
            f'<div style="display:flex;align-items:baseline;gap:6px;padding:3px 0">'
            f"<span style=\"flex:1;font:400 10.5px 'IBM Plex Sans',sans-serif;color:#8c948f\">{lbl}</span>"
            f"<span style=\"font:700 10.5px 'IBM Plex Mono',monospace;color:{vc};"
            f'min-width:50px;text-align:right">{v:+.1f}</span>'
            f'{best_cell}</div>'
        )
    ds = float(scored_row.get("draft_score") or 0)
    best_ds_cell = ""
    if has_best:
        bds = float(best_scored.get("draft_score") or 0)
        best_ds_cell = (
            f'<span style="font:800 12px Archivo,sans-serif;color:#a8c686;'
            f'min-width:55px;text-align:right">{bds:.1f}</span>'
        )
    return (
        '<div style="flex:1;min-width:230px;background:#101314;border:1px solid #23282b;'
        'border-radius:14px;padding:15px">'
        "<div style=\"font:600 9px 'IBM Plex Mono',monospace;letter-spacing:.12em;"
        'color:#7fa8d9;text-transform:uppercase;margin-bottom:10px">Score breakdown</div>'
        f'{header}{rows}'
        '<div style="border-top:1px solid #23282b;margin:8px 0;padding-top:8px;'
        'display:flex;align-items:baseline;gap:6px">'
        "<span style=\"flex:1;font:600 11px 'IBM Plex Sans',sans-serif;color:#f4f6f2\">Draft Score</span>"
        f'<span style="font:800 14px Archivo,sans-serif;color:#7fa8d9;'
        f'min-width:50px;text-align:right">{ds:.1f}</span>'
        f'{best_ds_cell}</div></div>'
    )


def _compare_section_html(prep_row, scored_row, best_scored, surv_pick,
                          avail, needs, mine_df, status, filter_reason=None):
    """Render the player comparison panel (card + score breakdown)."""
    name = _e(prep_row["player_name"])

    if status == "drafted":
        by = prep_row.get("drafted_by")
        by_txt = f"Team {int(by)}" if pd.notna(by) else "another team"
        return (
            '<div style="background:#101314;border:1px solid rgba(127,168,217,.2);'
            'border-radius:14px;padding:15px;margin-bottom:14px">'
            "<div style=\"font:600 9px 'IBM Plex Mono',monospace;letter-spacing:.12em;"
            'color:#7fa8d9;text-transform:uppercase;margin-bottom:8px">Compare</div>'
            f"<div style=\"font:600 13px 'IBM Plex Sans',sans-serif;color:#8c948f\">"
            f'<span style="color:#f4f6f2">{name}</span> was drafted by '
            f'<span style="color:#e08a7f">{_e(by_txt)}</span>.</div></div>'
        )

    if status == "filtered":
        reason = filter_reason or "Removed by a draft discipline rule (QB hold, TE hold, or position cap)."
        return (
            '<div style="background:#101314;border:1px solid rgba(127,168,217,.2);'
            'border-radius:14px;padding:15px;margin-bottom:14px">'
            "<div style=\"font:600 9px 'IBM Plex Mono',monospace;letter-spacing:.12em;"
            'color:#7fa8d9;text-transform:uppercase;margin-bottom:8px">Compare</div>'
            f"<div style=\"font:600 13px 'IBM Plex Sans',sans-serif;color:#e0b45e\">"
            f'{name} is not in the recommendation pool.</div>'
            f"<div style=\"font:400 11px/1.6 'IBM Plex Sans',sans-serif;color:#8c948f;"
            f'margin-top:4px">{_e(reason)}</div></div>'
        )

    row = prep_row.copy()
    row["_notes"] = _model_notes(prep_row, avail, needs, mine_df)
    model_rank = int(scored_row["draft_rank"])
    card = _rec_card_html(row, surv_pick, mode="compare", model_rank=model_rank)
    breakdown = _score_breakdown_html(scored_row, best_scored)
    return (
        '<div style="background:rgba(127,168,217,.04);border:1px solid rgba(127,168,217,.2);'
        'border-radius:16px;padding:16px;margin-bottom:14px">'
        f'<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:start">'
        f'{card}{breakdown}</div></div>'
    )


def render_command_center(pool, drafted, my_team, drafted_by, num_picks,
                          num_teams, my_slot, scoring_label, sleeper_code,
                          complete=False, compare_player=None):
    """Return the Draft Command Center HTML (a single string for
    st.markdown). Everything is computed from the arguments -- no JS.
    `complete` (Sleeper draft status == "complete") swaps the recommendation
    area for a done banner and marks the clock finished."""
    drafted = drafted or []
    my_team = my_team or []
    teams = int(num_teams or 12)
    slot = int(my_slot) if my_slot else None

    current_pick = (num_picks or 0) + 1
    rnd = math.ceil(current_pick / teams)
    on_clock_team = team_on_clock(current_pick, teams)
    i_am_on_clock = slot is not None and on_clock_team == slot
    my_picks = my_pick_numbers(teams, slot) if slot else []
    my_pick_now = next((p for p in my_picks if p >= current_pick), None)
    my_next = next((p for p in my_picks if p > current_pick), None)

    mine_df = pool[pool["player_name"].isin(my_team)].copy()
    needs = _needs(mine_df)
    need = needs[0] if needs else "BPA"

    prepared = prepare_pool(pool, drafted, drafted_by, my_team, my_next)
    avail = prepared[~prepared["drafted"]].copy()

    is_manual = sleeper_code == "Manual"
    source_label = "MANUAL" if is_manual else f"SLEEPER · {sleeper_code}"
    if complete:
        sync = dict(text=f"COMPLETE · {source_label}" if sleeper_code else "DRAFT COMPLETE",
                    tc="#7fc98a", dot="#7fc98a", glow="0 0 7px #7fc98a")
    elif sleeper_code:
        sync = dict(text=f"LIVE · {source_label}", tc="#7fc98a", dot="#7fc98a", glow="0 0 7px #7fc98a")
    else:
        sync = dict(text="NOT SYNCED · demo", tc="#e0b45e", dot="#e0b45e", glow="none")
    clock_color = "#a8c686" if i_am_on_clock else "#f4f6f2"
    on_clock_txt = "Done" if complete else ("You" if i_am_on_clock else f"Team {on_clock_team}")
    pick_txt = "—" if complete else current_pick
    need_color = _MUTED if (need == "BPA" or complete) else _pos_color(need)

    def _stat(lbl, val, color):
        return (f'<div style="text-align:center"><div style="font:600 8.5px \'IBM Plex Mono\',monospace;'
                f'letter-spacing:.14em;color:#6f776f">{lbl}</div>'
                f'<div style="font:800 15px Archivo,sans-serif;color:{color}">{_e(val)}</div></div>')

    topbar = (
        '<div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;padding:11px 4px 14px;border-bottom:1px solid #23282b;margin-bottom:14px">'
        '<a href="/" target="_self" style="font:700 11px \'IBM Plex Mono\',monospace;color:#a8c686;'
        'text-decoration:none;padding:5px 12px;border-radius:7px;background:rgba(168,198,134,.08);'
        'border:1px solid rgba(168,198,134,.2);white-space:nowrap">'
        '← Rankings</a>'
        f'<span style="display:inline-flex;align-items:center;gap:7px;font:700 11px \'IBM Plex Mono\',monospace;letter-spacing:.08em;color:{sync["tc"]}">'
        f'<span style="width:7px;height:7px;border-radius:99px;background:{sync["dot"]};box-shadow:{sync["glow"]}"></span>{_e(sync["text"])}</span>'
        '<div style="display:flex;gap:16px">'
        f'{_stat("ROUND", "—" if complete else rnd, "#f4f6f2")}{_stat("PICK", pick_txt, "#f4f6f2")}{_stat("ON CLOCK", on_clock_txt, clock_color)}{_stat("NEED", "—" if complete else need, need_color)}{_stat("SCORING", scoring_label, "#f4f6f2")}'
        '</div></div>'
    )

    if complete:
        next_move = "Draft complete — your roster is set. Review your team below."
    elif i_am_on_clock:
        next_move = (f"You're on the clock at #{current_pick}. "
                     + (f"{needs[0]} is your thinnest spot — the best pick is highlighted in gold."
                        if needs else "Starters are set — take the highest-scored player available."))
    elif my_pick_now:
        next_move = f"Your next pick is #{my_pick_now}. " + (f"{needs[0]} is your priority." if needs else "Best available.")
    elif sleeper_code == "Manual":
        next_move = "Set your draft slot above to track your team, needs, and byes live."
    else:
        next_move = "Paste your Sleeper draft link and set your slot to track your team live."

    mine_ordered = mine_df.reset_index(drop=True)
    roster_rows = ""
    for i, tmpl in enumerate(_ROSTER_SLOTS):
        if i < len(mine_ordered):
            f = mine_ordered.iloc[i]
            fp = str(f["position"]).upper()
            roster_rows += (f'<div style="display:flex;align-items:center;gap:9px;padding:6px 8px;border-radius:8px;background:#141718;opacity:1">'
                            f'<span style="font:700 9px \'IBM Plex Mono\',monospace;padding:2px 6px;border-radius:5px;color:#0d0f10;background:{_pos_color(fp)};min-width:30px;text-align:center">{_e(fp)}</span>'
                            f'<span style="font:500 12px \'IBM Plex Sans\',sans-serif;color:#e6eae4">{_e(f["player_name"])}</span></div>')
        else:
            roster_rows += (f'<div style="display:flex;align-items:center;gap:9px;padding:6px 8px;border-radius:8px;background:#141718;opacity:.55">'
                            f'<span style="font:700 9px \'IBM Plex Mono\',monospace;padding:2px 6px;border-radius:5px;color:#0d0f10;background:#3a3f42;min-width:30px;text-align:center">{_e(tmpl)}</span>'
                            f'<span style="font:500 12px \'IBM Plex Sans\',sans-serif;color:#6f776f;font-style:italic">open</span></div>')

    left = (
        '<div style="display:flex;flex-direction:column;gap:12px">'
        '<div style="background:#101314;border:1px solid #23282b;border-radius:14px;padding:15px">'
        '<div style="font:600 9px \'IBM Plex Mono\',monospace;letter-spacing:.14em;color:#a8c686;text-transform:uppercase">Your next move</div>'
        f'<div style="font:600 13px/1.5 \'IBM Plex Sans\',sans-serif;color:#e6eae4;margin-top:8px">{_e(next_move)}</div>'
        '</div>'
        '<div style="background:#101314;border:1px solid #23282b;border-radius:14px;padding:14px">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px">'
        '<div style="font:700 10.5px \'IBM Plex Mono\',monospace;letter-spacing:.1em;color:#8c948f;text-transform:uppercase">My Team</div>'
        f'<div style="font:500 10px \'IBM Plex Mono\',monospace;color:#6f776f">{len(mine_ordered)}/{len(_ROSTER_SLOTS)}</div>'
        '</div>'
        f'<div style="display:flex;flex-direction:column;gap:5px">{roster_rows}</div>'
        '</div></div>'
    )

    if not avail.empty:
        top_tier = int(avail["tier"].min())
        tt_players = avail[avail["tier"] == top_tier]
        counts_html = ""
        for pos in ("RB", "WR", "TE", "QB"):
            n = int((tt_players["position"].astype(str).str.upper() == pos).sum())
            c = "#4a5054" if n == 0 else _pos_color(pos)
            counts_html += (f'<div style="text-align:center;min-width:52px;padding:6px 10px;border-radius:9px;background:#141718;border:1px solid #23282b">'
                            f'<div style="font:800 17px Archivo,sans-serif;color:{c}">{n}</div>'
                            f'<div style="font:600 9px \'IBM Plex Mono\',monospace;letter-spacing:.1em;color:#6f776f;margin-top:1px">{pos}</div></div>')
        tier_badge, tier_bg = f"T{top_tier}", _tier_color(top_tier)
        tier_title = f"Tier {top_tier}"
        tier_sub = f"{len(tt_players)} player{'' if len(tt_players)==1 else 's'} left before the tier breaks"
    else:
        counts_html, tier_badge, tier_bg, tier_title, tier_sub = "", "—", "#3a3f42", "none", "Draft complete"

    tier_avail = (
        '<div style="background:#101314;border:1px solid #23282b;border-radius:14px;padding:13px 16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
        f'<div style="width:40px;height:40px;border-radius:10px;background:{tier_bg};color:#0d0f10;font:900 16px Archivo,sans-serif;display:flex;align-items:center;justify-content:center;flex-shrink:0">{tier_badge}</div>'
        f'<div><div style="font:700 12.5px \'IBM Plex Sans\',sans-serif;color:#f4f6f2">Highest tier available · {tier_title}</div>'
        f'<div style="font:400 11px \'IBM Plex Sans\',sans-serif;color:#8c948f;margin-top:1px">{_e(tier_sub)}</div></div>'
        f'<div style="display:flex;gap:8px;margin-left:auto;flex-wrap:wrap">{counts_html}</div></div>'
    )

    if complete:
        recs_html = (
            '<div style="background:linear-gradient(165deg,rgba(168,198,134,.12),#101314 70%);'
            'border:1px solid #1f5c33;border-radius:16px;padding:22px 20px;text-align:center">'
            '<div style="font:900 22px Archivo,sans-serif;color:#a8c686">✓ Draft complete</div>'
            f'<div style="font:400 12.5px/1.6 \'IBM Plex Sans\',sans-serif;color:#c9d0c9;margin-top:6px">'
            f'All {num_picks} picks are in. You drafted {len(my_team)} player'
            f'{"" if len(my_team)==1 else "s"} — your roster is on the left'
            + (f', and you still had {", ".join(needs)} open.' if needs else ' with every starting slot filled.')
            + ' Good luck this season.</div></div>'
        )
    else:
        surv_pick = my_next if i_am_on_clock else my_pick_now
        rec_label = ("You're on the clock · best picks" if i_am_on_clock
                     else f"Best available · your pick #{my_pick_now or '—'}")
        scoring_mode = "dads" if "Dad" in str(scoring_label) else "standard"

        if compare_player:
            full_scored = recommend_picks(
                pool, drafted_players=drafted, my_team=my_team, top_n=250,
                current_round=rnd, current_pick=current_pick, my_next_pick=surv_pick,
                scoring_mode=scoring_mode, diverse=False,
            )
            ranked = diverse_slate(full_scored, 4)
        else:
            ranked = recommend_picks(
                pool, drafted_players=drafted, my_team=my_team, top_n=4,
                current_round=rnd, current_pick=current_pick, my_next_pick=surv_pick,
                scoring_mode=scoring_mode, diverse=True,
            )
            full_scored = None

        compare_html = ""
        if compare_player:
            prep_row, scored_row, cmp_status = _find_compare(compare_player, prepared, full_scored)
            if cmp_status == "not_found":
                compare_html = (
                    '<div style="background:#101314;border:1px solid rgba(127,168,217,.2);'
                    'border-radius:14px;padding:15px;margin-bottom:14px">'
                    "<div style=\"font:600 9px 'IBM Plex Mono',monospace;letter-spacing:.12em;"
                    'color:#7fa8d9;text-transform:uppercase;margin-bottom:8px">Compare</div>'
                    f"<div style=\"font:400 12px 'IBM Plex Sans',sans-serif;color:#8c948f\">"
                    f'No player found matching &ldquo;{_e(compare_player)}&rdquo;.</div></div>'
                )
            elif cmp_status:
                filter_reason = None
                if cmp_status == "filtered":
                    cp = str(prep_row["position"]).upper()
                    mine_pos = mine_df["position"].astype(str).str.upper().value_counts().to_dict() if not mine_df.empty else {}
                    if cp in ROSTER_CAPS and mine_pos.get(cp, 0) >= ROSTER_CAPS[cp]:
                        filter_reason = f"You already have a {cp} — position cap reached."
                    elif cp in ("QB", "TE") and scoring_mode != "dads" and rnd:
                        elite_left = elite_available_for(pool, avail, cp)
                        hold_round = ELITE_HOLD_ROUND if elite_left else NO_ELITE_HOLD_ROUND
                        if rnd < hold_round:
                            filter_reason = (
                                (f"{cp} hold active until round {hold_round} (standard scoring) — "
                                 f"an elite {cp} is still on the board, only at value (fallen to/past his ADP).")
                                if elite_left else
                                (f"No elite {cp} left on the board — hold active until round {hold_round} "
                                 f"(standard scoring). Only a {cp} fallen {POSITION_VALUE_MARGIN}+ picks "
                                 f"past ADP breaks through.")
                            )
                best_scored = full_scored.iloc[0] if full_scored is not None and not full_scored.empty else None
                compare_html = _compare_section_html(
                    prep_row, scored_row, best_scored, surv_pick, avail, needs, mine_df,
                    cmp_status, filter_reason,
                )

        rec_cards = ""
        if ranked.empty:
            rec_cards = '<div style="color:#8c948f;font:400 12px \'IBM Plex Sans\',sans-serif">No available players.</div>'
        else:
            prep_by_name = prepared.set_index("player_name")
            for i, name in enumerate(ranked["player_name"].tolist()):
                if name not in prep_by_name.index:
                    continue
                row = prep_by_name.loc[name].copy()
                row["player_name"] = name
                row["_notes"] = _model_notes(row, avail, needs, mine_df)
                row["_verdict"] = _verdict_for(row, needs)
                rec_cards += _rec_card_html(row, surv_pick, mode="best" if i == 0 else "rec")

        recs_html = (
            '<div>'
            f'{compare_html}'
            f'<div style="font:600 10.5px \'IBM Plex Mono\',monospace;letter-spacing:.13em;color:#8c948f;text-transform:uppercase;margin-bottom:9px">{_e(rec_label)}</div>'
            f'<div class="dcc-scroll" style="display:flex;gap:12px;overflow-x:auto;padding-bottom:6px">{rec_cards}</div></div>'
        )

    legend = ('<div style="display:flex;align-items:center;gap:14px;margin-bottom:9px;font:400 10px \'IBM Plex Sans\',sans-serif;color:#6f776f">'
              '<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:9px;height:3px;border-radius:2px;background:#7fc98a"></span>safe to wait</span>'
              '<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:9px;height:3px;border-radius:2px;background:#e0b45e"></span>coin flip</span>'
              '<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:9px;height:3px;border-radius:2px;background:#e08a7f"></span>gone before your pick</span></div>')

    cols_html = ""
    for tier in sorted(prepared["tier"].dropna().unique()):
        tcol = prepared[prepared["tier"] == tier].sort_values("our_rank")
        if tcol.empty:
            continue
        remain = int((~tcol["drafted"]).sum())
        # Tiers are size-capped (MAX_TIER_SIZE), so every player in a tier is
        # shown -- no hidden "+N more", nothing to scroll for out of reach.
        tiles = ""
        for _, p in tcol.iterrows():
            pos = str(p["position"]).upper()
            drafted_flag = bool(p["drafted"])
            pct = int(p["survival"])
            edge = "#2a3033" if drafted_flag else _surv_color(pct)
            name = str(p["player_name"])
            disp = name  # full name -- the full-width board has room to wrap
            adp = p.get("adp"); adp_txt = f"{float(adp):.1f}" if pd.notna(adp) else "—"
            bye = p.get("bye_week"); bye_txt = f"{int(bye)}" if pd.notna(bye) else "—"
            delta = p.get("adp_value_delta")
            if pd.notna(delta) and delta > 1.5:
                vlabel, vbg, vfg = "VALUE", "rgba(127,201,138,.16)", "#7fc98a"
            elif pd.notna(delta) and delta < -1.5:
                vlabel, vbg, vfg = "REACH", "rgba(224,138,127,.16)", "#e08a7f"
            else:
                vlabel, vbg, vfg = "FAIR", "#1a1e20", _MUTED
            if drafted_flag:
                fore, fore_color = "", "#6f776f"
            else:
                fore = (f"gone by ~{round(float(adp))}" if pct < 35 and pd.notna(adp)
                        else "borderline" if pct < 65 else "safe to wait")
                fore_color = "#e08a7f" if pct < 35 else "#e0b45e" if pct < 65 else "#7fc98a"
            ribbon = ""
            if drafted_flag:
                by = p.get("drafted_by")
                is_me = slot is not None and by == slot
                rb = "YOU" if is_me else (f"T{int(by)}" if pd.notna(by) else "OUT")
                rbg = "rgba(168,198,134,.25)" if is_me else "rgba(224,138,127,.18)"
                rfg = "#a8c686" if is_me else "#e08a7f"
                ribbon = (f'<span style="position:absolute;top:7px;right:8px;font:800 8px \'IBM Plex Mono\',monospace;'
                          f'padding:1px 6px;border-radius:5px;background:{rbg};color:{rfg}">{rb}</span>')
            name_color = "#8c948f" if drafted_flag else "#f4f6f2"
            op = ".4" if drafted_flag else "1"
            tiles += (
                f'<div style="background:#101314;border:1px solid #23282b;border-left:3px solid {edge};border-radius:9px;padding:8px 10px;opacity:{op};position:relative">'
                f'{ribbon}'
                '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:6px">'
                f'<span style="font:600 12px Archivo,sans-serif;color:{name_color}">{_e(disp)}</span>'
                f'<span style="font:800 8.5px \'IBM Plex Mono\',monospace;padding:1px 6px;border-radius:4px;color:#0d0f10;background:{_pos_color(pos)}">{_e(pos)}</span></div>'
                f'<div style="font:400 9.5px \'IBM Plex Sans\',sans-serif;color:#7c847e;margin-top:3px">{_e(p["team"])} · ADP {adp_txt} · Bye {bye_txt}</div>'
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px">'
                f'<span style="font:700 8.5px \'IBM Plex Mono\',monospace;padding:1px 6px;border-radius:5px;background:{vbg};color:{vfg}">{vlabel}</span>'
                f'<span style="font:500 9px \'IBM Plex Sans\',sans-serif;color:{fore_color}">{_e(fore)}</span></div></div>'
            )
        cols_html += (
            '<div style="min-width:198px;flex-shrink:0">'
            '<div style="display:flex;justify-content:space-between;align-items:baseline;padding:0 2px 8px">'
            f'<span style="font:700 10.5px \'IBM Plex Mono\',monospace;letter-spacing:.13em;color:{_tier_color(tier)};text-transform:uppercase">Tier {int(tier)}</span>'
            f'<span style="font:500 9.5px \'IBM Plex Mono\',monospace;color:#6f776f">{remain} left</span></div>'
            f'<div style="display:flex;flex-direction:column;gap:7px">{tiles}</div></div>'
        )

    board_html = (
        '<div>'
        '<div style="font:700 14px Archivo,sans-serif;color:#f4f6f2;margin-bottom:10px">Tiered board</div>'
        f'{legend}'
        f'<div class="dcc-scroll" style="display:flex;gap:10px;overflow-x:auto;padding-bottom:10px">{cols_html}</div></div>'
    )

    center = f'<div style="display:flex;flex-direction:column;gap:16px;min-width:0">{tier_avail}{recs_html}</div>'

    plan = _dynamic_plan(mine_df, rnd)
    plan_rows = ""
    for label, positions, active, done in plan:
        bg = "rgba(168,198,134,.1)" if active else "#141718"
        bd = "#1f5c33" if active else "#23282b"
        rc = "#a8c686" if active else "#6f776f"
        # Rounds already drafted show what you took (dimmed, checked); the
        # current + future rounds show adaptive targets from remaining needs.
        chip_bg = (lambda p: "#3a3f42" if done and p in ("FLEX", "BPA") else _pos_color(p) if p not in ("FLEX", "BPA") else "#3a3f42")
        chips = "".join(
            f'<span style="font:700 9px \'IBM Plex Mono\',monospace;padding:1px 6px;border-radius:5px;'
            f'color:#0d0f10;background:{chip_bg(p)};opacity:{".55" if done else "1"}">{p}</span>'
            for p in positions
        )
        note = "this round" if active else ("✓" if done else "")
        note_color = "#8c948f" if active else "#5f8f5f" if done else "#8c948f"
        plan_rows += (f'<div style="display:flex;align-items:center;gap:9px;padding:7px 9px;border-radius:9px;background:{bg};border:1px solid {bd}">'
                      f'<span style="font:700 9.5px \'IBM Plex Mono\',monospace;color:{rc};min-width:20px">{label}</span>'
                      f'<div style="display:flex;gap:4px;flex-wrap:wrap">{chips}</div>'
                      f'<span style="font:400 10px \'IBM Plex Sans\',sans-serif;color:{note_color};margin-left:auto">{note}</span></div>')

    need_chips = ("".join(f'<span style="font:700 10px \'IBM Plex Mono\',monospace;padding:3px 9px;border-radius:6px;color:#0d0f10;background:{_pos_color(p)}">{p}</span>' for p in needs)
                  if needs else '<span style="font:700 10px \'IBM Plex Mono\',monospace;padding:3px 9px;border-radius:6px;color:#0d0f10;background:#3a3f42">SET</span>')
    needs_text = (f"Still need {', '.join(needs)}. Recommendations weight these first, then best available."
                  if needs else "Starting lineup is covered — draft best available and upside.")

    right = (
        '<div style="display:flex;flex-direction:column;gap:12px">'
        '<div style="background:#101314;border:1px solid #23282b;border-radius:14px;padding:15px">'
        '<div style="font:800 13px Archivo,sans-serif;color:#a8c686">The plan</div>'
        '<div style="font:400 10.5px/1.5 \'IBM Plex Sans\',sans-serif;color:#8c948f;margin-top:3px">Round-by-round targets from your roster build. Adapts to your needs.</div>'
        f'<div style="display:flex;flex-direction:column;gap:6px;margin-top:12px">{plan_rows}</div></div>'
        '<div style="background:#101314;border:1px solid #23282b;border-radius:14px;padding:15px">'
        '<div style="font:700 10.5px \'IBM Plex Mono\',monospace;letter-spacing:.1em;color:#8c948f;text-transform:uppercase;margin-bottom:9px">Roster needs</div>'
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px">{need_chips}</div>'
        f'<div style="font:400 11px/1.7 \'IBM Plex Sans\',sans-serif;color:#8c948f">{_e(needs_text)}</div></div></div>'
    )

    grid = (f'<div style="display:grid;grid-template-columns:236px minmax(0,1fr) 250px;gap:16px;'
            f'align-items:start">{left}{center}{right}</div>')

    # Tiered board spans the full width below the 3-column grid.
    return f'<div class="dcc-scroll" style="color:#eef1ec">{topbar}{grid}<div style="margin-top:16px">{board_html}</div></div>'
